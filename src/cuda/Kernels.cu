#include "Kernels.cuh"

#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <cuda_runtime.h>
#include <surface_indirect_functions.h>
#include <surface_types.h>
#include "CudaMacros.h"

using namespace ggg;

namespace
{
    struct VertexShaderOut
    {
        Vec3f n{};
        Vec2f uv{};

        __device__ static VertexShaderOut Interpolate(const std::array<VertexShaderOut, 3>& vals, const Vec3f& weights)
        {
            return [&] <std::size_t... Is> (std::index_sequence<Is...>) -> VertexShaderOut
            {
                return VertexShaderOut
                {
                    .n = ((vals[Is].n * weights[Is]) + ...),
                    .uv = ((vals[Is].uv * weights[Is]) + ...)
                };
            } (std::make_index_sequence<3>());
        }
    };

    __device__ Vec4f VertexShader(const Vertex& vert, const cuda::UniformBuffer& uniformBuffer, VertexShaderOut& out)
    {
        Vec4f pos { vert.p.x(), vert.p.y(), vert.p.z(), 1.f };

        const Vec4f viewPos = uniformBuffer.m_viewMatrix * uniformBuffer.m_modelMatrix * pos;

        Mat3f normalMatrix = Mat3f{
            uniformBuffer.m_modelMatrix[0].xyz(),
            uniformBuffer.m_modelMatrix[1].xyz(),
            uniformBuffer.m_modelMatrix[2].xyz()
        }.Inverse().Transpose();

        out.n = (normalMatrix * vert.n).Normalized();
        out.uv = vert.uv;

        return uniformBuffer.m_projectionMatrix * viewPos;
    }

    __device__ Vec4f FragmentShader(const VertexShaderOut& in)
    {
        const Vec3f lightDir{0.3508772f, 0.9022557f, 0.2506266f}; // normalize({0.35,0.9,0.25})
        const float ndotl = fmaxf(0.0f, Dot(in.n, lightDir));
        return Vec4f{0.20f + ndotl * (0.70f), 0.22f + ndotl * (0.60f), 0.25f + ndotl * (0.50f), 1.0f};
    }

    __device__ __forceinline__ uchar4 ToRGBA8(const Vec4f& c)
    {
        const unsigned char r = static_cast<unsigned char>(fminf(fmaxf(c.x(), 0.0f), 1.0f) * 255.0f);
        const unsigned char g = static_cast<unsigned char>(fminf(fmaxf(c.y(), 0.0f), 1.0f) * 255.0f);
        const unsigned char b = static_cast<unsigned char>(fminf(fmaxf(c.z(), 0.0f), 1.0f) * 255.0f);
        const unsigned char a = static_cast<unsigned char>(fminf(fmaxf(c.w(), 0.0f), 1.0f) * 255.0f);
        return make_uchar4(r, g, b, a);
    }

    // Grid size: (ceil(surfaceSize.x / 16), ceil(surfaceSize.y / 16), 1); Block size: (16, 16, 1)
    __global__ void ClearKernel(cudaSurfaceObject_t renderSurface,
                                Vec2u surfaceSize,
                                std::uint32_t* depthBuffer)
    {
        const unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x >= surfaceSize.x() || y >= surfaceSize.y())
        {
            return;
        }
        static constexpr std::uint32_t FLT_MAX_BITS = 0x7f7fffffU;
        depthBuffer[y * surfaceSize.x() + x] = FLT_MAX_BITS;
        surf2Dwrite(ToRGBA8(Vec4f{0.f, 0.f, 0.f, 1.0f}), renderSurface, static_cast<int>(x * sizeof(uchar4)), static_cast<int>(y));
    }

    // Grid size: (ceil(vertex_count / 256), 1, 1); Block size: (256, 1, 1)
    __global__ void TransformVerticesKernel(const Vertex* cudaVertexBuffer,
                                            std::size_t vertexCount,
                                            const cuda::UniformBuffer* uniformBuffer,
                                            std::pair<VertexShaderOut, Vec4f>* transformedVertexCache)
    {
        std::size_t vertexIdx = blockIdx.x * blockDim.x + threadIdx.x;
        if (vertexIdx >= vertexCount)
        {
            return;
        }
        transformedVertexCache[vertexIdx].second = VertexShader(
            cudaVertexBuffer[vertexIdx], *uniformBuffer, transformedVertexCache[vertexIdx].first
        );
    }

    // Grid size: (triangle_count, 1, 1); Block size: (16, 16, 1)
    __global__ void RasterizeAndShadeKernel(cudaSurfaceObject_t renderSurface,
                                            std::uint32_t* depthBuffer,
                                            Vec2u surfaceSize,
                                            const std::pair<VertexShaderOut, Vec4f>* transformedVertexCache,
                                            const std::uint32_t* cudaIndexBuffer,
                                            std::size_t indexCount)
    {
        const std::size_t triIdx = blockIdx.x;
        if (triIdx >= indexCount / 3)
        {
            return;
        }
        const std::uint32_t i0 = cudaIndexBuffer[triIdx * 3 + 0];
        const std::uint32_t i1 = cudaIndexBuffer[triIdx * 3 + 1];
        const std::uint32_t i2 = cudaIndexBuffer[triIdx * 3 + 2];

        const std::array<VertexShaderOut, 3> vsOut{
            transformedVertexCache[i0].first,
            transformedVertexCache[i1].first,
            transformedVertexCache[i2].first
        };

        // clip space [(-w, -w) (bottom left), (w, w) (top right)]
        const std::array<Vec4f, 3> clipSpace{
            transformedVertexCache[i0].second,
            transformedVertexCache[i1].second,
            transformedVertexCache[i2].second
        };
        for (const Vec4f& c : clipSpace)
        {
            if (c.w() == 0.f)
            {
                return;
            }
        }

        // NDC [(-1, -1, -inf) (bottom left), (1, 1, inf) (top right)]
        std::array<Vec3f, 3> ndc{};
        for (std::size_t i = 0; i < 3; ++i)
        {
            ndc[i] = clipSpace[i].xyz() / clipSpace[i].w();
        }

        // pixel space [(0, height-1) (bottom left), (width-1, 0) (top right)]
        std::array<Vec2f, 3> pixelSpace{};
        for (std::size_t i = 0; i < 3; ++i)
        {
            pixelSpace[i] = (ndc[i].xy() * Vec2f{0.5f, -0.5f} + Vec2f{0.5f, 0.5f}) * Vec2f{surfaceSize.x() - 1.0f, surfaceSize.y() - 1.0f};
        }

        int minX = static_cast<int>(floorf(fminf(pixelSpace[0].x(), fminf(pixelSpace[1].x(), pixelSpace[2].x()))));
        int minY = static_cast<int>(floorf(fminf(pixelSpace[0].y(), fminf(pixelSpace[1].y(), pixelSpace[2].y()))));
        int maxX = static_cast<int>(ceilf(fmaxf(pixelSpace[0].x(), fmaxf(pixelSpace[1].x(), pixelSpace[2].x()))));
        int maxY = static_cast<int>(ceilf(fmaxf(pixelSpace[0].y(), fmaxf(pixelSpace[1].y(), pixelSpace[2].y()))));
        minX = max(0, min(minX, static_cast<int>(surfaceSize.x()) - 1));
        minY = max(0, min(minY, static_cast<int>(surfaceSize.y()) - 1));
        maxX = max(0, min(maxX, static_cast<int>(surfaceSize.x()) - 1));
        maxY = max(0, min(maxY, static_cast<int>(surfaceSize.y()) - 1));

        const float area = Cross(pixelSpace[2] - pixelSpace[0], pixelSpace[1] - pixelSpace[0]).z();
        if (area == 0.0f)
        {
            return;
        }
        const float invArea = 1.0f / area;

        // Edge functions for barycentric weights:
        // w = edgeMat * Vec3f{px, py, 1}, where each row is an edge equation (A*x + B*y + C).
        const Mat3f edgeMat{
            // A coefficients
            Vec3f{
                pixelSpace[2].y() - pixelSpace[1].y(),
                pixelSpace[0].y() - pixelSpace[2].y(),
                pixelSpace[1].y() - pixelSpace[0].y()
            },
            // B coefficients
            Vec3f{
                pixelSpace[1].x() - pixelSpace[2].x(),
                pixelSpace[2].x() - pixelSpace[0].x(),
                pixelSpace[0].x() - pixelSpace[1].x()
            },
            // C coefficients
            Vec3f{
                pixelSpace[1].y() * pixelSpace[2].x() - pixelSpace[1].x() * pixelSpace[2].y(),
                pixelSpace[2].y() * pixelSpace[0].x() - pixelSpace[2].x() * pixelSpace[0].y(),
                pixelSpace[0].y() * pixelSpace[1].x() - pixelSpace[0].x() * pixelSpace[1].y()
            }
        };
        const Vec3f stepX = edgeMat[0] * (static_cast<float>(blockDim.x));
        const Vec3f negNdcZ{-ndc[0].z(), -ndc[1].z(), -ndc[2].z()};

        for (int y = minY + static_cast<int>(threadIdx.y); y <= maxY; y += static_cast<int>(blockDim.y))
        {
            const float py = static_cast<float>(y) + 0.5f;
            int x = minX + static_cast<int>(threadIdx.x);
            float px = static_cast<float>(x) + 0.5f;

            Vec3f wE = edgeMat * Vec3f{px, py, 1.0f};

            for (; x <= maxX; x += static_cast<int>(blockDim.x))
            {
                if (std::signbit(area) == std::signbit(wE.x()) &&
                    std::signbit(area) == std::signbit(wE.y()) &&
                    std::signbit(area) == std::signbit(wE.z()))
                {
                    const Vec3f bary = wE * invArea;
                    const float viewZ = Dot(bary, negNdcZ);
                    if (viewZ > 0.0f)
                    {
                        const int depthIdx = y * static_cast<int>(surfaceSize.x()) + x;
                        const unsigned int newDepthBits = __float_as_uint(viewZ);
                        const unsigned int oldDepthBits = atomicMin(depthBuffer + depthIdx, newDepthBits);
                        if (newDepthBits < oldDepthBits)
                        {
                            const Vec4f color = FragmentShader(VertexShaderOut::Interpolate(vsOut, bary));
                            surf2Dwrite(ToRGBA8(color), renderSurface, x * static_cast<int>(sizeof(uchar4)), y);
                        }
                    }
                }

                wE += stepX;
            }
        }
    }

    cuda::UniformBuffer* g_uniformBuffer{};
    inline constexpr std::size_t MAX_VERTICES_PER_DRAW = 1 << 20;
    std::pair<VertexShaderOut, Vec4f>* g_transformedVertexCache{};
}

__host__ void cuda::InitCudaGraphics()
{
    CHECK_CUDA(cudaMalloc(&g_uniformBuffer, sizeof(UniformBuffer)));
    CHECK_CUDA(cudaMalloc(&g_transformedVertexCache, sizeof(g_transformedVertexCache[0]) * MAX_VERTICES_PER_DRAW));
}

__host__ void cuda::StopCudaGraphics()
{
    CHECK_CUDA(cudaFree(g_transformedVertexCache));
    g_transformedVertexCache = nullptr;
    CHECK_CUDA(cudaFree(g_uniformBuffer));
    g_uniformBuffer = nullptr;
}

__host__ void cuda::SetUniformBuffer(const UniformBuffer& uniform)
{
    CHECK_CUDA(cudaMemcpy(g_uniformBuffer, &uniform, sizeof(UniformBuffer), cudaMemcpyHostToDevice));
}

__host__ void cuda::LaunchClear(cudaSurfaceObject_t renderSurface,
                                CudaBuffer<std::uint32_t>& depthBuffer,
                                Vec2u surfaceSize,
                                cudaStream_t stream)
{
    const dim3 block(16, 16, 1);
    const dim3 grid((surfaceSize.x() + block.x - 1) / block.x, (surfaceSize.y() + block.y - 1) / block.y, 1);
    ClearKernel<<<grid, block, 0, stream>>>(renderSurface, surfaceSize, depthBuffer.GetGpuPtr());
    CHECK_CUDA(cudaGetLastError());
}

__host__ void cuda::LaunchDraw(cudaSurfaceObject_t renderSurface,
                               CudaBuffer<std::uint32_t>& depthBuffer,
                               Vec2u surfaceSize,
                               cudaStream_t stream,
                               const CudaBuffer<Vertex>& vertexBuffer,
                               std::size_t vertexCount,
                               const CudaBuffer<std::uint32_t>& indexBuffer,
                               std::size_t indexCount)
{
    if (vertexCount > MAX_VERTICES_PER_DRAW)
    {
        throw std::runtime_error("too many vertices in draw call");
    }

    {
        const int blockSize = 256;
        const int numBlocks = (static_cast<int>(vertexCount) + blockSize - 1) / blockSize;
        TransformVerticesKernel<<<numBlocks, blockSize, 0, stream>>>(
            vertexBuffer.GetGpuPtr(), vertexCount, g_uniformBuffer, g_transformedVertexCache
        );
        CHECK_CUDA(cudaGetLastError());
    }
    {
        const dim3 blockSize(16, 16, 1);
        const dim3 numBlocks(static_cast<unsigned>(indexCount / 3), 1, 1);
        RasterizeAndShadeKernel<<<numBlocks, blockSize, 0, stream>>>(
            renderSurface, depthBuffer.GetGpuPtr(), surfaceSize, g_transformedVertexCache,
            indexBuffer.GetGpuPtr(), indexCount
        );
        CHECK_CUDA(cudaGetLastError());
    }
}
