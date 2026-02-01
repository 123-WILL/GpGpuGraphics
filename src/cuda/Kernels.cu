#include "Kernels.cuh"

#include <cstdint>
#include <cuda_runtime.h>
#include <surface_indirect_functions.h>
#include <surface_types.h>

using namespace ggg;

namespace
{
    struct VertexShaderOut
    {
        Vec3f n{};
        Vec2f uv{};
        float viewZ{};
    };

    __device__ Vec4f VertexShader(const Vertex& vert, float timeSeconds, float aspect, VertexShaderOut& out)
    {
        const Vec3f p = vert.p;
        const float s = 0.85f;
        const float ang = timeSeconds * 0.6f;
        const float c = cosf(ang);
        const float sn = sinf(ang);

        const Vec3f pWorld{
            (p.x() * c + p.z() * sn) * s,
            p.y() * s - 0.25f,
            (-p.x() * sn + p.z() * c) * s
        };

        const Vec3f nIn = vert.n;
        const Vec3f nWorld{
            (nIn.x() * c + nIn.z() * sn),
            nIn.y(),
            (-nIn.x() * sn + nIn.z() * c)
        };

        out.n = nWorld;
        out.uv = vert.uv;

        // View + projection (right-handed, camera looks down -Z)
        const Vec3f camPos{0.0f, 0.0f, 2.2f};
        const Vec3f pView = pWorld - camPos;
        out.viewZ = -pView.z();

        const float fovY = 60.0f * (3.14159265358979323846f / 180.0f);
        const float f = 1.0f / tanf(fovY * 0.5f);

        Vec4f clip{
            (pView.x() * f / aspect),
            (pView.y() * f),
            pView.z(),
            -pView.z()
        };
        return clip;
    }

    __device__ Vec4f FragmentShader(const VertexShaderOut& in)
    {
        const Vec3f n = in.n.Normalized();
        const Vec3f lightDir{0.3508772f, 0.9022557f, 0.2506266f}; // normalize({0.35,0.9,0.25})
        const float ndotl = fmaxf(0.0f, Dot(n, lightDir));
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

    __global__ void ClearKernel(cudaSurfaceObject_t surface,
                                unsigned width,
                                unsigned height,
                                float timeSeconds,
                                unsigned int* depthBuffer)
    {
        const unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x >= width || y >= height)
        {
            return;
        }
        static constexpr unsigned int FLT_MAX_BITS = 0x7f7fffffU;
        depthBuffer[y * width + x] = FLT_MAX_BITS;
        surf2Dwrite(ToRGBA8(Vec4f{0.f, 0.f, 0.f, 1.0f}), surface, static_cast<int>(x * sizeof(uchar4)), static_cast<int>(y));
    }

    __global__ void DrawKernel(cudaSurfaceObject_t surface,
                               unsigned width,
                               unsigned height,
                               float timeSeconds,
                               unsigned int* __restrict__ depthBuffer,
                               const Vertex* __restrict__ cudaVertexBuffer,
                               std::size_t vertexCount,
                               const std::uint32_t* __restrict__ cudaIndexBuffer,
                               std::size_t indexCount)
    {
        const std::size_t triCount = indexCount / 3;
        const std::size_t triId = blockIdx.x;
        if (triId >= triCount)
        {
            return;
        }

        const float aspect = static_cast<float>(width) / static_cast<float>(height);

        const std::uint32_t i0 = cudaIndexBuffer[triId * 3 + 0];
        const std::uint32_t i1 = cudaIndexBuffer[triId * 3 + 1];
        const std::uint32_t i2 = cudaIndexBuffer[triId * 3 + 2];
        if (i0 >= vertexCount || i1 >= vertexCount || i2 >= vertexCount)
        {
            return;
        }

        VertexShaderOut vsOut0{};
        VertexShaderOut vsOut1{};
        VertexShaderOut vsOut2{};

        // clip space [(-w, -w) (bottom left), (w, w) (top right)]
        const Vec4f clip0 = VertexShader(cudaVertexBuffer[i0], timeSeconds, aspect, vsOut0);
        const Vec4f clip1 = VertexShader(cudaVertexBuffer[i1], timeSeconds, aspect, vsOut1);
        const Vec4f clip2 = VertexShader(cudaVertexBuffer[i2], timeSeconds, aspect, vsOut2);

        if (clip0.w() == 0.0f || clip1.w() == 0.0f || clip2.w() == 0.0f)
        {
            return;
        }

        // NDC [(-1, -1) (bottom left), (1, 1) (top right)]
        const Vec2f ndc0{clip0.x() / clip0.w(), clip0.y() / clip0.w()};
        const Vec2f ndc1{clip1.x() / clip1.w(), clip1.y() / clip1.w()};
        const Vec2f ndc2{clip2.x() / clip2.w(), clip2.y() / clip2.w()};

        // pixel space [(0, height-1) (bottom left), (width-1, 0) (top right)]
        const Vec2f p0{(ndc0.x() * 0.5f + 0.5f) * (static_cast<float>(width) - 1.0f),
                       (1.0f - (ndc0.y() * 0.5f + 0.5f)) * (static_cast<float>(height) - 1.0f)};
        const Vec2f p1{(ndc1.x() * 0.5f + 0.5f) * (static_cast<float>(width) - 1.0f),
                       (1.0f - (ndc1.y() * 0.5f + 0.5f)) * (static_cast<float>(height) - 1.0f)};
        const Vec2f p2{(ndc2.x() * 0.5f + 0.5f) * (static_cast<float>(width) - 1.0f),
                       (1.0f - (ndc2.y() * 0.5f + 0.5f)) * (static_cast<float>(height) - 1.0f)};

        const float minXf = fminf(p0.x(), fminf(p1.x(), p2.x()));
        const float minYf = fminf(p0.y(), fminf(p1.y(), p2.y()));
        const float maxXf = fmaxf(p0.x(), fmaxf(p1.x(), p2.x()));
        const float maxYf = fmaxf(p0.y(), fmaxf(p1.y(), p2.y()));

        const int minX = max(0, min(static_cast<int>(floorf(minXf)), static_cast<int>(width) - 1));
        const int minY = max(0, min(static_cast<int>(floorf(minYf)), static_cast<int>(height) - 1));
        const int maxX = max(0, min(static_cast<int>(ceilf(maxXf)), static_cast<int>(width) - 1));
        const int maxY = max(0, min(static_cast<int>(ceilf(maxYf)), static_cast<int>(height) - 1));

        const float area = (p2.x() - p0.x()) * (p1.y() - p0.y()) - (p2.y() - p0.y()) * (p1.x() - p0.x());
        if (area == 0.0f)
        {
            return;
        }
        const float invArea = 1.0f / area;
        const bool topLeftIsPositive = area > 0.0f;

        // Precompute edge equations for incremental evaluation:
        // Edge(a,b,p) = A*x + B*y + C, where A=(b.y-a.y), B=-(b.x-a.x), C=(a.y*b.x-a.x*b.y)
        const float A0 = p2.y() - p1.y();
        const float B0 = -(p2.x() - p1.x());
        const float C0 = p1.y() * p2.x() - p1.x() * p2.y();

        const float A1 = p0.y() - p2.y();
        const float B1 = -(p0.x() - p2.x());
        const float C1 = p2.y() * p0.x() - p2.x() * p0.y();

        const float A2 = p1.y() - p0.y();
        const float B2 = -(p1.x() - p0.x());
        const float C2 = p0.y() * p1.x() - p0.x() * p1.y();

        const float stepX0 = A0 * static_cast<float>(blockDim.x);
        const float stepX1 = A1 * static_cast<float>(blockDim.x);
        const float stepX2 = A2 * static_cast<float>(blockDim.x);

        for (int y = minY + static_cast<int>(threadIdx.y); y <= maxY; y += static_cast<int>(blockDim.y))
        {
            const float py = static_cast<float>(y) + 0.5f;
            int x = minX + static_cast<int>(threadIdx.x);
            float px = static_cast<float>(x) + 0.5f;

            float w0e = A0 * px + B0 * py + C0;
            float w1e = A1 * px + B1 * py + C1;
            float w2e = A2 * px + B2 * py + C2;

            for (; x <= maxX; x += static_cast<int>(blockDim.x))
            {
                const bool inside = topLeftIsPositive ? (w0e >= 0.0f && w1e >= 0.0f && w2e >= 0.0f)
                                                      : (w0e <= 0.0f && w1e <= 0.0f && w2e <= 0.0f);
                if (inside)
                {
                    const float w0 = w0e * invArea;
                    const float w1 = w1e * invArea;
                    const float w2 = w2e * invArea;
                    const float viewZ = w0 * vsOut0.viewZ + w1 * vsOut1.viewZ + w2 * vsOut2.viewZ;
                    if (viewZ > 0.0f)
                    {
                        const int depthIdx = y * static_cast<int>(width) + x;
                        const unsigned int newDepthBits = __float_as_uint(viewZ);
                        const unsigned int oldDepthBits = atomicMin(depthBuffer + depthIdx, newDepthBits);
                        if (newDepthBits < oldDepthBits)
                        {
                            const Vec3f n = (vsOut0.n * w0) + (vsOut1.n * w1) + (vsOut2.n * w2);
                            const Vec2f uv = (vsOut0.uv * w0) + (vsOut1.uv * w1) + (vsOut2.uv * w2);
                            const VertexShaderOut fsIn = { .n = n, .uv = uv, .viewZ = viewZ };
                            const Vec4f color = FragmentShader(fsIn);
                            surf2Dwrite(ToRGBA8(color), surface, x * static_cast<int>(sizeof(uchar4)), y);
                        }
                    }
                }

                w0e += stepX0;
                w1e += stepX1;
                w2e += stepX2;
            }
        }
    }

    unsigned int* g_depthBuffer = nullptr;
    std::size_t g_depthCapacity = 0;

    __host__ void EnsureDepthBufferSize(unsigned width, unsigned height)
    {
        if (std::size_t neededDepth = width * height; neededDepth > g_depthCapacity)
        {
            if (g_depthBuffer)
            {
                cudaFree(g_depthBuffer);
                g_depthBuffer = nullptr;
            }
            cudaMalloc(&g_depthBuffer, neededDepth * sizeof(unsigned int));
            g_depthCapacity = neededDepth;
        }
    }
}

__host__ void cuda::LaunchClear(cudaSurfaceObject_t surface,
                                unsigned width,
                                unsigned height,
                                float timeSeconds,
                                cudaStream_t stream)
{
    if (!surface || width == 0 || height == 0)
    {
        return;
    }

    EnsureDepthBufferSize(width, height);

    const dim3 block(16, 16, 1);
    const dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y, 1);
    ClearKernel<<<grid, block, 0, stream>>>(surface, width, height, timeSeconds, g_depthBuffer);
}

__host__ void cuda::LaunchDraw(cudaSurfaceObject_t surface,
                               unsigned width,
                               unsigned height,
                               float timeSeconds,
                               cudaStream_t stream,
                               const Vertex* cudaVertexBuffer,
                               std::size_t vertexCount,
                               const std::uint32_t* cudaIndexBuffer,
                               std::size_t indexCount)
{
    if (!surface || !cudaVertexBuffer || !cudaIndexBuffer ||  width == 0 || height == 0 || vertexCount == 0 || indexCount < 3)
    {
        return;
    }

    EnsureDepthBufferSize(width, height);

    const dim3 block(16, 16, 1);
    const dim3 grid(static_cast<unsigned>(indexCount / 3), 1, 1);
    DrawKernel<<<grid, block, 0, stream>>>(surface, width, height, timeSeconds, g_depthBuffer, cudaVertexBuffer, vertexCount, cudaIndexBuffer, indexCount);
}
