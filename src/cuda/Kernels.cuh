#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <surface_types.h>

#include "CudaBuffer.h"
#include "graphics/Model.h"
#include "math/Matrix.h"

namespace ggg::cuda
{
    struct UniformBuffer
    {
        Mat4f m_modelMatrix;
        Mat4f m_viewMatrix;
        Mat4f m_projectionMatrix;
        float m_timeSeconds;
        float m_aspectRatio;
    };

    void InitCudaGraphics();
    void StopCudaGraphics();

    void SetUniformBuffer(const UniformBuffer& uniform);

    void LaunchClear(cudaSurfaceObject_t renderSurface,
                     CudaBuffer<std::uint32_t>& depthBuffer,
                     Vec2u surfaceSize,
                     cudaStream_t stream);

    void LaunchDraw(cudaSurfaceObject_t renderSurface,
                    CudaBuffer<std::uint32_t>& depthBuffer,
                    Vec2u surfaceSize,
                    cudaStream_t stream,
                    const CudaBuffer<Vertex>& vertexBuffer,
                    std::size_t vertexCount,
                    const CudaBuffer<std::uint32_t>& indexBuffer,
                    std::size_t indexCount);
}
