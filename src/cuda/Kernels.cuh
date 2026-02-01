#pragma once

#include <cuda_runtime.h>
#include <surface_types.h>
#include "graphics/Model.h"

namespace ggg::cuda
{
    void LaunchClear(cudaSurfaceObject_t surface,
                     unsigned width,
                     unsigned height,
                     float timeSeconds,
                     cudaStream_t stream);

    void LaunchDraw(cudaSurfaceObject_t surface,
                    unsigned width,
                    unsigned height,
                    float timeSeconds,
                    cudaStream_t stream,
                    const Vertex* cudaVertexBuffer,
                    std::size_t vertexCount,
                    const std::uint32_t* cudaIndexBuffer,
                    std::size_t indexCount);
}
