#pragma once

#include <cuda_runtime.h>
#include <surface_types.h>

namespace ggg::cuda
{
    void LaunchFillSurface(cudaSurfaceObject_t surface,
                           unsigned width,
                           unsigned height,
                           float timeSeconds,
                           cudaStream_t stream);
}

