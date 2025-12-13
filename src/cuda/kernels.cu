#include "kernels.cuh"

#include <cuda_runtime.h>
#include <surface_indirect_functions.h>
#include <surface_types.h>

namespace
{
    __global__ void FillSurfaceKernel(cudaSurfaceObject_t surface,
                                      unsigned width,
                                      unsigned height,
                                      float timeSeconds)
    {
        const unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x >= width || y >= height)
        {
            return;
        }

        const float fx = static_cast<float>(x) / static_cast<float>(width);
        const float fy = static_cast<float>(y) / static_cast<float>(height);

        const float r = 0.5f + 0.5f * __sinf(timeSeconds + fx * 6.28318f);
        const float g = 0.5f + 0.5f * __cosf(timeSeconds * 0.5f + fy * 3.14159f);
        const float b = 0.5f + 0.5f * __sinf(timeSeconds * 0.25f + (fx + fy) * 3.14159f);

        uchar4 color{
            static_cast<unsigned char>(r * 255.0f),
            static_cast<unsigned char>(g * 255.0f),
            static_cast<unsigned char>(b * 255.0f),
            255
        };

        surf2Dwrite(color, surface, x * sizeof(uchar4), y);
    }
}

namespace ggg::cuda
{
    void LaunchFillSurface(cudaSurfaceObject_t surface,
                           unsigned width,
                           unsigned height,
                           float timeSeconds,
                           cudaStream_t stream)
    {
        constexpr dim3 blockDim(16u, 16u, 1u);
        const dim3 gridDim((width + blockDim.x - 1u) / blockDim.x,
                           (height + blockDim.y - 1u) / blockDim.y,
                           1u);

        FillSurfaceKernel<<<gridDim, blockDim, 0u, stream>>>(surface, width, height, timeSeconds);
    }
}

