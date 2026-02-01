#include "core/Window.h"
#include "cuda/Kernels.cuh"
#include "graphics/Model.h"

namespace
{
    constexpr UINT WIDTH = 1280;
    constexpr UINT HEIGHT = 720;
}

int main()
{
    using namespace ggg;

    Window window(WIDTH, HEIGHT);
    bool running = true;
    const auto startTime = std::chrono::steady_clock::now();

    Model model("stanford_bunny");
    const auto [vertexBuffer, vertexCount] = model.GetCudaVertexBuffer();
    const auto [indexBuffer, indexCount] = model.GetCudaIndexBuffer();

    std::uint32_t* cudaDepthBuffer{};
    cudaMalloc(&cudaDepthBuffer, sizeof(std::uint32_t) * WIDTH * HEIGHT);

    cuda::InitCudaGraphics();
    while (running)
    {
        running = window.DrainMessages();
        Window::FrameContext frameCtx = window.BeginFrame();

        cuda::SetUniformBuffer(cuda::UniformBuffer{
            .m_timeSeconds = std::chrono::duration<float>(std::chrono::steady_clock::now() - startTime).count(),
            .m_aspectRatio = static_cast<float>(frameCtx.width) / static_cast<float>(frameCtx.height)
        });
        cuda::LaunchClear(
            frameCtx.surface, cudaDepthBuffer, Vec2u{frameCtx.width, frameCtx.height}, frameCtx.stream
        );
        cuda::LaunchDraw(
            frameCtx.surface, cudaDepthBuffer, Vec2u{frameCtx.width, frameCtx.height}, frameCtx.stream,
            vertexBuffer, vertexCount, indexBuffer, indexCount
        );

        window.EndFrame(frameCtx);
    }
    cuda::StopCudaGraphics();
    cudaFree(cudaDepthBuffer);

    return 0;
}
