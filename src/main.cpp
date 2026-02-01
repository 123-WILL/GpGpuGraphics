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

    while (running)
    {
        running = window.DrainMessages();
        const float timeSeconds = std::chrono::duration<float>(std::chrono::steady_clock::now() - startTime).count();
        Window::FrameContext frameCtx = window.BeginFrame();
        cuda::LaunchClear(
            frameCtx.surface, frameCtx.width, frameCtx.height, timeSeconds, frameCtx.stream
        );
        cuda::LaunchDraw(
            frameCtx.surface, frameCtx.width, frameCtx.height, timeSeconds, frameCtx.stream,
            vertexBuffer, vertexCount, indexBuffer, indexCount
        );
        window.EndFrame(frameCtx);
    }

    return 0;
}
