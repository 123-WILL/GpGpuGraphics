#include "core/Window.h"
#include "cuda/kernels.cuh"

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

    while (running)
    {
        running = window.PumpMessages();
        const float timeSeconds = std::chrono::duration<float>(std::chrono::steady_clock::now() - startTime).count();

        Window::FrameContext frameCtx = window.BeginFrame();
        cuda::LaunchFillSurface(frameCtx.surface, frameCtx.width, frameCtx.height, timeSeconds, frameCtx.stream);
        window.EndFrame(frameCtx);
    }

    return 0;
}
