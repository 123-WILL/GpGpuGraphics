#include <cmath>
#include "core/Window.h"
#include "cuda/Kernels.cuh"
#include "graphics/Model.h"

using namespace ggg;

namespace
{
    constexpr UINT WIDTH = 1280;
    constexpr UINT HEIGHT = 720;

    [[nodiscard]] Mat4f InitTranslation(const Vec3f& t)
    {
        return Mat4f{
            Vec4f{1.0f, 0.0f, 0.0f, 0.0f},
            Vec4f{0.0f, 1.0f, 0.0f, 0.0f},
            Vec4f{0.0f, 0.0f, 1.0f, 0.0f},
            Vec4f{t.x(), t.y(), t.z(), 1.0f},
        };
    }

    [[nodiscard]] Mat4f InitScale(float s)
    {
        return Mat4f{
            Vec4f{s, 0.0f, 0.0f, 0.0f},
            Vec4f{0.0f, s, 0.0f, 0.0f},
            Vec4f{0.0f, 0.0f, s, 0.0f},
            Vec4f{0.0f, 0.0f, 0.0f, 1.0f},
        };
    }

    [[nodiscard]] Mat4f InitRotationY(float radians)
    {
        float c = std::cos(radians);
        float s = std::sin(radians);
        return Mat4f{
            Vec4f{c, 0.0f, -s, 0.0f},
            Vec4f{0.0f, 1.0f, 0.0f, 0.0f},
            Vec4f{s, 0.0f, c, 0.0f},
            Vec4f{0.0f, 0.0f, 0.0f, 1.0f},
        };
    }

    [[nodiscard]] Mat4f InitProjection(float fovYRadians, float aspect, float zNear, float zFar)
    {
        float tanHalfFov = std::tan(fovYRadians * 0.5f);
        return Mat4f{
            Vec4f{1.0f / (aspect * tanHalfFov), 0.0f, 0.0f, 0.0f},
            Vec4f{0.0f, 1.0f / tanHalfFov, 0.0f, 0.0f},
            Vec4f{0.0f, 0.0f, (-zNear - zFar) / (zNear - zFar), -1.0f},
            Vec4f{0.0f, 0.0f, (2.0f * zFar * zNear) / (zNear - zFar), 0.0f},
        };
    }
}

int main()
{
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

        const float timeSeconds = std::chrono::duration<float>(std::chrono::steady_clock::now() - startTime).count();
        const float aspectRatio = static_cast<float>(frameCtx.width) / static_cast<float>(frameCtx.height);

        cuda::SetUniformBuffer(cuda::UniformBuffer{
            .m_modelMatrix = InitTranslation(Vec3f{0.0f, -0.25f, 0.0f}) *
                             InitRotationY(timeSeconds * 0.6f) *
                             InitScale(0.85f),
            .m_viewMatrix = InitTranslation(Vec3f{0.0f, 0.0f, -2.2f}),
            .m_projectionMatrix = InitProjection(3.1415f / 3.f, aspectRatio, -0.01f, -1000.f),
            .m_timeSeconds = timeSeconds,
            .m_aspectRatio = aspectRatio,
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
