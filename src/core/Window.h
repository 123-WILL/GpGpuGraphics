#pragma once

#include <chrono>
#include <vector>
#define NOMINMAX
#include <windows.h>
#include <d3d11.h>
#include <dxgi1_2.h>
#include <wrl/client.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace ggg
{
    class Window
    {
    public:
        struct FrameContext
        {
            cudaSurfaceObject_t surface{};
            cudaStream_t stream{};
            UINT width{};
            UINT height{};
        };

        Window(UINT width, UINT height);
        Window(const Window&) = delete;
        Window(Window&& other) noexcept;
        Window& operator=(const Window&) = delete;
        Window& operator=(Window&& other) noexcept;
        ~Window();

        bool PumpMessages();
        FrameContext BeginFrame();
        void EndFrame(FrameContext& ctx);

    private:
        class Frame
        {
        public:
            Frame(ID3D11Device* device, UINT width, UINT height);
            Frame(const Frame&) = delete;
            Frame(Frame&& other) noexcept;
            Frame& operator=(const Frame&) = delete;
            Frame& operator=(Frame&& other) noexcept;
            ~Frame();

            cudaSurfaceObject_t MapSurface();
            void UnmapSurface(cudaSurfaceObject_t surface);
            void WaitForReuse(ID3D11DeviceContext* context);
            bool PresentIfReady(ID3D11DeviceContext* context, IDXGISwapChain* swapChain, ID3D11Texture2D* backBuffer);

        private:
            void MoveFrom(Frame&& other);
            void Cleanup();

        public:
            Microsoft::WRL::ComPtr<ID3D11Texture2D> m_texture;
            Microsoft::WRL::ComPtr<ID3D11Query> m_reuseFence;
            cudaGraphicsResource* m_cudaResource{};
            cudaStream_t m_stream{};
            cudaEvent_t m_renderDone{};
            bool m_pendingRender{false};
            bool m_inFlight{false};
            bool m_mapped{false};
        };

        static constexpr UINT FRAMES_IN_FLIGHT = 3;

        HWND m_hwnd{};
        UINT m_width{};
        UINT m_height{};
        Microsoft::WRL::ComPtr<ID3D11Device> m_device;
        Microsoft::WRL::ComPtr<ID3D11DeviceContext> m_context;
        Microsoft::WRL::ComPtr<IDXGISwapChain> m_swapChain;
        Microsoft::WRL::ComPtr<ID3D11Texture2D> m_backBuffer;
        std::vector<Frame> m_frames;
        UINT m_nextRender{0};
        UINT m_nextPresent{0};

        void PresentReady();
        void MoveFrom(Window&& other);
        void Cleanup();
    };
}

