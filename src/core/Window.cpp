#include "Window.h"

#include <stdexcept>
#include <utility>
#include <cuda_d3d11_interop.h>

using namespace ggg;

#define CHECK_HR(expr)                                                                                              \
    do                                                                                                              \
    {                                                                                                               \
        const HRESULT hr__ = (expr);                                                                                \
        if (FAILED(hr__))                                                                                           \
        {                                                                                                           \
            throw std::runtime_error(std::string(#expr " failed with HRESULT 0x") + std::to_string(hr__));          \
        }                                                                                                           \
    } while (0)

#define CHECK_CUDA(expr)                                                                                            \
    do                                                                                                              \
    {                                                                                                               \
        const cudaError_t err__ = (expr);                                                                           \
        if (err__ != cudaSuccess)                                                                                   \
        {                                                                                                           \
            throw std::runtime_error(std::string(#expr ": ") + cudaGetErrorString(err__));                          \
        }                                                                                                           \
    } while (0)

namespace
{
    LRESULT CALLBACK wnd_proc(HWND hwnd, UINT msg, WPARAM wparam, LPARAM lparam)
    {
        switch (msg)
        {
        case WM_DESTROY:
            PostQuitMessage(0);
            return 0;
        default:
            return DefWindowProc(hwnd, msg, wparam, lparam);
        }
    }
}

Window::Window(UINT width, UINT height)
    : m_width(width), m_height(height)
{
    const HINSTANCE instance = GetModuleHandle(NULL);
    const wchar_t* className = L"CudaDx11Window";
    WNDCLASSEXW wc{};
    wc.cbSize = sizeof(wc);
    wc.style = CS_HREDRAW | CS_VREDRAW;
    wc.lpfnWndProc = wnd_proc;
    wc.hInstance = instance;
    wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
    wc.lpszClassName = className;

    RegisterClassExW(&wc);

    RECT rect{0, 0, static_cast<LONG>(width), static_cast<LONG>(height)};
    AdjustWindowRect(&rect, WS_OVERLAPPEDWINDOW, FALSE);

    const int winWidth = rect.right - rect.left;
    const int winHeight = rect.bottom - rect.top;

    m_hwnd = CreateWindowExW(
        0,
        className,
        L"CUDA + Direct3D11",
        WS_OVERLAPPEDWINDOW,
        CW_USEDEFAULT,
        CW_USEDEFAULT,
        winWidth,
        winHeight,
        nullptr,
        nullptr,
        instance,
        nullptr
    );

    if (!m_hwnd)
    {
        throw std::runtime_error("Failed to create window");
    }

    ShowWindow(m_hwnd, SW_SHOWDEFAULT);
    UpdateWindow(m_hwnd);

    DXGI_SWAP_CHAIN_DESC swapDesc{};
    swapDesc.BufferCount = 2;
    swapDesc.BufferDesc.Width = m_width;
    swapDesc.BufferDesc.Height = m_height;
    swapDesc.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    swapDesc.BufferDesc.RefreshRate.Numerator = 60;
    swapDesc.BufferDesc.RefreshRate.Denominator = 1;
    swapDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    swapDesc.OutputWindow = m_hwnd;
    swapDesc.SampleDesc.Count = 1;
    swapDesc.SampleDesc.Quality = 0;
    swapDesc.Windowed = TRUE;
    swapDesc.SwapEffect = DXGI_SWAP_EFFECT_DISCARD;
    swapDesc.Flags = 0;

    const D3D_FEATURE_LEVEL featureLevels[] = {D3D_FEATURE_LEVEL_11_1, D3D_FEATURE_LEVEL_11_0};
    D3D_FEATURE_LEVEL createdLevel{};

    CHECK_HR(D3D11CreateDeviceAndSwapChain(
        nullptr,
        D3D_DRIVER_TYPE_HARDWARE,
        nullptr,
        D3D11_CREATE_DEVICE_BGRA_SUPPORT,
        featureLevels,
        ARRAYSIZE(featureLevels),
        D3D11_SDK_VERSION,
        &swapDesc,
        m_swapChain.GetAddressOf(),
        m_device.GetAddressOf(),
        &createdLevel,
        m_context.GetAddressOf())
    );

    CHECK_HR(m_swapChain->GetBuffer(0, __uuidof(ID3D11Texture2D),
        reinterpret_cast<void**>(m_backBuffer.GetAddressOf()))
    );

    m_frames.reserve(FRAMES_IN_FLIGHT);
    for (UINT i = 0; i < FRAMES_IN_FLIGHT; ++i)
    {
        m_frames.emplace_back(Frame{m_device.Get(), m_width, m_height});
    }
}

Window::Window(Window&& other) noexcept
{
    MoveFrom(std::move(other));
}

Window& Window::operator=(Window&& other) noexcept
{
    if (this != &other)
    {
        Cleanup();
        MoveFrom(std::move(other));
    }
    return *this;
}

Window::~Window()
{
    if (m_context)
    {
        m_context->ClearState();
    }
    if (m_swapChain)
    {
        m_swapChain->SetFullscreenState(FALSE, nullptr);
    }
}

bool Window::DrainMessages()
{
    MSG msg{};
    while (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE))
    {
        if (msg.message == WM_QUIT)
        {
            return false;
        }

        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }
    return true;
}

Window::FrameContext Window::BeginFrame()
{
    Frame& frame = m_frames[m_nextRender];
    frame.WaitForReuse(m_context.Get());
    while (frame.m_pendingRender)
    {
        PresentReady();
        frame.WaitForReuse(m_context.Get());
        Sleep(0);
    }

    cudaSurfaceObject_t surface = frame.MapSurface();
    FrameContext ctx{};
    ctx.surface = surface;
    ctx.stream = frame.m_stream;
    ctx.width = m_width;
    ctx.height = m_height;
    return ctx;
}

void Window::EndFrame(FrameContext& ctx)
{
    m_frames[m_nextRender].UnmapSurface(ctx.surface);
    m_nextRender = (m_nextRender + 1) % static_cast<UINT>(m_frames.size());
    PresentReady();
}

Window::Frame::Frame(ID3D11Device* device, UINT width, UINT height)
{
    D3D11_TEXTURE2D_DESC desc{};
    desc.Width = width;
    desc.Height = height;
    desc.MipLevels = 1;
    desc.ArraySize = 1;
    desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    desc.SampleDesc.Count = 1;
    desc.SampleDesc.Quality = 0;
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;

    CHECK_HR(device->CreateTexture2D(&desc, nullptr, m_texture.GetAddressOf()));

    D3D11_QUERY_DESC qdesc{};
    qdesc.Query = D3D11_QUERY_EVENT;
    CHECK_HR(device->CreateQuery(&qdesc, m_reuseFence.GetAddressOf()));

    CHECK_CUDA(cudaGraphicsD3D11RegisterResource(
        &m_cudaResource,
        m_texture.Get(),
        cudaGraphicsRegisterFlagsSurfaceLoadStore
    ));
    CHECK_CUDA(cudaGraphicsResourceSetMapFlags(m_cudaResource, cudaGraphicsMapFlagsWriteDiscard));
    CHECK_CUDA(cudaStreamCreate(&m_stream));
    CHECK_CUDA(cudaEventCreateWithFlags(&m_renderDone, cudaEventDisableTiming));
}

Window::Frame::Frame(Frame&& other) noexcept
{
    MoveFrom(std::move(other));
}

Window::Frame& Window::Frame::operator=(Frame&& other) noexcept
{
    if (this != &other)
    {
        Cleanup();
        MoveFrom(std::move(other));
    }
    return *this;
}

Window::Frame::~Frame()
{
    Cleanup();
}

cudaSurfaceObject_t Window::Frame::MapSurface()
{
    if (m_mapped)
    {
        throw std::runtime_error("Surface already mapped");
    }

    CHECK_CUDA(cudaGraphicsMapResources(1, &m_cudaResource, m_stream));

    cudaArray_t array{};
    CHECK_CUDA(cudaGraphicsSubResourceGetMappedArray(&array, m_cudaResource, 0, 0));

    cudaResourceDesc resDesc{};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = array;

    cudaSurfaceObject_t surface{};
    CHECK_CUDA(cudaCreateSurfaceObject(&surface, &resDesc));
    m_mapped = true;
    return surface;
}

void Window::Frame::UnmapSurface(cudaSurfaceObject_t surface)
{
    if (!m_mapped)
    {
        throw std::runtime_error("Surface not mapped");
    }
    CHECK_CUDA(cudaDestroySurfaceObject(surface));
    CHECK_CUDA(cudaGraphicsUnmapResources(1, &m_cudaResource, m_stream));
    CHECK_CUDA(cudaEventRecord(m_renderDone, m_stream));
    m_pendingRender = true;
    m_mapped = false;
}

void Window::Frame::WaitForReuse(ID3D11DeviceContext* context)
{
    if (!m_inFlight)
    {
        return;
    }

    while (context->GetData(m_reuseFence.Get(), nullptr, 0, 0) == S_FALSE)
    {
        Sleep(0);
    }
    m_inFlight = false;
}

bool Window::Frame::PresentIfReady(ID3D11DeviceContext* context, IDXGISwapChain* swapChain, ID3D11Texture2D* backBuffer)
{
    if (!m_pendingRender)
    {
        return false;
    }

    const cudaError_t status = cudaEventQuery(m_renderDone);
    if (status == cudaErrorNotReady)
    {
        return false;
    }
    CHECK_CUDA(status);

    context->CopyResource(backBuffer, m_texture.Get());
    context->End(m_reuseFence.Get());
    m_inFlight = true;
    m_pendingRender = false;

    CHECK_HR(swapChain->Present(1, 0));
    return true;
}

void Window::Frame::MoveFrom(Frame&& other)
{
    m_texture = std::move(other.m_texture);
    m_reuseFence = std::move(other.m_reuseFence);
    m_cudaResource = std::exchange(other.m_cudaResource, nullptr);
    m_stream = std::exchange(other.m_stream, nullptr);
    m_renderDone = std::exchange(other.m_renderDone, nullptr);
    m_pendingRender = std::exchange(other.m_pendingRender, false);
    m_inFlight = std::exchange(other.m_inFlight, false);
    m_mapped = std::exchange(other.m_mapped, false);
}

void Window::Frame::Cleanup()
{
    if (m_cudaResource)
    {
        cudaGraphicsUnregisterResource(m_cudaResource);
        m_cudaResource = nullptr;
    }
    if (m_renderDone)
    {
        cudaEventDestroy(m_renderDone);
        m_renderDone = nullptr;
    }
    if (m_stream)
    {
        cudaStreamDestroy(m_stream);
        m_stream = nullptr;
    }
    m_reuseFence.Reset();
    m_texture.Reset();
    m_pendingRender = false;
    m_inFlight = false;
    m_mapped = false;
}

void Window::PresentReady()
{
    for (UINT i = 0; i < m_frames.size(); ++i)
    {
        Frame& frame = m_frames[m_nextPresent];
        if (frame.PresentIfReady(m_context.Get(), m_swapChain.Get(), m_backBuffer.Get()))
        {
            m_nextPresent = (m_nextPresent + 1) % static_cast<UINT>(m_frames.size());
        }
        else
        {
            break;
        }
    }
}

void Window::MoveFrom(Window&& other)
{
    m_hwnd = std::exchange(other.m_hwnd, nullptr);
    m_width = std::exchange(other.m_width, 0u);
    m_height = std::exchange(other.m_height, 0u);
    m_device = std::move(other.m_device);
    m_context = std::move(other.m_context);
    m_swapChain = std::move(other.m_swapChain);
    m_backBuffer = std::move(other.m_backBuffer);
    m_frames = std::move(other.m_frames);
    m_nextRender = std::exchange(other.m_nextRender, 0u);
    m_nextPresent = std::exchange(other.m_nextPresent, 0u);
}

void Window::Cleanup()
{
    if (m_context)
    {
        m_context->ClearState();
    }
    if (m_swapChain)
    {
        m_swapChain->SetFullscreenState(FALSE, nullptr);
    }

    m_backBuffer.Reset();
    m_swapChain.Reset();
    m_context.Reset();
    m_device.Reset();
    m_frames.clear();
    m_hwnd = nullptr;
    m_width = 0;
    m_height = 0;
    m_nextRender = 0;
    m_nextPresent = 0;
}
