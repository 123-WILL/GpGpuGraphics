#pragma once

#include <cstddef>
#include <utility>
#include <cuda_runtime.h>
#include <span>
#include "CudaMacros.h"

template<typename T>
class CudaBuffer
{
public:
    CudaBuffer() = default;

    explicit CudaBuffer(std::size_t n)
    {
        CHECK_CUDA(cudaMalloc(&m_cudaBufferPtr, n * sizeof(T)));
        m_count = n;
    }

    CudaBuffer(const CudaBuffer&) = delete;
    CudaBuffer& operator=(const CudaBuffer&) = delete;

    CudaBuffer(CudaBuffer&& other) noexcept
    {
        m_cudaBufferPtr = std::exchange(other.m_cudaBufferPtr, nullptr);
        m_count = std::exchange(other.m_count, 0);
    }

    CudaBuffer& operator=(CudaBuffer&& other) noexcept
    {
        if (this == &other)
        {
            return *this;
        }
        m_cudaBufferPtr = std::exchange(other.m_cudaBufferPtr, nullptr);
        m_count = std::exchange(other.m_count, 0);
        return *this;
    }

    ~CudaBuffer()
    {
        if (m_cudaBufferPtr)
        {
            cudaFree(m_cudaBufferPtr);
            m_cudaBufferPtr = nullptr;
            m_count = 0;
        }
    }

    void CopyToGpuBuffer(const T& data)
    {
        cudaMemcpy(m_cudaBufferPtr, &data, sizeof(T), cudaMemcpyHostToDevice);
    }

    void CopyToGpuBuffer(std::span<T> data)
    {
        cudaMemcpy(m_cudaBufferPtr, data.data(), data.size() * sizeof(T), cudaMemcpyHostToDevice);
    }

    [[nodiscard]] T* GetGpuPtr()
    {
        return m_cudaBufferPtr;
    }

    [[nodiscard]] const T* GetGpuPtr() const
    {
        return m_cudaBufferPtr;
    }

private:
    std::size_t m_count{};
    T* m_cudaBufferPtr{};
};
