#pragma once

#include <stdexcept>
#include <string>
#include <cuda_runtime.h>

#define CHECK_CUDA(expr)                                                                        \
    do                                                                                          \
    {                                                                                           \
        const cudaError_t err__ = (expr);                                                       \
        if (err__ != cudaSuccess)                                                               \
        {                                                                                       \
            throw std::runtime_error(std::string(#expr ": ") + cudaGetErrorString(err__));      \
        }                                                                                       \
    } while (0)