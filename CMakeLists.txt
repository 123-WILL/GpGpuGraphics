cmake_minimum_required(VERSION 3.27)
project(GpGpuGraphics LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CPU_SOURCES
        src/main.cpp
)

set(CUDA_SOURCES
        src/kernels.cu
        src/kernels.cuh
)

add_executable(GpGpuGraphics)

target_sources(GpGpuGraphics
    PRIVATE
        ${CPU_SOURCES}
        ${CUDA_SOURCES}
)

set_target_properties(GpGpuGraphics
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

target_include_directories(GpGpuGraphics
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_compile_options(GpGpuGraphics
    PRIVATE
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:
            /W4
        >
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:
            -Wall -Wextra -Wpedantic
        >
        $<$<COMPILE_LANGUAGE:CUDA>:
            --expt-relaxed-constexpr --expt-extended-lambda
            -Xcompiler=-fno-omit-frame-pointer
        >
)

target_compile_definitions(GpGpuGraphics
    PRIVATE
        $<$<CONFIG:Debug>:DEBUG>
        $<$<CONFIG:Release>:NDEBUG>
        $<$<CONFIG:RelWithDebInfo>:NDEBUG>
)
