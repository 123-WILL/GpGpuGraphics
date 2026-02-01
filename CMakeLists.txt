cmake_minimum_required(VERSION 3.27)
project(GpGpuGraphics LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CPU_SOURCES
        src/core/Window.cpp
        src/core/Window.h
        src/graphics/Model.cpp
        src/graphics/Model.h
        src/math/Matrix.h
        src/math/Vector.h
        src/main.cpp
)

set(CUDA_SOURCES
        src/cuda/kernels.cu
        src/cuda/kernels.cuh
)

find_package(CUDAToolkit REQUIRED)

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
        ${CMAKE_CURRENT_SOURCE_DIR}/external/include
        ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(GpGpuGraphics
    PRIVATE
        d3d11
        dxgi
        CUDA::cuda_driver
        CUDA::cudart
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
