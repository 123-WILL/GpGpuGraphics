#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include "cuda/CudaBuffer.h"
#include "math/Vector.h"

namespace ggg
{
    struct Vertex
    {
        Vec3f p{};
        Vec3f n{};
        Vec2f uv{};
    };

    class Model
    {
    public:
        explicit Model(const std::string& modelName);
        ~Model();

        [[nodiscard]] const std::vector<Vertex>& GetVertices() const noexcept;
        [[nodiscard]] const std::vector<std::uint32_t>& GetIndices() const noexcept;
        [[nodiscard]] std::pair<const CudaBuffer<Vertex>&, std::size_t> GetCudaVertexBuffer() const noexcept;
        [[nodiscard]] std::pair<const CudaBuffer<std::uint32_t>&, std::size_t> GetCudaIndexBuffer() const noexcept;

    private:
        std::vector<Vertex> m_vertices;
        std::vector<std::uint32_t> m_indices;
        CudaBuffer<Vertex> m_cudaVertexBuffer{};
        CudaBuffer<std::uint32_t> m_cudaIndexBuffer{};
    };
}
