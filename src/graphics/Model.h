#pragma once

#include <cstdint>
#include <string>
#include <vector>
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

        [[nodiscard]] const std::vector<Vertex>& GetVertices() const noexcept;
        [[nodiscard]] const std::vector<std::uint32_t>& GetIndices() const noexcept;
        [[nodiscard]] std::pair<const Vertex*, std::size_t> GetCudaVertexBuffer() const noexcept;
        [[nodiscard]] std::pair<const std::uint32_t*, std::size_t> GetCudaIndexBuffer() const noexcept;

    private:
        std::vector<Vertex> m_vertices;
        std::vector<std::uint32_t> m_indices;
        Vertex* m_cudaVertexBuffer{};
        std::uint32_t* m_cudaIndexBuffer{};
    };
}
