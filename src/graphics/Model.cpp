#include "Model.h"

#include <stdexcept>
#include <unordered_map>
#define TINYOBJLOADER_IMPLEMENTATION
#include <tinyobjloader/tiny_obj_loader.h>

using namespace ggg;

namespace
{
    struct IndexHash
    {
        size_t operator()(const tinyobj::index_t& idx) const noexcept
        {
            size_t h1 = std::hash<int>{}(idx.vertex_index);
            size_t h2 = std::hash<int>{}(idx.normal_index);
            size_t h3 = std::hash<int>{}(idx.texcoord_index);
            return h1 ^ (h2 << 1) ^ (h3 << 2);
        }
    };

    struct IndexEqual
    {
        bool operator()(const tinyobj::index_t& a, const tinyobj::index_t& b) const noexcept
        {
            return a.vertex_index == b.vertex_index &&
                   a.normal_index == b.normal_index &&
                   a.texcoord_index == b.texcoord_index;
        }
    };
}

Model::Model(const std::string& modelName)
{
    std::string basePath = "data/models/" + modelName + "/";
    std::string objPath = basePath + modelName + ".obj";

    tinyobj::ObjReaderConfig config;
    config.mtl_search_path = basePath;

    tinyobj::ObjReader reader;
    if (!reader.ParseFromFile(objPath, config))
    {
        throw std::runtime_error("Failed to load OBJ: " + objPath + " (" + reader.Error() + ")");
    }

    const tinyobj::attrib_t& attrib = reader.GetAttrib();
    const std::vector<tinyobj::shape_t>& shapes = reader.GetShapes();

    m_vertices.clear();
    m_indices.clear();
    m_vertices.reserve(attrib.vertices.size() / 3);

    std::unordered_map<tinyobj::index_t, std::uint32_t, IndexHash, IndexEqual> indexMap;

    for (const tinyobj::shape_t& shape : shapes)
    {
        for (const tinyobj::index_t& idx : shape.mesh.indices)
        {
            auto it = indexMap.find(idx);
            if (it == indexMap.end())
            {
                Vertex v{};

                if (idx.vertex_index >= 0)
                {
                    std::size_t vOffset = static_cast<std::size_t>(idx.vertex_index) * 3;
                    v.p.x() = attrib.vertices.at(vOffset + 0);
                    v.p.y() = attrib.vertices.at(vOffset + 1);
                    v.p.z() = attrib.vertices.at(vOffset + 2);
                }

                if (idx.normal_index >= 0)
                {
                    std::size_t nOffset = static_cast<std::size_t>(idx.normal_index) * 3;
                    v.n.x() = attrib.normals.at(nOffset + 0);
                    v.n.y() = attrib.normals.at(nOffset + 1);
                    v.n.z() = attrib.normals.at(nOffset + 2);
                }

                if (idx.texcoord_index >= 0)
                {
                    size_t tOffset = static_cast<std::size_t>(idx.texcoord_index) * 2;
                    v.uv.x() = attrib.texcoords.at(tOffset + 0);
                    v.uv.y() = attrib.texcoords.at(tOffset + 1);
                }

                const std::uint32_t newIndex = static_cast<std::uint32_t>(m_vertices.size());
                m_vertices.push_back(v);
                m_indices.push_back(newIndex);
                indexMap.emplace(idx, newIndex);
            }
            else
            {
                m_indices.push_back(it->second);
            }
        }
    }

    m_cudaVertexBuffer = CudaBuffer<Vertex>(m_vertices.size());
    m_cudaVertexBuffer.CopyToGpuBuffer(m_vertices);

    m_cudaIndexBuffer = CudaBuffer<std::uint32_t>(m_indices.size());
    m_cudaIndexBuffer.CopyToGpuBuffer(m_indices);
}

Model::~Model() = default;

const std::vector<Vertex>& Model::GetVertices() const noexcept
{
    return m_vertices;
}

const std::vector<std::uint32_t>& Model::GetIndices() const noexcept
{
    return m_indices;
}

std::pair<const CudaBuffer<Vertex>&, std::size_t> Model::GetCudaVertexBuffer() const noexcept
{
    return { m_cudaVertexBuffer, m_vertices.size() };
}

std::pair<const CudaBuffer<std::uint32_t>&, std::size_t> Model::GetCudaIndexBuffer() const noexcept
{
    return { m_cudaIndexBuffer, m_indices.size() };
}
