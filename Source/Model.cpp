#include "Model.hpp"

#include <spdlog/spdlog.h>

void Model::Load(const char *meshPath, const char *texturePath)
{
    tinyobj::ObjReader reader;
    if (!reader.ParseFromFile(meshPath))
    {
        throw std::runtime_error("Failed to load model");
    }

    if (!reader.Warning().empty())
    {
        spdlog::warn("TinyObjReader: {}", reader.Warning());
    }

    const tinyobj::attrib_t &attrib = reader.GetAttrib();
    const std::vector<tinyobj::shape_t> &shapes = reader.GetShapes();
    assert("Only OBJ files with a single shape are currently supported" && shapes.size() == 1);
    const tinyobj::shape_t &shape = shapes[0];

    std::size_t indexOffset = 0;
    for (std::size_t faceIndex = 0; faceIndex < shape.mesh.num_face_vertices.size(); ++faceIndex)
    {
        const std::size_t faceVertexCount = shape.mesh.num_face_vertices[faceIndex];

        for (std::size_t vertexIndex = 0; vertexIndex < faceVertexCount; ++vertexIndex)
        {
            constexpr std::size_t vertexDimensionCount = 3;
            const tinyobj::index_t index = shape.mesh.indices[vertexIndex + indexOffset];
            const tinyobj::real_t xPos = attrib.vertices[vertexDimensionCount * (std::size_t)index.vertex_index + 0];
            const tinyobj::real_t yPos = attrib.vertices[vertexDimensionCount * (std::size_t)index.vertex_index + 1];
            const tinyobj::real_t zPos = attrib.vertices[vertexDimensionCount * (std::size_t)index.vertex_index + 2];

            assert("Meshes should contain texture coordinate info" && index.texcoord_index >= 0);
            constexpr std::size_t textureDimensionCount = 2;
            const tinyobj::real_t u = attrib.texcoords[textureDimensionCount * (std::size_t)index.texcoord_index + 0];
            const tinyobj::real_t v = attrib.texcoords[textureDimensionCount * (std::size_t)index.texcoord_index + 1];

            const Vertex vertex = {
                .Position = { xPos, yPos, zPos },
                .TexCoord = { u, v },
            };

            Vertices.Data.push_back(vertex);
        }

        indexOffset += faceVertexCount;
    }

    Indices.Count = static_cast<std::uint32_t>(Vertices.Data.size());
    Texture.Data = Tga::Image::Load(texturePath);
}

std::array<Buffer, 3> Model::Init(
    const Device &device,
    vk::DescriptorSetLayout textureSamplerSetLayout,
    vk::DescriptorPool descriptorPool,
    vk::Sampler textureSampler,
    vk::CommandBuffer commandBuffer
)
{
    const vk::DeviceSize verticesSize = SizeInBytes(Vertices.Data);

    const Buffer vertexStagingBuffer = device.CreateBuffer(
        verticesSize,
        vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent
    );

    std::vector<std::uint32_t> indices(Indices.Count);
    std::iota(indices.begin(), indices.end(), 0);

    const vk::DeviceSize indicesSize = SizeInBytes(indices);

    const Buffer indexStagingBuffer = device.CreateBuffer(
        indicesSize,
        vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent
    );

    const Buffer textureStagingBuffer = device.CreateBuffer(
        Texture.Data.GetSize(),
        vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent
    );

    WithMappedMemory(device.GetHandle(), vertexStagingBuffer.Memory, 0, verticesSize, [&](void *destination)
    {
        std::memcpy(destination, Vertices.Data.data(), verticesSize);
    });

    WithMappedMemory(device.GetHandle(), indexStagingBuffer.Memory, 0, indicesSize, [&](void *destination)
    {
        std::memcpy(destination, indices.data(), indicesSize);
    });

    WithMappedMemory(device.GetHandle(), textureStagingBuffer.Memory, 0, Texture.Data.GetSize(), [&](void *destination)
    {
        std::memcpy(destination, Texture.Data.Pixels.data(), Texture.Data.GetSize());
    });

    Vertices.Buffer = device.CreateBuffer(
        verticesSize,
        vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eVertexBuffer,
        vk::MemoryPropertyFlagBits::eDeviceLocal
    );

    Indices.Buffer = device.CreateBuffer(
        indicesSize,
        vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eIndexBuffer,
        vk::MemoryPropertyFlagBits::eDeviceLocal
    );

    Texture.Image = device.CreateImage(
        Texture.Data.Width,
        Texture.Data.Height,
        vk::Format::eB8G8R8A8Srgb,
        vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled,
        vk::MemoryPropertyFlagBits::eDeviceLocal
    );

    Texture.View = device.CreateImageView(
        Texture.Image.Handle,
        vk::Format::eB8G8R8A8Srgb,
        vk::ImageAspectFlagBits::eColor
    );

    CopyBuffer(
        commandBuffer,
        vertexStagingBuffer.Handle,
        Vertices.Buffer.Handle,
        verticesSize
    );

    CopyBuffer(
        commandBuffer,
        indexStagingBuffer.Handle,
        Indices.Buffer.Handle,
        indicesSize
    );

    TransitionImageLayout(
        commandBuffer,
        Texture.Image.Handle,
        vk::ImageLayout::eUndefined,
        vk::ImageLayout::eTransferDstOptimal
    );

    CopyBufferToImage(
        commandBuffer,
        textureStagingBuffer.Handle,
        Texture.Image.Handle,
        Texture.Data.Width,
        Texture.Data.Height
    );

    TransitionImageLayout(
        commandBuffer,
        Texture.Image.Handle,
        vk::ImageLayout::eTransferDstOptimal,
        vk::ImageLayout::eShaderReadOnlyOptimal
    );

    const std::array setLayouts = { textureSamplerSetLayout };

    const auto textureAllocInfo = vk::DescriptorSetAllocateInfo()
        .setDescriptorPool(descriptorPool)
        .setDescriptorSetCount(1)
        .setSetLayouts(setLayouts);

    (void)device.GetHandle().allocateDescriptorSets(&textureAllocInfo, &Texture.DescriptorSet);

    const vk::DescriptorImageInfo textureImageInfo = {
        .sampler     = textureSampler,
        .imageView   = Texture.View,
        .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
    };

    const vk::WriteDescriptorSet textureDescriptorWrite = {
        .dstSet          = Texture.DescriptorSet,
        .dstBinding      = 0,
        .dstArrayElement = 0,
        .descriptorCount = 1,
        .descriptorType  = vk::DescriptorType::eCombinedImageSampler,
        .pImageInfo      = &textureImageInfo,
    };

    device.GetHandle().updateDescriptorSets({ textureDescriptorWrite }, { });

    return {
        vertexStagingBuffer,
        indexStagingBuffer,
        textureStagingBuffer,
    };
}
