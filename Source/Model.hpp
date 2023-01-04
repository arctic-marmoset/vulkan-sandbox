#ifndef VULKAN_RENDERER_MODEL_HPP
#define VULKAN_RENDERER_MODEL_HPP

#include "Device.hpp"
#include "Tga.hpp"

#include <tiny_obj_loader.h>

#include <glm/glm.hpp>
#include <vulkan/vulkan.hpp>

#include <array>
#include <numeric>
#include <vector>

struct Vertex
{
    glm::vec3 Position;
    glm::vec2 TexCoord;

    static constexpr vk::VertexInputBindingDescription GetBindingDescription()
    {
        return {
            .binding   = 0,
            .stride    = sizeof(Vertex),
            .inputRate = vk::VertexInputRate::eVertex,
        };
    }

    static constexpr auto GetAttributeDescriptions()
    {
        return std::to_array<vk::VertexInputAttributeDescription>({
            {
                .location = 0,
                .binding  = 0,
                .format   = vk::Format::eR32G32B32Sfloat,
                .offset   = offsetof(Vertex, Position),
            },
            {
                .location = 1,
                .binding  = 0,
                .format   = vk::Format::eR32G32Sfloat,
                .offset   = offsetof(Vertex, TexCoord),
            },
        });
    }
};

struct Model
{
    struct
    {
        std::vector<Vertex> Data;
        Buffer Buffer;
    } Vertices;

    struct
    {
        Buffer Buffer;
        std::uint32_t Count = 0;
    } Indices;

    struct
    {
        Tga::Image Data;
        Image Image;
        vk::ImageView View;
        vk::DescriptorSet DescriptorSet;
    } Texture;

    void Load(const char *meshPath, const char *texturePath);

    [[nodiscard("Returned buffers must be freed once command buffer has completed execution")]]
    std::array<Buffer, 3> Init(
        const Device &device,
        vk::DescriptorSetLayout textureSamplerSetLayout,
        vk::DescriptorPool descriptorPool,
        vk::Sampler textureSampler,
        vk::CommandBuffer commandBuffer
    );

    void Destroy(vk::Device device)
    {
        Vertices.Buffer.Destroy(device);
        Indices.Buffer.Destroy(device);

        Texture.Image.Destroy(device);
        device.destroyImageView(Texture.View);
    }
};

#endif // !VULKAN_RENDERER_MODEL_HPP
