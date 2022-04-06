#ifndef VULKAN_SANDBOX_UTILITY_HPP
#define VULKAN_SANDBOX_UTILITY_HPP

#include <glm/glm.hpp>
#include <vulkan/vulkan.hpp>

#include <cstddef>
#include <fstream>
#include <optional>
#include <vector>

struct vertex {
    glm::vec2 position;
    glm::vec3 color;

    static constexpr vk::VertexInputBindingDescription binding_description()
    {
        return {
            .binding   = 0,
            .stride    = sizeof(::vertex),
            .inputRate = vk::VertexInputRate::eVertex,
        };
    }

    static constexpr auto attribute_descriptions()
    {
        return std::to_array<vk::VertexInputAttributeDescription>({
            {
                .location = 0,
                .binding  = 0,
                .format   = vk::Format::eR32G32Sfloat,
                .offset   = offsetof(::vertex, position),
            },
            {
                .location = 1,
                .binding  = 0,
                .format   = vk::Format::eR32G32B32Sfloat,
                .offset   = offsetof(::vertex, color),
            },
        });
    }
};

struct queue_family_indices {
    std::optional<std::uint32_t> graphics_family;
    std::optional<std::uint32_t> present_family;

    [[nodiscard]]
    bool complete() const
    {
        return graphics_family.has_value()
            && present_family.has_value();
    }
};

struct swapchain_support_details {
    vk::SurfaceCapabilitiesKHR capabilities;
    std::vector<vk::SurfaceFormatKHR> formats;
    std::vector<vk::PresentModeKHR> present_modes;
};

struct uniform_buffer_object {
    alignas(16) glm::mat4 model;
    alignas(16) glm::mat4 view;
    alignas(16) glm::mat4 proj;
};

inline std::vector<std::byte> read_file(const char *filepath)
{
    std::ifstream file(filepath, std::ios::binary | std::ios::ate);

    if (!file) {
        throw std::runtime_error("Could not open file!");
    }

    const auto end = file.tellg();
    file.seekg(0, std::ios::beg);
    const auto start = file.tellg();

    std::vector<std::byte> buffer;

    const auto size = static_cast<std::size_t>(end - start);

    if (size == 0) {
        return buffer;
    }

    buffer.resize(size);
    file.read(reinterpret_cast<char *>(buffer.data()), static_cast<std::streamsize>(size));

    return buffer;
}

#endif // VULKAN_SANDBOX_UTILITY_HPP
