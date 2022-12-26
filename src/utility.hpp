#ifndef VULKAN_SANDBOX_UTILITY_HPP
#define VULKAN_SANDBOX_UTILITY_HPP

#include <glm/glm.hpp>
#include <vulkan/vulkan.hpp>

#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <optional>
#include <span>
#include <stdexcept>
#include <vector>

#if defined(_MSC_VER)
#define SANDBOX_PACKED
#elif defined(__GNUC__) || defined(__GNUG__) || defined(__clang__)
#define SANDBOX_PACKED __attribute__((packed))
#else
#error "Unsupported compiler. This project currently only supports MSVC, GCC, and Clang."
#endif

struct vertex {
    glm::vec3 position;
    glm::vec3 color;
    glm::vec2 tex_coord;

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
                .format   = vk::Format::eR32G32B32Sfloat,
                .offset   = offsetof(::vertex, position),
            },
            {
                .location = 1,
                .binding  = 0,
                .format   = vk::Format::eR32G32B32Sfloat,
                .offset   = offsetof(::vertex, color),
            },
            {
                .location = 2,
                .binding  = 0,
                .format   = vk::Format::eR32G32Sfloat,
                .offset   = offsetof(::vertex, tex_coord),
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

namespace vkm {

template<typename T>
glm::mat<4, 4, T, glm::defaultp> perspective(T vertical_fov, T aspect, T near)
{
    const T focal_length = static_cast<T>(1.0) / glm::tan(vertical_fov / static_cast<T>(2.0));

    const T x = focal_length / aspect;
    const T y = -focal_length;
    const T a = static_cast<T>(0.0);
    const T b = near;

    glm::mat<4, 4, T, glm::defaultp> result(static_cast<T>(0.0));
    result[0][0] = x;
    result[1][1] = y;
    result[2][2] = a;
    result[2][3] = -static_cast<T>(1.0);
    result[3][2] = b;

    return result;
}

}

inline std::vector<char> read_file(const char *filepath)
{
    std::ifstream file(filepath, std::ios::binary | std::ios::ate);

    if (!file) {
        throw std::runtime_error("Could not open file: " + std::string(filepath) + "!");
    }

    const auto end = file.tellg();
    file.seekg(0, std::ios::beg);
    const auto start = file.tellg();

    std::vector<char> buffer;

    const auto size = static_cast<std::size_t>(end - start);

    if (size == 0) {
        return buffer;
    }

    buffer.resize(size);
    file.read(buffer.data(), static_cast<std::streamsize>(size));

    return buffer;
}

template<std::size_t N>
constexpr auto le_bytes_to_uint(std::span<const char, N> bytes)
{
    static_assert(N <= sizeof(std::uint64_t), "integers above 64 bits are unsupported!");

    std::uint64_t result = 0;
    if constexpr (std::endian::native == std::endian::little) {
        std::memcpy(&result, bytes.data(), N);
    } else {
        for (std::size_t i = 0; i < N; ++i) {
            result |= static_cast<std::uint64_t>(bytes[N - i - 1]) << (8 * i);
        }
    }

    if constexpr (N == 1) {
        return static_cast<std::uint8_t>(result);
    } else if constexpr (N == 2) {
        return static_cast<std::uint16_t>(result);
    } else if constexpr (N <= 4) {
        return static_cast<std::uint32_t>(result);
    } else {
        return result;
    }
}

namespace tga {

#ifdef _MSC_VER
#pragma pack(push, 1)
#endif
struct SANDBOX_PACKED color_map_specification {
    std::uint16_t first_entry_index;
    std::uint16_t entry_count;
    std::uint8_t  color_depth;
};
#ifdef _MSC_VER
#pragma pack(pop)
#endif

static_assert(sizeof(color_map_specification) == 5, "tga::color_map_specification is not exactly 5 bytes!");

#ifdef _MSC_VER
#pragma pack(push, 1)
#endif
struct SANDBOX_PACKED image_specification {
    std::uint16_t x_origin;
    std::uint16_t y_origin;
    std::uint16_t width;
    std::uint16_t height;
    std::uint8_t  color_depth;
    std::uint8_t  descriptor;
};
#ifdef _MSC_VER
#pragma pack(pop)
#endif

static_assert(sizeof(image_specification) == 10, "tga::image_specification is not exactly 10 bytes!");

#ifdef _MSC_VER
#pragma pack(push, 1)
#endif
struct SANDBOX_PACKED header {
    std::uint8_t                 id_length;
    std::uint8_t                 color_map_type;
    std::uint8_t                 image_type;
    tga::color_map_specification color_map_specification;
    tga::image_specification     image_specification;
};
#ifdef _MSC_VER
#pragma pack(pop)
#endif

static_assert(sizeof(header) == 18, "tga::header is not exactly 18 bytes!");

#ifdef _MSC_VER
#pragma pack(push, 1)
#endif
struct SANDBOX_PACKED footer {
    std::uint32_t extension_offset;
    std::uint32_t developer_offset;
    char          signature[16];
    char          dot;
    char          nul;
};
#ifdef _MSC_VER
#pragma pack(pop)
#endif

struct file {
    std::vector<char> pixels;
    std::uint32_t     width;
    std::uint32_t     height;
    std::uint8_t      color_depth;

    std::size_t size() const
    {
        return ((static_cast<std::size_t>(width) * color_depth + 31) / 32) * 4 * height;
    }

    static file from(std::span<const char> bytes)
    {
        static_assert(std::endian::native == std::endian::little, "Big-endian systems are not yet supported!");

        if (bytes.size() < sizeof(tga::header)) {
            throw std::runtime_error("Too few bytes to be a TGA file!");
        }

        const auto header_bytes = bytes.subspan<0, sizeof(tga::header)>();
        tga::header header = { };
        std::memcpy(&header, header_bytes.data(), sizeof(header));

        if (header.image_type != 0x02) {
            throw std::runtime_error("Only uncompressed true-color TGA files are supported!");
        }

        const auto footer_bytes = bytes.subspan(bytes.size() - sizeof(tga::footer));
        tga::footer footer = { };
        std::memcpy(&footer, footer_bytes.data(), sizeof(footer));

        if (footer.extension_offset != 0 || footer.developer_offset != 0) {
            throw std::runtime_error("TGA extension area and developer area are not yet supported!");
        }

        const auto &image_specification = header.image_specification;

        const std::uint8_t color_depth = image_specification.color_depth;
        if (color_depth != 32) {
            throw std::runtime_error("Only 32-bit color depths TGA files are supported!");
        }

        tga::file file = {
            .width       = image_specification.width,
            .height      = image_specification.height,
            .color_depth = color_depth,
        };

        const std::size_t size = file.size();
        const auto pixel_bytes = bytes.subspan(sizeof(tga::header), size);
        file.pixels.resize(size);
        std::memcpy(file.pixels.data(), pixel_bytes.data(), size);

         return file;
    }
};

}

#endif // VULKAN_SANDBOX_UTILITY_HPP
