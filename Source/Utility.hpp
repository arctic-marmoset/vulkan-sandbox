#ifndef VULKAN_RENDERER_UTILITY_HPP
#define VULKAN_RENDERER_UTILITY_HPP

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
#define VULKAN_RENDERER_PACKED
#elif defined(__GNUC__) || defined(__GNUG__) || defined(__clang__)
#define VULKAN_RENDERER_PACKED __attribute__((packed))
#else
#error "Unsupported compiler. This project currently only supports MSVC, GCC, and Clang."
#endif

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

struct QueueFamilyIndices
{
    std::optional<std::uint32_t> Graphics;
    std::optional<std::uint32_t> Present;

    [[nodiscard]]
    bool IsComplete() const
    {
        return Graphics.has_value()
               && Present.has_value();
    }
};

struct SwapChainSupportDetails
{
    vk::SurfaceCapabilitiesKHR Capabilities;
    std::vector<vk::SurfaceFormatKHR> Formats;
    std::vector<vk::PresentModeKHR> PresentModes;
};

struct UniformBufferObject
{
    alignas(16) glm::mat4 Model;
    alignas(16) glm::mat4 View;
    alignas(16) glm::mat4 Proj;
};

namespace vkm
{

    template<typename T>
    glm::mat<4, 4, T, glm::defaultp> perspective(T verticalFov, T aspect, T near)
    {
        const T focalLength = static_cast<T>(1.0) / glm::tan(verticalFov / static_cast<T>(2.0));

        const T x = focalLength / aspect;
        const T y = -focalLength;
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

inline std::vector<char> ReadFile(const char *filepath)
{
    std::ifstream file(filepath, std::ios::binary | std::ios::ate);

    if (!file)
    {
        throw std::runtime_error("Could not open file: " + std::string(filepath));
    }

    const auto end = file.tellg();
    file.seekg(0, std::ios::beg);
    const auto start = file.tellg();

    std::vector<char> buffer;

    const auto size = static_cast<std::size_t>(end - start);

    if (size == 0)
    {
        return buffer;
    }

    buffer.resize(size);
    file.read(buffer.data(), static_cast<std::streamsize>(size));

    return buffer;
}

template<std::size_t N>
constexpr auto LittleEndianBytesToUint(std::span<const char, N> bytes)
{
    static_assert(N <= sizeof(std::uint64_t), "Integers above 64 bits are unsupported");

    std::uint64_t result = 0;
    if constexpr (std::endian::native == std::endian::little)
    {
        std::memcpy(&result, bytes.data(), N);
    }
    else
    {
        for (std::size_t i = 0; i < N; ++i)
        {
            result |= static_cast<std::uint64_t>(bytes[N - i - 1]) << (8 * i);
        }
    }

    if constexpr (N == 1)
    {
        return static_cast<std::uint8_t>(result);
    }
    else if constexpr (N == 2)
    {
        return static_cast<std::uint16_t>(result);
    }
    else if constexpr (N <= 4)
    {
        return static_cast<std::uint32_t>(result);
    }
    else
    {
        return result;
    }
}

namespace Tga
{

#ifdef _MSC_VER
#pragma pack(push, 1)
#endif
    struct VULKAN_RENDERER_PACKED ColorMapSpecification
    {
        std::uint16_t FirstEntryIndex;
        std::uint16_t EntryCount;
        std::uint8_t  ColorDepth;
    };
#ifdef _MSC_VER
#pragma pack(pop)
#endif

    static_assert(sizeof(ColorMapSpecification) == 5, "Tga::ColorMapSpecification is not exactly 5 bytes");

#ifdef _MSC_VER
#pragma pack(push, 1)
#endif
    struct VULKAN_RENDERER_PACKED ImageSpecification
    {
        std::uint16_t XOrigin;
        std::uint16_t YOrigin;
        std::uint16_t Width;
        std::uint16_t Height;
        std::uint8_t  ColorDepth;
        std::uint8_t  Descriptor;
    };
#ifdef _MSC_VER
#pragma pack(pop)
#endif

    static_assert(sizeof(ImageSpecification) == 10, "Tga::ImageSpecification is not exactly 10 bytes");

#ifdef _MSC_VER
#pragma pack(push, 1)
#endif
    struct VULKAN_RENDERER_PACKED Header
    {
        std::uint8_t IdLength;
        std::uint8_t ColorMapType;
        std::uint8_t ImageType;
        Tga::ColorMapSpecification ColorMapSpecification;
        Tga::ImageSpecification ImageSpecification;
    };
#ifdef _MSC_VER
#pragma pack(pop)
#endif

    static_assert(sizeof(Header) == 18, "Tga::Header is not exactly 18 bytes");

#ifdef _MSC_VER
#pragma pack(push, 1)
#endif
    struct VULKAN_RENDERER_PACKED Footer
    {
        std::uint32_t ExtensionOffset;
        std::uint32_t DeveloperOffset;
        char Signature[16];
        char Dot;
        char Null;
    };
#ifdef _MSC_VER
#pragma pack(pop)
#endif

    struct File
    {
        std::vector<char> Pixels;
        std::uint32_t Width;
        std::uint32_t Height;
        std::uint8_t ColorDepth;

        std::size_t GetSize() const
        {
            return ((static_cast<std::size_t>(Width) * ColorDepth + 31) / 32) * 4 * Height;
        }

        static File CreateFrom(std::span<const char> bytes)
        {
            static_assert(std::endian::native == std::endian::little, "Big-endian systems are not yet supported");

            if (bytes.size() < sizeof(Tga::Header))
            {
                throw std::runtime_error("Too few bytes to be a TGA file");
            }

            const auto headerBytes = bytes.subspan<0, sizeof(Tga::Header)>();
            Tga::Header header = { };
            std::memcpy(&header, headerBytes.data(), sizeof(header));

            if (header.ImageType != 0x02)
            {
                throw std::runtime_error("Only uncompressed true-color TGA files are supported");
            }

            const auto footerBytes = bytes.subspan(bytes.size() - sizeof(Tga::Footer));
            Tga::Footer footer = { };
            std::memcpy(&footer, footerBytes.data(), sizeof(footer));

            if (footer.ExtensionOffset != 0 || footer.DeveloperOffset != 0)
            {
                throw std::runtime_error("TGA extension area and developer area are not yet supported");
            }

            const auto &imageSpecification = header.ImageSpecification;

            const std::uint8_t colorDepth = imageSpecification.ColorDepth;
            if (colorDepth != 32)
            {
                throw std::runtime_error("Only 32-bit color depths TGA files are supported");
            }

            Tga::File file = {
                .Width       = imageSpecification.Width,
                .Height      = imageSpecification.Height,
                .ColorDepth = colorDepth,
            };

            const std::size_t size = file.GetSize();
            const auto pixelBytes = bytes.subspan(sizeof(Tga::Header), size);
            file.Pixels.resize(size);
            std::memcpy(file.Pixels.data(), pixelBytes.data(), size);

            return file;
        }
    };
}

#endif // !VULKAN_RENDERER_UTILITY_HPP
