#ifndef VULKAN_RENDERER_TGA_HPP
#define VULKAN_RENDERER_TGA_HPP

#include "Utility.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace Tga
{
    struct ColorMapSpecification
    {
        std::uint16_t FirstEntryIndex;
        std::uint16_t EntryCount;
        std::uint8_t  ColorDepth;
    };

    struct ImageSpecification
    {
        std::uint16_t XOrigin;
        std::uint16_t YOrigin;
        std::uint16_t Width;
        std::uint16_t Height;
        std::uint8_t  ColorDepth;
        std::uint8_t  Descriptor;
    };

    struct Header
    {
        std::uint8_t IdLength;
        std::uint8_t ColorMapType;
        std::uint8_t ImageType;
        Tga::ColorMapSpecification ColorMapSpecification;
        Tga::ImageSpecification ImageSpecification;
    };

    constexpr std::size_t HeaderSize = 18;

    struct Footer
    {
        std::uint32_t ExtensionOffset;
        std::uint32_t DeveloperOffset;
    };

    constexpr std::size_t FooterSize = 26;

    constexpr std::array Signature =
        { 'T', 'R', 'U', 'E', 'V', 'I', 'S', 'I', 'O', 'N', '-', 'X', 'F', 'I', 'L', 'E'  };

    struct Image
    {
        std::vector<std::uint8_t> Pixels;
        std::uint32_t Width;
        std::uint32_t Height;
        std::uint8_t ColorDepth;

        std::size_t GetSize() const
        {
            return ((static_cast<std::size_t>(Width) * ColorDepth + 31) / 32) * 4 * Height;
        }

        static Image Load(const char *filepath);
    };
}

#endif // !VULKAN_RENDERER_TGA_HPP
