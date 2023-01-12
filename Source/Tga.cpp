#include "Tga.hpp"

#include "Utility.hpp"

#include <span>

Tga::Image Tga::Image::Load(const char *filepath)
{
    static_assert(std::endian::native == std::endian::little, "Big-endian systems are not yet supported");

    std::vector<std::uint8_t> file = ReadFile(filepath);
    const std::span<const std::uint8_t> bytes = file;

    if (file.size() < HeaderSize)
    {
        throw std::runtime_error("Too few bytes to be a TGA image");
    }

    constexpr std::size_t idLengthOffset = 0;
    constexpr std::size_t colorMapTypeOffset = idLengthOffset + sizeof(Tga::Header::IdLength);
    constexpr std::size_t imageTypeOffset = colorMapTypeOffset + sizeof(Tga::Header::ColorMapType);
    constexpr std::size_t firstEntryIndexOffset = imageTypeOffset + sizeof(Tga::Header::ImageType);
    constexpr std::size_t entryCountOffset = firstEntryIndexOffset + sizeof(ColorMapSpecification::FirstEntryIndex);
    constexpr std::size_t colorMapDepthOffset = entryCountOffset + sizeof(ColorMapSpecification::EntryCount);
    constexpr std::size_t xOriginOffset = colorMapDepthOffset + sizeof(ColorMapSpecification::ColorDepth);
    constexpr std::size_t yOriginOffset = xOriginOffset + sizeof(ImageSpecification::XOrigin);
    constexpr std::size_t widthOffset = yOriginOffset + sizeof(ImageSpecification::YOrigin);
    constexpr std::size_t heightOffset = widthOffset + sizeof(ImageSpecification::Width);
    constexpr std::size_t imageDepthOffset = heightOffset + sizeof(ImageSpecification::Height);
    constexpr std::size_t descriptorOffset = imageDepthOffset + sizeof(ImageSpecification::ColorDepth);
    constexpr std::size_t headerEndOffset = descriptorOffset + sizeof(ImageSpecification::Descriptor);
    static_assert(headerEndOffset == HeaderSize);

    const Tga::Header header = {
        .IdLength              = bytes[0],
        .ColorMapType          = bytes[1],
        .ImageType             = bytes[2],
        .ColorMapSpecification = {
            .FirstEntryIndex   = BytesToUInt(bytes.subspan<firstEntryIndexOffset, sizeof(Tga::ColorMapSpecification::FirstEntryIndex)>()),
            .EntryCount        = BytesToUInt(bytes.subspan<entryCountOffset, sizeof(Tga::ColorMapSpecification::EntryCount)>()),
            .ColorDepth        = bytes[colorMapDepthOffset],
        },
        .ImageSpecification    = {
            .XOrigin           = BytesToUInt(bytes.subspan<xOriginOffset, sizeof(ImageSpecification::XOrigin)>()),
            .YOrigin           = BytesToUInt(bytes.subspan<yOriginOffset, sizeof(ImageSpecification::YOrigin)>()),
            .Width             = BytesToUInt(bytes.subspan<widthOffset, sizeof(ImageSpecification::Width)>()),
            .Height            = BytesToUInt(bytes.subspan<heightOffset, sizeof(ImageSpecification::Height)>()),
            .ColorDepth        = BytesToUInt(bytes.subspan<imageDepthOffset, sizeof(ImageSpecification::ColorDepth)>()),
            .Descriptor        = BytesToUInt(bytes.subspan<descriptorOffset, sizeof(ImageSpecification::Descriptor)>()),
        },
    };

    if (header.ImageType != 0x02)
    {
        throw std::runtime_error("Only uncompressed true-color TGA files are supported");
    }

    const auto &imageSpecification = header.ImageSpecification;

    const std::uint8_t colorDepth = imageSpecification.ColorDepth;
    if (colorDepth != 32)
    {
        throw std::runtime_error("Only 32-bit color depth TGA files are supported");
    }

    const std::ptrdiff_t dataOffset = (std::ptrdiff_t)HeaderSize + header.IdLength;

    constexpr std::size_t extensionOffsetOffset = 0;
    constexpr std::size_t developerOffsetOffset = extensionOffsetOffset + sizeof(Tga::Footer::ExtensionOffset);
    constexpr std::size_t signatureOffset = developerOffsetOffset + sizeof(Tga::Footer::DeveloperOffset);
    constexpr std::size_t dotOffset = signatureOffset + Signature.size();
    constexpr std::size_t nullOffset = dotOffset + sizeof(char);
    constexpr std::size_t footerEndOffset = nullOffset + sizeof(char);
    static_assert(footerEndOffset == FooterSize);

    const auto footerBytes = bytes.subspan(bytes.size() - FooterSize);
    const Tga::Footer footer = {
        .ExtensionOffset = BytesToUInt(footerBytes.subspan<extensionOffsetOffset, sizeof(Tga::Footer::ExtensionOffset)>()),
        .DeveloperOffset = BytesToUInt(footerBytes.subspan<developerOffsetOffset, sizeof(Tga::Footer::DeveloperOffset)>()),
    };

    if (footer.ExtensionOffset != 0 || footer.DeveloperOffset != 0)
    {
        throw std::runtime_error("TGA extension area and developer area are not yet supported");
    }

    Tga::Image image = {
        .Width      = imageSpecification.Width,
        .Height     = imageSpecification.Height,
        .ColorDepth = colorDepth,
    };

    const std::size_t size = image.GetSize();
    file.erase(file.begin(), file.begin() + dataOffset);
    file.resize(size);

    image.Pixels = std::move(file);

    return image;
}
