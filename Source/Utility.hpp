#ifndef VULKAN_RENDERER_UTILITY_HPP
#define VULKAN_RENDERER_UTILITY_HPP

#include "Device.hpp"
#include "Tga.hpp"

#include <glm/glm.hpp>
#include <vulkan/vulkan.hpp>

#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <optional>
#include <set>
#include <span>
#include <stdexcept>
#include <vector>

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

template<typename Container>
constexpr std::size_t SizeInBytes(const Container &container)
{
    return std::size(container) * sizeof(typename Container::value_type);
}

template<typename T, std::size_t N>
constexpr std::size_t SizeInBytes(const T (&container)[N])
{
    return sizeof(container);
}

template<typename Callback>
void WithMappedMemory(
    vk::Device device,
    vk::DeviceMemory memory,
    std::uint32_t offset,
    vk::DeviceSize size,
    Callback &&callback
)
{
    void *mappedMemory = device.mapMemory(memory, offset, size);

    if (!mappedMemory)
    {
        throw std::runtime_error("Failed to map memory");
    }

    callback(mappedMemory);

    device.unmapMemory(memory);
}

inline void CopyBuffer(vk::CommandBuffer commandBuffer, vk::Buffer source, vk::Buffer destination, vk::DeviceSize size)
{
    const vk::BufferCopy copyRegion = {
        .srcOffset = 0,
        .dstOffset = 0,
        .size      = size,
    };

    commandBuffer.copyBuffer(source, destination, copyRegion);
}

inline void CopyBufferToImage(
    vk::CommandBuffer commandBuffer,
    VkBuffer buffer,
    VkImage image,
    uint32_t width,
    uint32_t height
)
{
    const vk::BufferImageCopy region = {
        .bufferOffset       = 0,
        .bufferRowLength    = 0,
        .bufferImageHeight  = 0,
        .imageSubresource   = {
            .aspectMask     = vk::ImageAspectFlagBits::eColor,
            .mipLevel       = 0,
            .baseArrayLayer = 0,
            .layerCount     = 1,
        },
        .imageOffset        = {
            .x              = 0,
            .y              = 0,
            .z              = 0,
        },
        .imageExtent        = {
            .width          = width,
            .height         = height,
            .depth          = 1,
        },
    };

    commandBuffer.copyBufferToImage(buffer, image, vk::ImageLayout::eTransferDstOptimal, { region });
}

inline void TransitionImageLayout(
    vk::CommandBuffer commandBuffer,
    vk::Image image,
    vk::ImageLayout oldLayout,
    vk::ImageLayout newLayout,
    const vk::ImageSubresourceRange &subresourceRange = {
        .aspectMask      = vk::ImageAspectFlagBits::eColor,
        .baseMipLevel    = 0,
        .levelCount      = 1,
        .baseArrayLayer  = 0,
        .layerCount      = 1,
    }
)
{
    vk::ImageMemoryBarrier barrier = {
        .oldLayout           = oldLayout,
        .newLayout           = newLayout,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image               = image,
        .subresourceRange    = subresourceRange,
    };

    vk::PipelineStageFlags sourceStage;
    vk::PipelineStageFlags destinationStage;

    if (oldLayout == vk::ImageLayout::eUndefined
        && newLayout == vk::ImageLayout::eTransferDstOptimal)
    {
        barrier.srcAccessMask = vk::AccessFlagBits::eNone;
        barrier.dstAccessMask = vk::AccessFlagBits::eTransferWrite;

        sourceStage = vk::PipelineStageFlagBits::eTopOfPipe;
        destinationStage = vk::PipelineStageFlagBits::eTransfer;
    }
    else if (oldLayout == vk::ImageLayout::eTransferDstOptimal
             && newLayout == vk::ImageLayout::eShaderReadOnlyOptimal)
    {
        barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
        barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

        sourceStage = vk::PipelineStageFlagBits::eTransfer;
        destinationStage = vk::PipelineStageFlagBits::eFragmentShader;
    }
    else
    {
        throw std::invalid_argument("Unsupported layout transition");
    }

    commandBuffer.pipelineBarrier(
        sourceStage,
        destinationStage,
        { },
        { },
        { },
        { barrier }
    );
}

inline std::pair<Image, Buffer> LoadCubeMap(
    const std::array<const char *, 6> &filepaths,
    const Device &device,
    vk::CommandBuffer commandBuffer
)
{
    const std::array<Tga::Image, 6> images = {
        Tga::Image::Load(filepaths[0]),
        Tga::Image::Load(filepaths[1]),
        Tga::Image::Load(filepaths[2]),
        Tga::Image::Load(filepaths[3]),
        Tga::Image::Load(filepaths[4]),
        Tga::Image::Load(filepaths[5]),
    };

    // TODO: Make sure dimensions are all the same.

    const vk::DeviceSize layerSize = images[0].GetSize();
    const vk::DeviceSize cubeMapSize = layerSize * 6;

    const Buffer stagingBuffer = device.CreateBuffer(
        cubeMapSize,
        vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent
    );

    WithMappedMemory(device.GetHandle(), stagingBuffer.Memory, 0, cubeMapSize, [&](void *memory)
    {
        char *destination = static_cast<char *>(memory);

        for (const Tga::Image &image : images)
        {
            std::memcpy(destination, image.Pixels.data(), layerSize);
            destination += layerSize;
        }
    });

    const std::uint32_t layerWidth = images[0].Width;
    const std::uint32_t layerHeight = images[0].Height;

    const Image image = device.CreateCubeMap(
        layerWidth,
        layerHeight,
        vk::Format::eB8G8R8A8Srgb,
        vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled,
        vk::MemoryPropertyFlagBits::eDeviceLocal
    );

    // TODO: Handle multiple mip levels.
    std::array<vk::BufferImageCopy, 6> copyRegions;
    for (std::uint32_t face = 0; face < 6; ++face)
    {
        const std::size_t offset = face * layerSize;

        copyRegions[face] = vk::BufferImageCopy()
            .setBufferOffset(offset)
            .setImageSubresource(
                vk::ImageSubresourceLayers()
                    .setAspectMask(vk::ImageAspectFlagBits::eColor)
                    .setMipLevel(0)
                    .setBaseArrayLayer(face)
                    .setLayerCount(1)
            )
            .setImageExtent({ layerWidth, layerHeight, 1 });
    }

    constexpr auto imageSubresourceRange = vk::ImageSubresourceRange()
        .setAspectMask(vk::ImageAspectFlagBits::eColor)
        .setLevelCount(1)
        .setLayerCount(6);

    TransitionImageLayout(
        commandBuffer,
        image.Handle,
        vk::ImageLayout::eUndefined,
        vk::ImageLayout::eTransferDstOptimal,
        imageSubresourceRange
    );

    commandBuffer.copyBufferToImage(
        stagingBuffer.Handle,
        image.Handle,
        vk::ImageLayout::eTransferDstOptimal,
        copyRegions
    );

    TransitionImageLayout(
        commandBuffer,
        image.Handle,
        vk::ImageLayout::eTransferDstOptimal,
        vk::ImageLayout::eShaderReadOnlyOptimal,
        imageSubresourceRange
    );

    return { image, stagingBuffer };
}

inline std::vector<std::uint8_t> ReadFile(const char *filepath)
{
    std::ifstream file(filepath, std::ios::binary | std::ios::ate);

    if (!file)
    {
        throw std::runtime_error("Could not open file: " + std::string(filepath));
    }

    const auto end = file.tellg();
    file.seekg(0, std::ios::beg);
    const auto start = file.tellg();

    std::vector<std::uint8_t> buffer;

    const auto size = static_cast<std::size_t>(end - start);

    if (size == 0)
    {
        return buffer;
    }

    buffer.resize(size);
    file.read(reinterpret_cast<char *>(buffer.data()), static_cast<std::streamsize>(size));

    return buffer;
}

template<std::size_t N>
constexpr auto BytesToUInt(std::span<const std::uint8_t, N> bytes)
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

#endif // !VULKAN_RENDERER_UTILITY_HPP
