#ifndef VULKAN_RENDERER_DEVICE_HPP
#define VULKAN_RENDERER_DEVICE_HPP

#include <vulkan/vulkan.hpp>

#include <array>
#include <cassert>
#include <cstdint>
#include <set>
#include <span>

struct Buffer
{
    vk::Buffer Handle;
    vk::DeviceMemory Memory;

    void Destroy(vk::Device device)
    {
        device.destroyBuffer(Handle);
        device.freeMemory(Memory);

        Handle = nullptr;
        Memory = nullptr;
    }
};

struct Image
{
    vk::Image Handle;
    vk::DeviceMemory Memory;

    void Destroy(vk::Device device)
    {
        device.destroyImage(Handle);
        device.freeMemory(Memory);

        Handle = nullptr;
        Memory = nullptr;
    }
};

class Device
{
public:
    struct QueueFamilyIndices
    {
        std::uint32_t Graphics = InvalidIndex;
        std::uint32_t Present  = InvalidIndex;

        static constexpr auto InvalidIndex = static_cast<std::uint32_t>(-1);

        static QueueFamilyIndices Find(vk::PhysicalDevice physicalDevice, vk::SurfaceKHR surface);

        constexpr bool IsComplete() const
        {
            return Graphics != InvalidIndex
                   && Present != InvalidIndex;
        }

        constexpr std::array<std::uint32_t, 2> ToArray() const
        {
            assert("ToArray requires all indices to be valid" && IsComplete());

            return {
                Graphics,
                Present,
            };
        }

        std::set<std::uint32_t> ToUnique() const;
    };

    struct InitInfo
    {
        vk::PhysicalDevice PhysicalDevice;
        QueueFamilyIndices QueueFamilyIndices;
        std::vector<const char *> RequiredExtensions;
    };

    void Init(const InitInfo &info);

    void Destroy();

    std::uint32_t FindMemoryType(std::uint32_t typeBits, vk::MemoryPropertyFlags properties) const;

    [[nodiscard]]
    Buffer CreateBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage, vk::MemoryPropertyFlags properties) const;

    [[nodiscard]]
    Image CreateImage(
        std::uint32_t width,
        std::uint32_t height,
        vk::Format format,
        vk::ImageTiling tiling,
        vk::ImageUsageFlags usage,
        vk::MemoryPropertyFlags properties
    ) const;

    [[nodiscard]]
    Image CreateCubeMap(
        std::uint32_t layerWidth,
        std::uint32_t layerHeight,
        vk::Format format,
        vk::ImageTiling tiling,
        vk::ImageUsageFlags usage,
        vk::MemoryPropertyFlags properties
    ) const;

    [[nodiscard]]
    vk::ImageView CreateImageView(vk::Image image, vk::Format format, vk::ImageAspectFlags aspectFlags) const;

    vk::PhysicalDevice GetPhysicalDevice() const
    {
        return m_PhysicalDevice;
    }

    const vk::PhysicalDeviceProperties &GetProperties() const
    {
        return m_Properties;
    }

    const vk::PhysicalDeviceMemoryProperties &GetMemoryProperties() const
    {
        return m_MemoryProperties;
    }

    const QueueFamilyIndices &GetQueueFamilyIndices() const
    {
        return m_QueueFamilyIndices;
    }

    const std::set<std::uint32_t> &GetUniqueQueueFamilyIndices() const
    {
        return m_UniqueQueueFamilyIndices;
    }

    vk::Device GetHandle() const
    {
        return m_Device;
    }

    vk::Queue GetGraphicsQueue() const
    {
        return m_GraphicsQueue;
    }

    vk::Queue GetPresentQueue() const
    {
        return m_PresentQueue;
    }

private:
    [[nodiscard]]
    Image CreateImage(
        std::uint32_t width,
        std::uint32_t height,
        std::uint32_t layerCount,
        vk::Format format,
        vk::ImageTiling tiling,
        vk::ImageUsageFlags usage,
        vk::ImageCreateFlags flags,
        vk::MemoryPropertyFlags properties
    ) const;

private:
    vk::PhysicalDevice m_PhysicalDevice;
    vk::PhysicalDeviceProperties m_Properties;
    vk::PhysicalDeviceMemoryProperties m_MemoryProperties;
    QueueFamilyIndices m_QueueFamilyIndices;
    std::set<std::uint32_t> m_UniqueQueueFamilyIndices;
    vk::Device m_Device;
    vk::Queue m_GraphicsQueue;
    vk::Queue m_PresentQueue;
};

#endif // !VULKAN_RENDERER_DEVICE_HPP
