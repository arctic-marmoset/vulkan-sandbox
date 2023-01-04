#ifndef VULKAN_RENDERER_DEVICE_HPP
#define VULKAN_RENDERER_DEVICE_HPP

#include <vulkan/vulkan.hpp>

#include <array>
#include <cassert>
#include <set>
#include <span>

class Device
{
public:
    struct QueueFamilyIndices
    {
        std::uint32_t Graphics = InvalidIndex;
        std::uint32_t Present  = InvalidIndex;

        static constexpr auto InvalidIndex = static_cast<std::uint32_t>(-1);

        [[nodiscard]]
        static QueueFamilyIndices Find(vk::PhysicalDevice physicalDevice, vk::SurfaceKHR surface);

        [[nodiscard]]
        constexpr bool IsComplete() const
        {
            return Graphics != InvalidIndex
                   && Present != InvalidIndex;
        }

        [[nodiscard]]
        constexpr std::array<std::uint32_t, 2> ToArray() const
        {
            assert("ToArray requires all indices to be valid" && IsComplete());

            return {
                Graphics,
                Present,
            };
        }

        [[nodiscard]]
        std::set<std::uint32_t> ToUnique() const;
    };

    void Init(
        vk::PhysicalDevice physicalDevice,
        QueueFamilyIndices queueFamilyIndices,
        std::span<const char *const> requiredExtensions
    );

    void Destroy();

    [[nodiscard]]
    vk::PhysicalDevice GetPhysicalDevice() const
    {
        return m_PhysicalDevice;
    }

    [[nodiscard]]
    const vk::PhysicalDeviceProperties &GetProperties() const
    {
        return m_Properties;
    }

    [[nodiscard]]
    const vk::PhysicalDeviceMemoryProperties &GetMemoryProperties() const
    {
        return m_MemoryProperties;
    }

    [[nodiscard]]
    const QueueFamilyIndices &GetQueueFamilyIndices() const
    {
        return m_QueueFamilyIndices;
    }

    [[nodiscard]]
    const std::set<std::uint32_t> &GetUniqueQueueFamilyIndices() const
    {
        return m_UniqueQueueFamilyIndices;
    }

    [[nodiscard]]
    vk::Device GetHandle() const
    {
        return m_Device;
    }

    [[nodiscard]]
    vk::Queue GetGraphicsQueue() const
    {
        return m_GraphicsQueue;
    }

    [[nodiscard]]
    vk::Queue GetPresentQueue() const
    {
        return m_PresentQueue;
    }

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
