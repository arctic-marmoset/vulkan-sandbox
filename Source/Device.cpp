#include "Device.hpp"

static vk::Device CreateDevice(
    vk::PhysicalDevice physicalDevice,
    const std::set<std::uint32_t> &indices,
    std::span<const char *const> extensions
);

std::set<std::uint32_t> Device::QueueFamilyIndices::ToUnique() const
{
    assert("ToUnique requires all indices to be valid" && IsComplete());

    return {
        Graphics,
        Present,
    };
}

Device::QueueFamilyIndices Device::QueueFamilyIndices::Find(vk::PhysicalDevice physicalDevice, vk::SurfaceKHR surface)
{
    Device::QueueFamilyIndices indices;

    const std::vector<vk::QueueFamilyProperties> queueFamiliesProperties = physicalDevice.getQueueFamilyProperties();

    for (std::uint32_t i = 0; i < queueFamiliesProperties.size(); ++i)
    {
        const auto &queueFamilyProperties = queueFamiliesProperties[i];

        if (queueFamilyProperties.queueFlags & vk::QueueFlagBits::eGraphics)
        {
            indices.Graphics = i;
        }

        const vk::Bool32 isPresentToSurfaceSupported = physicalDevice.getSurfaceSupportKHR(i, surface);

        if (isPresentToSurfaceSupported)
        {
            indices.Present = i;
        }

        if (indices.IsComplete())
        {
            break;
        }
    }

    return indices;
}

void Device::Init(
    vk::PhysicalDevice physicalDevice,
    Device::QueueFamilyIndices queueFamilyIndices,
    std::span<const char *const> requiredExtensions
)
{
    m_PhysicalDevice = physicalDevice;
    m_Properties = m_PhysicalDevice.getProperties();
    m_MemoryProperties = m_PhysicalDevice.getMemoryProperties();
    m_QueueFamilyIndices = queueFamilyIndices;
    m_UniqueQueueFamilyIndices = m_QueueFamilyIndices.ToUnique();

    m_Device = CreateDevice(physicalDevice, m_UniqueQueueFamilyIndices, requiredExtensions);
    VULKAN_HPP_DEFAULT_DISPATCHER.init(m_Device);
    m_GraphicsQueue = m_Device.getQueue(m_QueueFamilyIndices.Graphics, 0);
    m_PresentQueue = m_Device.getQueue(m_QueueFamilyIndices.Present, 0);
}

void Device::Destroy()
{
    vkDestroyDevice(m_Device, nullptr);
}

vk::Device CreateDevice(
    vk::PhysicalDevice physicalDevice,
    const std::set<std::uint32_t> &indices,
    std::span<const char *const> extensions
)
{
    constexpr std::array queuePriorities = { 1.0F };

    std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos;
    std::ranges::transform(
        indices,
        std::back_inserter(queueCreateInfos),
        [&queuePriorities](std::uint32_t index)
        {
            return vk::DeviceQueueCreateInfo()
                .setQueueCount(queuePriorities.size())
                .setQueuePriorities(queuePriorities)
                .setQueueFamilyIndex(index);
        }
    );

    const vk::PhysicalDeviceFeatures features = {
        .samplerAnisotropy = VK_TRUE,
    };

    const auto info = vk::DeviceCreateInfo()
        .setQueueCreateInfos(queueCreateInfos)
        .setPEnabledExtensionNames(extensions)
        .setPEnabledFeatures(&features);

    return physicalDevice.createDevice(info);
}
