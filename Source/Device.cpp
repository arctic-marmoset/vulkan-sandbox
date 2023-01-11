#include "Device.hpp"

#include <bitset>
#include <ranges>

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

void Device::Init(const InitInfo &info)
{
    m_PhysicalDevice = info.PhysicalDevice;
    m_Properties = m_PhysicalDevice.getProperties();
    m_MemoryProperties = m_PhysicalDevice.getMemoryProperties();
    m_QueueFamilyIndices = info.QueueFamilyIndices;
    m_UniqueQueueFamilyIndices = m_QueueFamilyIndices.ToUnique();

    m_Device = CreateDevice(m_PhysicalDevice, m_UniqueQueueFamilyIndices, info.RequiredExtensions);
    VULKAN_HPP_DEFAULT_DISPATCHER.init(m_Device);
    m_GraphicsQueue = m_Device.getQueue(m_QueueFamilyIndices.Graphics, 0);
    m_PresentQueue = m_Device.getQueue(m_QueueFamilyIndices.Present, 0);
}

void Device::Destroy()
{
    vkDestroyDevice(m_Device, nullptr);
}

std::uint32_t Device::FindMemoryType(std::uint32_t typeBits, vk::MemoryPropertyFlags properties) const
{
    constexpr auto typeBitCount = std::numeric_limits<decltype(typeBits)>::digits;
    const std::bitset<typeBitCount> types(typeBits);

    const auto isTypeAcceptable = [&types](std::uint32_t index)
    {
        return types.test(index);
    };

    const auto arePropertiesPresent = [&properties](const vk::MemoryType &memoryType)
    {
        return (memoryType.propertyFlags & properties) == properties;
    };

    for (std::uint32_t i = 0; i < m_MemoryProperties.memoryTypeCount; ++i)
    {
        const auto &candidateType = m_MemoryProperties.memoryTypes[i];
        if (isTypeAcceptable(i)
            && arePropertiesPresent(candidateType))
        {
            return i;
        }
    }

    throw std::runtime_error("Failed to find a suitable memory type");
}

Buffer Device::CreateBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage, vk::MemoryPropertyFlags properties) const
{
    const vk::BufferCreateInfo bufferInfo = {
        .size        = size,
        .usage       = usage,
        .sharingMode = vk::SharingMode::eExclusive,
    };

    const vk::Buffer buffer = m_Device.createBuffer(bufferInfo);

    const vk::MemoryRequirements memoryRequirements = m_Device.getBufferMemoryRequirements(buffer);
    const std::uint32_t memoryTypeIndex = FindMemoryType(memoryRequirements.memoryTypeBits, properties);

    const vk::MemoryAllocateInfo allocateInfo = {
        .allocationSize  = memoryRequirements.size,
        .memoryTypeIndex = memoryTypeIndex,
    };

    const vk::DeviceMemory memory = m_Device.allocateMemory(allocateInfo);
    m_Device.bindBufferMemory(buffer, memory, 0);

    return {
        .Handle = buffer,
        .Memory = memory,
    };
}

Image Device::CreateImage(
    std::uint32_t width,
    std::uint32_t height,
    vk::Format format,
    vk::ImageTiling tiling,
    vk::ImageUsageFlags usage,
    vk::MemoryPropertyFlags properties
) const
{
    const vk::ImageCreateInfo createInfo = {
        .imageType   = vk::ImageType::e2D,
        .format      = format,
        .extent      = {
            .width   = width,
            .height  = height,
            .depth   = 1,
        },
        .mipLevels   = 1,
        .arrayLayers = 1,
        .samples     = vk::SampleCountFlagBits::e1,
        .tiling      = tiling,
        .usage       = usage,
        .sharingMode = vk::SharingMode::eExclusive,
    };

    const vk::Image image = m_Device.createImage(createInfo);

    const vk::MemoryRequirements memoryRequirements = m_Device.getImageMemoryRequirements(image);

    const vk::MemoryAllocateInfo allocInfo = {
        .allocationSize = memoryRequirements.size,
        .memoryTypeIndex = FindMemoryType(memoryRequirements.memoryTypeBits, properties),
    };

    const vk::DeviceMemory memory = m_Device.allocateMemory(allocInfo);
    m_Device.bindImageMemory(image, memory, 0);

    return {
        .Handle = image,
        .Memory = memory,
    };
}

vk::ImageView Device::CreateImageView(vk::Image image, vk::Format format, vk::ImageAspectFlags aspectFlags) const
{
    const vk::ImageViewCreateInfo createInfo = {
        .image              = image,
        .viewType           = vk::ImageViewType::e2D,
        .format             = format,
        .subresourceRange   = {
            .aspectMask     = aspectFlags,
            .baseMipLevel   = 0,
            .levelCount     = 1,
            .baseArrayLayer = 0,
            .layerCount     = 1,
        },
    };

    return m_Device.createImageView(createInfo);
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
