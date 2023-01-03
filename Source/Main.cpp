#include "Utility.hpp"

#include <GLFW/glfw3.h>
#include <glm/gtc/matrix_transform.hpp>
#include <vulkan/vulkan.hpp>

#include <algorithm>
#include <array>
#include <bitset>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <functional>
#include <iostream>
#include <limits>
#include <optional>
#include <set>
#include <stdexcept>

VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

constexpr std::uint32_t WindowWidth = 1280;
constexpr std::uint32_t WindowHeight = 720;

constexpr std::array<const char *, 1> ValidationLayers = {
    "VK_LAYER_KHRONOS_validation",
};

constexpr std::array<const char *, 1> DeviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME,
};

constexpr bool IsDebugMode = true;

constexpr std::array Vertices = {
    // Ground
    Vertex{ .Position = { -3.0F,  1.8F, -3.0F }, .TexCoord = { 0.0F, (1.0F - 1.0F) } },
    Vertex{ .Position = {  3.0F,  1.8F, -3.0F }, .TexCoord = { 1.0F, (1.0F - 1.0F) } },
    Vertex{ .Position = {  3.0F,  1.8F,  3.0F }, .TexCoord = { 1.0F, (1.0F - 0.0F) } },
    Vertex{ .Position = { -3.0F,  1.8F,  3.0F }, .TexCoord = { 0.0F, (1.0F - 0.0F) } },

    // Front Wall
    Vertex{ .Position = { -3.0F,  1.8F,  3.0F }, .TexCoord = { 0.0F, (1.0F - 1.0F) } },
    Vertex{ .Position = {  3.0F,  1.8F,  3.0F }, .TexCoord = { 1.0F, (1.0F - 1.0F) } },
    Vertex{ .Position = {  3.0F, -0.7F,  3.0F }, .TexCoord = { 1.0F, (1.0F - 0.0F) } },
    Vertex{ .Position = { -3.0F, -0.7F,  3.0F }, .TexCoord = { 0.0F, (1.0F - 0.0F) } },

    // Right Wall
    Vertex{ .Position = {  3.0F,  1.8F,  3.0F }, .TexCoord = { 0.0F, (1.0F - 1.0F) } },
    Vertex{ .Position = {  3.0F,  1.8F, -3.0F }, .TexCoord = { 1.0F, (1.0F - 1.0F) } },
    Vertex{ .Position = {  3.0F, -0.7F, -3.0F }, .TexCoord = { 1.0F, (1.0F - 0.0F) } },
    Vertex{ .Position = {  3.0F, -0.7F,  3.0F }, .TexCoord = { 0.0F, (1.0F - 0.0F) } },

    // Back Wall
    Vertex{ .Position = {  3.0F,  1.8F, -3.0F }, .TexCoord = { 0.0F, (1.0F - 1.0F) } },
    Vertex{ .Position = { -3.0F,  1.8F, -3.0F }, .TexCoord = { 1.0F, (1.0F - 1.0F) } },
    Vertex{ .Position = { -3.0F, -0.7F, -3.0F }, .TexCoord = { 1.0F, (1.0F - 0.0F) } },
    Vertex{ .Position = {  3.0F, -0.7F, -3.0F }, .TexCoord = { 0.0F, (1.0F - 0.0F) } },

    // Left Wall
    Vertex{ .Position = { -3.0F,  1.8F, -3.0F }, .TexCoord = { 0.0F, (1.0F - 1.0F) } },
    Vertex{ .Position = { -3.0F,  1.8F,  3.0F }, .TexCoord = { 1.0F, (1.0F - 1.0F) } },
    Vertex{ .Position = { -3.0F, -0.7F,  3.0F }, .TexCoord = { 1.0F, (1.0F - 0.0F) } },
    Vertex{ .Position = { -3.0F, -0.7F, -3.0F }, .TexCoord = { 0.0F, (1.0F - 0.0F) } },
};

constexpr std::array Indices =
    std::to_array<std::uint16_t>(
        {
            // Ground
             0,  1,  2,  2,  3,  0,
            // Front Wall
             4,  5,  6,  6,  7,  4,
            // Right Wall
             8,  9, 10, 10, 11,  8,
            // Back Wall
            12, 13, 14, 14, 15, 12,
            // Left Wall
            16, 17, 18, 18, 19, 16,
        }
    );

VKAPI_ATTR vk::Bool32 VKAPI_CALL VulkanDebugMessengerCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT severity,
    VkDebugUtilsMessageTypeFlagsEXT type,
    const VkDebugUtilsMessengerCallbackDataEXT *data,
    void *userData
)
{
    (void)type;
    (void)userData;

    switch (severity)
    {
    case VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT:
        [[fallthrough]];
    case VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT:
        std::cout << "[VULKAN] " << data->pMessage << '\n';
        break;

    default:
        std::cerr << "[VULKAN] Unknown debug message severity\n";
        [[fallthrough]];
    case VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT:
        [[fallthrough]];
    case VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT:
        std::cerr << "[VULKAN] " << data->pMessage << '\n';
        break;
    }

    return VK_FALSE;
}

bool IsValidationLayersSupported()
{
    auto layers = vk::enumerateInstanceLayerProperties();
    std::ranges::sort(layers, { }, &vk::LayerProperties::layerName);

    return std::ranges::includes(
        layers,
        ValidationLayers,
        { },
        [](const vk::LayerProperties &properties)
        {
            return static_cast<std::string_view>(properties.layerName);
        }
    );
}

std::vector<const char *> GetRequiredExtensions()
{
    std::uint32_t glfwExtensionCount = 0;
    const char **glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

    std::vector extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

    if constexpr (IsDebugMode)
    {
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    return extensions;
}

constexpr vk::DebugUtilsMessengerCreateInfoEXT ConsoleDebugMessengerInfo = {
    .messageSeverity = vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose
                     | vk::DebugUtilsMessageSeverityFlagBitsEXT::eInfo
                     | vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning
                     | vk::DebugUtilsMessageSeverityFlagBitsEXT::eError,

    .messageType     = vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral
                     | vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation
                     | vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance,

    .pfnUserCallback = VulkanDebugMessengerCallback,
};

class Application
{
public:
    static constexpr std::array DynamicStates = {
        vk::DynamicState::eScissor,
        vk::DynamicState::eViewport,
    };

    void SetResourcesPath(const char *path)
    {
        m_ResourcesPath = path;
    }

    void Run()
    {
        CreateWindow();
        InitVulkan();
        EnterMainLoop();
        Cleanup();
    }

private:
    std::string GetResourcePath(const char *relativePath)
    {
        return (m_ResourcesPath / relativePath).string();
    }

    void CreateWindow()
    {
        if (!glfwInit())
        {
            throw std::runtime_error("Failed to initialize GLFW");
        }

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

        m_Window = glfwCreateWindow(WindowWidth, WindowHeight, "Vulkan Renderer", nullptr, nullptr);

        if (!m_Window)
        {
            throw std::runtime_error("Failed to create window");
        }

        glfwSetWindowUserPointer(m_Window, this);
        glfwSetFramebufferSizeCallback(m_Window, FramebufferSizeCallback);
        glfwSetKeyCallback(m_Window, KeyCallback);
    }

    void InitVulkan()
    {
        CreateInstance();
        AttachDebugMessenger();
        CreateSurface();
        SelectPhysicalDevice();
        CreateLogicalDevice();
        CreateSwapChain();
        CreateImageViews();
        CreateRenderPass();
        CreateDescriptorSetLayout();
        CreateGraphicsPipeline();
        CreateCommandPool();
        CreateDepthResources();
        CreateFramebuffers();
        CreateTextureImage();
        CreateTextureImageView();
        CreateTextureSampler();
        CreateVertexBuffer();
        CreateIndexBuffer();
        CreateUniformBuffers();
        CreateDescriptorPool();
        CreateDescriptorSets();
        CreateCommandBuffers();
        CreateSyncObjects();
    }

    void CreateInstance()
    {
        auto *vkGetInstanceProcAddr = m_Loader.getProcAddress<PFN_vkGetInstanceProcAddr>("vkGetInstanceProcAddr");
        VULKAN_HPP_DEFAULT_DISPATCHER.init(vkGetInstanceProcAddr);

        if constexpr (IsDebugMode)
        {
            if (!IsValidationLayersSupported())
            {
                throw std::runtime_error("Validation layers requested but not supported");
            }
        }

        const vk::ApplicationInfo appInfo = {
            .apiVersion = VK_API_VERSION_1_3,
        };

        vk::StructureChain<vk::InstanceCreateInfo, vk::DebugUtilsMessengerCreateInfoEXT> chain;
        auto &createInfo = chain.get<vk::InstanceCreateInfo>();
        auto &debugCreateInfo = chain.get<vk::DebugUtilsMessengerCreateInfoEXT>();

        const auto extensions = GetRequiredExtensions();

        createInfo = {
            .pApplicationInfo        = &appInfo,
            .enabledExtensionCount   = static_cast<std::uint32_t>(extensions.size()),
            .ppEnabledExtensionNames = extensions.data(),
        };

        if constexpr (IsDebugMode)
        {
            debugCreateInfo = ConsoleDebugMessengerInfo;
            createInfo.enabledLayerCount = static_cast<std::uint32_t>(ValidationLayers.size());
            createInfo.ppEnabledLayerNames = ValidationLayers.data();
        }
        else
        {
            chain.unlink<vk::DebugUtilsMessengerCreateInfoEXT>();
        }

        m_Instance = vk::createInstance(createInfo);
        VULKAN_HPP_DEFAULT_DISPATCHER.init(m_Instance);
    }

    void AttachDebugMessenger()
    {
        if constexpr (!IsDebugMode)
        {
            return;
        }

        m_DebugMessenger = m_Instance.createDebugUtilsMessengerEXT(ConsoleDebugMessengerInfo);
    }

    void CreateSurface()
    {
        VkSurfaceKHR surface = VK_NULL_HANDLE;

        if (glfwCreateWindowSurface(m_Instance, m_Window, nullptr, &surface) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create a window surface");
        }

        m_Surface = surface;
    }

    void SelectPhysicalDevice()
    {
        const auto devices = m_Instance.enumeratePhysicalDevices();

        if (devices.empty())
        {
            throw std::runtime_error("Failed to find a GPU with Vulkan support");
        }

        const auto candidate = std::ranges::find_if(devices, [this](const vk::PhysicalDevice &device)
        {
            return IsDeviceSuitable(device);
        });

        if (candidate == devices.end())
        {
            throw std::runtime_error("Failed to find a GPU suitable for this Application");
        }

        m_PhysicalDevice = *candidate;
    }

    bool IsDeviceSuitable(const vk::PhysicalDevice &device)
    {
        const auto indices = FindQueueFamilies(device);

        const bool isRequiredExtensionsSupported = IsRequiredExtensionsSupported(device);

        const bool isSwapChainAdequate =
            isRequiredExtensionsSupported
            && ([&]
            {
                const auto details = QuerySwapChainSupport(device);
                return !details.Formats.empty() && !details.PresentModes.empty();
            }());

        const vk::PhysicalDeviceFeatures supportedFeatures = device.getFeatures();

        return indices.IsComplete()
               && isSwapChainAdequate
               && supportedFeatures.samplerAnisotropy;
    }

    [[nodiscard]]
    QueueFamilyIndices FindQueueFamilies(const vk::PhysicalDevice &device) const
    {
        QueueFamilyIndices indices;

        const std::vector<vk::QueueFamilyProperties> queueFamiliesProperties = device.getQueueFamilyProperties();

        for (std::uint32_t i = 0; i < queueFamiliesProperties.size(); ++i)
        {
            const auto &queueFamilyProperties = queueFamiliesProperties[i];

            if (queueFamilyProperties.queueFlags & vk::QueueFlagBits::eGraphics)
            {
                indices.Graphics = i;
            }

            const vk::Bool32 isPresentToSurfaceSupported = device.getSurfaceSupportKHR(i, m_Surface);

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

    void CreateLogicalDevice()
    {
        const QueueFamilyIndices indices = FindQueueFamilies(m_PhysicalDevice);

        constexpr float queuePriority = 1.0F;

        const std::set uniqueQueueFamilies = {
            indices.Graphics.value(),
            indices.Present.value(),
        };

        std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos;
        queueCreateInfos.reserve(uniqueQueueFamilies.size());

        for (auto queueFamily : uniqueQueueFamilies)
        {
            queueCreateInfos.push_back({
                .queueFamilyIndex = queueFamily,
                .queueCount       = 1,
                .pQueuePriorities = &queuePriority,
            });
        }

        const vk::PhysicalDeviceFeatures deviceFeatures = {
            .samplerAnisotropy = VK_TRUE,
        };

        const vk::DeviceCreateInfo createInfo = {
            .queueCreateInfoCount    = static_cast<std::uint32_t>(queueCreateInfos.size()),
            .pQueueCreateInfos       = queueCreateInfos.data(),
            .enabledExtensionCount   = static_cast<std::uint32_t>(DeviceExtensions.size()),
            .ppEnabledExtensionNames = DeviceExtensions.data(),
            .pEnabledFeatures        = &deviceFeatures,
        };

        m_Device = m_PhysicalDevice.createDevice(createInfo);
        VULKAN_HPP_DEFAULT_DISPATCHER.init(m_Device);

        m_GraphicsQueue = m_Device.getQueue(indices.Graphics.value(), 0);
        m_PresentQueue = m_Device.getQueue(indices.Present.value(), 0);
    }

    [[nodiscard]]
    SwapChainSupportDetails QuerySwapChainSupport(const vk::PhysicalDevice &device) const
    {
        return {
            .Capabilities = device.getSurfaceCapabilitiesKHR(m_Surface),
            .Formats      = device.getSurfaceFormatsKHR(m_Surface),
            .PresentModes = device.getSurfacePresentModesKHR(m_Surface),
        };
    }

    void CreateSwapChain()
    {
        const auto swapChainSupport = QuerySwapChainSupport(m_PhysicalDevice);

        const auto surfaceFormat = SelectSwapSurfaceFormat(swapChainSupport.Formats);
        const auto presentMode = SelectSwapPresentMode(swapChainSupport.PresentModes);
        const auto extent = SelectSwapExtent(swapChainSupport.Capabilities);

        const std::uint32_t desiredImageCount = swapChainSupport.Capabilities.minImageCount + 1;
        const std::uint32_t maxImageCount = swapChainSupport.Capabilities.maxImageCount;

        const auto imageCount = maxImageCount == 0
            ? desiredImageCount
            : std::min(desiredImageCount, maxImageCount);

        const auto indices = FindQueueFamilies(m_PhysicalDevice);

        const std::array queueFamilyIndices = {
            indices.Graphics.value(),
            indices.Present.value(),
        };

        m_OldSwapChain = m_SwapChain;

        const auto createInfo = ([&]
        {
            vk::SwapchainCreateInfoKHR result = {
                .surface          = m_Surface,
                .minImageCount    = imageCount,
                .imageFormat      = surfaceFormat.format,
                .imageColorSpace  = surfaceFormat.colorSpace,
                .imageExtent      = extent,
                .imageArrayLayers = 1,
                .imageUsage       = vk::ImageUsageFlagBits::eColorAttachment,
                .preTransform     = swapChainSupport.Capabilities.currentTransform,
                .presentMode      = presentMode,
                .clipped          = VK_TRUE,
                .oldSwapchain     = m_OldSwapChain,
            };

            if (indices.Graphics != indices.Present)
            {
                result.imageSharingMode = vk::SharingMode::eConcurrent;
                result.queueFamilyIndexCount = 2;
                result.pQueueFamilyIndices = queueFamilyIndices.data();
            }
            else
            {
                result.imageSharingMode = vk::SharingMode::eExclusive;
            }

            return result;
        }());

        m_SwapChainImageFormat = surfaceFormat.format;
        m_SwapChainExtent = extent;

        m_SwapChain = m_Device.createSwapchainKHR(createInfo);
        m_SwapChainImages = m_Device.getSwapchainImagesKHR(m_SwapChain);
    }

    [[nodiscard]]
    vk::Extent2D SelectSwapExtent(const vk::SurfaceCapabilitiesKHR &capabilities) const
    {
        if (capabilities.currentExtent.width != std::numeric_limits<std::uint32_t>::max())
        {
            return capabilities.currentExtent;
        }

        int frameWidth = 0;
        int frameHeight = 0;
        glfwGetFramebufferSize(m_Window, &frameWidth, &frameHeight);

        const auto width = static_cast<std::uint32_t>(frameWidth);
        const auto height = static_cast<std::uint32_t>(frameHeight);

        const auto [minWidth, minHeight] = capabilities.minImageExtent;
        const auto [maxWidth, maxHeight] = capabilities.maxImageExtent;

        return {
            .width  = std::clamp(width, minWidth, maxWidth),
            .height = std::clamp(height, minHeight, maxHeight),
        };
    }

    [[nodiscard]]
    vk::ImageView CreateImageView(vk::Image image, vk::Format format, vk::ImageAspectFlags aspectFlags) const
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

    void CreateImageViews()
    {
        m_SwapChainImageViews.reserve(m_SwapChainImages.size());

        for (const auto &image : m_SwapChainImages)
        {
            const auto imageView = CreateImageView(image, m_SwapChainImageFormat, vk::ImageAspectFlagBits::eColor);
            m_SwapChainImageViews.push_back(imageView);
        }
    }

    void CreateRenderPass()
    {
        enum : std::uint32_t
        {
            ColorAttachmentIndex,
            DepthAttachmentIndex,

            AttachmentCount,
        };

        std::array<vk::AttachmentDescription, AttachmentCount> attachments;
        vk::AttachmentDescription &colorAttachment = attachments[ColorAttachmentIndex];
        vk::AttachmentDescription &depthAttachment = attachments[DepthAttachmentIndex];

        colorAttachment = {
            .format         = m_SwapChainImageFormat,
            .samples        = vk::SampleCountFlagBits::e1,
            .loadOp         = vk::AttachmentLoadOp::eClear,
            .storeOp        = vk::AttachmentStoreOp::eStore,
            .stencilLoadOp  = vk::AttachmentLoadOp::eDontCare,
            .stencilStoreOp = vk::AttachmentStoreOp::eDontCare,
            .initialLayout  = vk::ImageLayout::eUndefined,
            .finalLayout    = vk::ImageLayout::ePresentSrcKHR,
        };

        depthAttachment = {
            .format         = FindDepthFormat(),
            .samples        = vk::SampleCountFlagBits::e1,
            .loadOp         = vk::AttachmentLoadOp::eClear,
            .storeOp        = vk::AttachmentStoreOp::eDontCare,
            .stencilLoadOp  = vk::AttachmentLoadOp::eDontCare,
            .stencilStoreOp = vk::AttachmentStoreOp::eDontCare,
            .initialLayout  = vk::ImageLayout::eUndefined,
            .finalLayout    = vk::ImageLayout::eDepthStencilAttachmentOptimal,
        };

        const std::array colorAttachmentReferences = {
            vk::AttachmentReference{
                .attachment = ColorAttachmentIndex,
                .layout     = vk::ImageLayout::eColorAttachmentOptimal,
            },
        };

        const vk::AttachmentReference depthAttachmentReference = {
            .attachment = DepthAttachmentIndex,
            .layout     = vk::ImageLayout::eDepthStencilAttachmentOptimal,
        };

        enum : std::uint32_t
        {
            ColorSubpassIndex,

            SubpassCount,
        };

        std::array<vk::SubpassDescription, SubpassCount> subpassDescriptions;
        vk::SubpassDescription &colorSubpass = subpassDescriptions[ColorSubpassIndex];
        colorSubpass = {
            .pipelineBindPoint       = vk::PipelineBindPoint::eGraphics,
            .colorAttachmentCount    = static_cast<std::uint32_t>(colorAttachmentReferences.size()),
            .pColorAttachments       = colorAttachmentReferences.data(),
            .pDepthStencilAttachment = &depthAttachmentReference,
        };

        enum : std::uint32_t
        {
            ColorTransitionDependencyIndex,

            SubpassDependencyCount,
        };

        std::array<vk::SubpassDependency, SubpassDependencyCount> dependencies;
        vk::SubpassDependency &colorTransitionDependency = dependencies[ColorTransitionDependencyIndex];
        colorTransitionDependency = {
            .srcSubpass    = VK_SUBPASS_EXTERNAL,
            .dstSubpass    = ColorSubpassIndex,
            .srcStageMask  = vk::PipelineStageFlagBits::eColorAttachmentOutput
                           | vk::PipelineStageFlagBits::eEarlyFragmentTests,

            .dstStageMask  = vk::PipelineStageFlagBits::eColorAttachmentOutput
                           | vk::PipelineStageFlagBits::eEarlyFragmentTests,

            .srcAccessMask = vk::AccessFlagBits::eNoneKHR,
            .dstAccessMask = vk::AccessFlagBits::eColorAttachmentWrite
                           | vk::AccessFlagBits::eDepthStencilAttachmentWrite,
        };

        const vk::RenderPassCreateInfo renderPassCreateInfo = {
            .attachmentCount = static_cast<std::uint32_t>(attachments.size()),
            .pAttachments    = attachments.data(),
            .subpassCount    = static_cast<std::uint32_t>(subpassDescriptions.size()),
            .pSubpasses      = subpassDescriptions.data(),
            .dependencyCount = static_cast<std::uint32_t>(dependencies.size()),
            .pDependencies   = dependencies.data(),
        };

        m_RenderPass = m_Device.createRenderPass(renderPassCreateInfo);
    }

    void CreateDescriptorSetLayout()
    {
        enum : std::uint32_t
        {
            UboLayoutBindingIndex,
            SamplerLayoutBindingIndex,

            LayoutBindingCount,
        };

        std::array<vk::DescriptorSetLayoutBinding, LayoutBindingCount> layoutBindings;
        vk::DescriptorSetLayoutBinding &uboLayoutBinding = layoutBindings[UboLayoutBindingIndex];
        vk::DescriptorSetLayoutBinding &samplerLayoutBinding = layoutBindings[SamplerLayoutBindingIndex];

        uboLayoutBinding = {
            .binding         = UboLayoutBindingIndex,
            .descriptorType  = vk::DescriptorType::eUniformBuffer,
            .descriptorCount = 1,
            .stageFlags      = vk::ShaderStageFlagBits::eVertex,
        };

        samplerLayoutBinding = {
            .binding         = SamplerLayoutBindingIndex,
            .descriptorType  = vk::DescriptorType::eCombinedImageSampler,
            .descriptorCount = 1,
            .stageFlags      = vk::ShaderStageFlagBits::eFragment,
        };

        const vk::DescriptorSetLayoutCreateInfo layoutInfo = {
            .bindingCount = static_cast<std::uint32_t>(layoutBindings.size()),
            .pBindings    = layoutBindings.data(),
        };

        m_DescriptorSetLayout = m_Device.createDescriptorSetLayout(layoutInfo);
    }

    void CreateGraphicsPipeline()
    {
        const auto vsBytecode = ReadFile(GetResourcePath(TEXTURE_VERT_SHADER_RELATIVE_PATH).c_str());
        const auto fsBytecode = ReadFile(GetResourcePath(TEXTURE_FRAG_SHADER_RELATIVE_PATH).c_str());

        const auto vsModule = CreateShaderModule(vsBytecode);
        const auto fsModule = CreateShaderModule(fsBytecode);

        const std::array shaderStages = {
            vk::PipelineShaderStageCreateInfo{
                .stage  = vk::ShaderStageFlagBits::eVertex,
                .module = vsModule,
                .pName  = "main",
            },
            vk::PipelineShaderStageCreateInfo{
                .stage  = vk::ShaderStageFlagBits::eFragment,
                .module = fsModule,
                .pName  = "main",
            },
        };

        const auto bindingDescription = Vertex::GetBindingDescription();
        const auto attributeDescriptions = Vertex::GetAttributeDescriptions();

        const vk::PipelineVertexInputStateCreateInfo vertexInputStateCreateInfo = {
            .vertexBindingDescriptionCount   = 1,
            .pVertexBindingDescriptions      = &bindingDescription,
            .vertexAttributeDescriptionCount = static_cast<std::uint32_t>(attributeDescriptions.size()),
            .pVertexAttributeDescriptions    = attributeDescriptions.data(),
        };

        const vk::PipelineInputAssemblyStateCreateInfo inputAssemblyStateCreateInfo = {
            .topology               = vk::PrimitiveTopology::eTriangleList,
            .primitiveRestartEnable = VK_FALSE,
        };

        const vk::PipelineViewportStateCreateInfo viewportStateCreateInfo = {
            .viewportCount = 1,
            .scissorCount  = 1,
        };

        const vk::PipelineRasterizationStateCreateInfo rasterizationStateCreateInfo = {
            .depthClampEnable        = VK_FALSE,
            .rasterizerDiscardEnable = VK_FALSE,
            .polygonMode             = vk::PolygonMode::eFill,
            .cullMode                = vk::CullModeFlagBits::eBack,
            .frontFace               = vk::FrontFace::eCounterClockwise,
            .depthBiasEnable         = VK_FALSE,
            .lineWidth               = 1.0F,
        };

        const vk::PipelineMultisampleStateCreateInfo multisampleStateCreateInfo = {
            .rasterizationSamples = vk::SampleCountFlagBits::e1,
            .sampleShadingEnable  = VK_FALSE,
        };

        const vk::PipelineDepthStencilStateCreateInfo depthStencilStateCreateInfo = {
            .depthTestEnable  = VK_TRUE,
            .depthWriteEnable = VK_TRUE,
            .depthCompareOp   = vk::CompareOp::eGreater,
            .minDepthBounds   = 0.0F,
            .maxDepthBounds   = 1.0F,
        };

        const vk::PipelineColorBlendAttachmentState colorBlendAttachmentState = {
            .blendEnable    = VK_FALSE,
            .colorWriteMask = vk::ColorComponentFlagBits::eR
                              | vk::ColorComponentFlagBits::eG
                              | vk::ColorComponentFlagBits::eB
                              | vk::ColorComponentFlagBits::eA,
        };

        const vk::PipelineColorBlendStateCreateInfo colorBlendStateCreateInfo = {
            .logicOpEnable   = VK_FALSE,
            .attachmentCount = 1,
            .pAttachments    = &colorBlendAttachmentState,
        };

        const vk::PipelineLayoutCreateInfo pipelineLayoutCreateInfo = {
            .setLayoutCount = 1,
            .pSetLayouts    = &m_DescriptorSetLayout,
        };

        m_PipelineLayout = m_Device.createPipelineLayout(pipelineLayoutCreateInfo);

        const vk::PipelineDynamicStateCreateInfo dynamicStateCreateInfo = {
            .dynamicStateCount = static_cast<std::uint32_t>(DynamicStates.size()),
            .pDynamicStates    = DynamicStates.data(),
        };

        const vk::GraphicsPipelineCreateInfo pipelineCreateInfo = {
            .stageCount          = static_cast<std::uint32_t>(shaderStages.size()),
            .pStages             = shaderStages.data(),
            .pVertexInputState   = &vertexInputStateCreateInfo,
            .pInputAssemblyState = &inputAssemblyStateCreateInfo,
            .pViewportState      = &viewportStateCreateInfo,
            .pRasterizationState = &rasterizationStateCreateInfo,
            .pMultisampleState   = &multisampleStateCreateInfo,
            .pDepthStencilState  = &depthStencilStateCreateInfo,
            .pColorBlendState    = &colorBlendStateCreateInfo,
            .pDynamicState       = &dynamicStateCreateInfo,
            .layout              = m_PipelineLayout,
            .renderPass          = m_RenderPass,
            .subpass             = 0,
        };

        const auto [result, pipeline] = m_Device.createGraphicsPipeline({ }, pipelineCreateInfo);

        if (result != vk::Result::eSuccess)
        {
            throw std::runtime_error("Failed to create graphics pipeline");
        }

        m_GraphicsPipeline = pipeline;

        m_Device.destroy(vsModule);
        m_Device.destroy(fsModule);
    }

    void CreateFramebuffers()
    {
        m_SwapChainFramebuffers.reserve(m_SwapChainImageViews.size());

        for (const auto &imageView : m_SwapChainImageViews)
        {
            const std::array attachments = {
                imageView,
                m_DepthImageView,
            };

            const vk::FramebufferCreateInfo createInfo = {
                .renderPass      = m_RenderPass,
                .attachmentCount = static_cast<std::uint32_t>(attachments.size()),
                .pAttachments    = attachments.data(),
                .width           = m_SwapChainExtent.width,
                .height          = m_SwapChainExtent.height,
                .layers          = 1,
            };

            const auto framebuffer = m_Device.createFramebuffer(createInfo);
            m_SwapChainFramebuffers.push_back(framebuffer);
        }
    }

    void CreateCommandPool()
    {
        const QueueFamilyIndices indices = FindQueueFamilies(m_PhysicalDevice);

        const vk::CommandPoolCreateInfo createInfo = {
            .flags            = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
            .queueFamilyIndex = indices.Graphics.value(),
        };

        m_CommandPool = m_Device.createCommandPool(createInfo);
    }

    void CreateBuffer(
        vk::DeviceSize size,
        vk::BufferUsageFlags usage,
        vk::MemoryPropertyFlags properties,
        vk::Buffer &buffer,
        vk::DeviceMemory &bufferMemory
    )
    {
        const vk::BufferCreateInfo bufferInfo = {
            .size        = size,
            .usage       = usage,
            .sharingMode = vk::SharingMode::eExclusive,
        };

        buffer = m_Device.createBuffer(bufferInfo);

        const vk::MemoryRequirements memoryRequirements = m_Device.getBufferMemoryRequirements(buffer);

        const std::uint32_t memoryTypeIndex = FindMemoryType(memoryRequirements.memoryTypeBits, properties);

        const vk::MemoryAllocateInfo allocateInfo = {
            .allocationSize  = memoryRequirements.size,
            .memoryTypeIndex = memoryTypeIndex,
        };

        bufferMemory = m_Device.allocateMemory(allocateInfo);
        m_Device.bindBufferMemory(buffer, bufferMemory, 0);
    }

    void CopyBuffer(vk::Buffer source, vk::Buffer destination, vk::DeviceSize size)
    {
        const vk::CommandBuffer commandBuffer = BeginOneTimeCommands();

        const vk::BufferCopy copyRegion = {
            .srcOffset = 0,
            .dstOffset = 0,
            .size      = size,
        };

        commandBuffer.copyBuffer(source, destination, copyRegion);

        EndOneTimeCommands(commandBuffer);
    }

    void CopyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height)
    {
        const vk::CommandBuffer commandBuffer = BeginOneTimeCommands();

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

        EndOneTimeCommands(commandBuffer);
    }

    void CreateImage(
        std::uint32_t width,
        std::uint32_t height,
        vk::Format format,
        vk::ImageTiling tiling,
        vk::ImageUsageFlags usage,
        vk::MemoryPropertyFlags properties,
        vk::Image &image,
        vk::DeviceMemory &imageMemory
    )
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

        image = m_Device.createImage(createInfo);

        const vk::MemoryRequirements memoryRequirements = m_Device.getImageMemoryRequirements(image);

        const vk::MemoryAllocateInfo allocInfo = {
            .allocationSize = memoryRequirements.size,
            .memoryTypeIndex = FindMemoryType(memoryRequirements.memoryTypeBits, properties),
        };

        imageMemory = m_Device.allocateMemory(allocInfo);
        m_Device.bindImageMemory(image, imageMemory, 0);
    }

    template<typename InputIterator>
    vk::Format FindSupportedFormat(
        InputIterator begin,
        InputIterator end,
        vk::ImageTiling tiling,
        vk::FormatFeatureFlags features
    )
    {
        static_assert(
            std::is_same<typename std::iterator_traits<InputIterator>::value_type, vk::Format>::value,
            "InputIterator must be an iterator over vk::Format"
        );

        for (auto it = begin; it != end; ++it)
        {
            const vk::FormatProperties properties = m_PhysicalDevice.getFormatProperties(*it);
            if (tiling == vk::ImageTiling::eLinear
                && (properties.linearTilingFeatures & features) == features)
            {
                return *it;
            }
            else if (tiling == vk::ImageTiling::eOptimal
                     && (properties.optimalTilingFeatures & features) == features)
            {
                return *it;
            }
        }

        throw std::runtime_error("Failed to find a supported format");
    }

    vk::Format FindDepthFormat()
    {
        constexpr std::array formats = {
            vk::Format::eD32SfloatS8Uint,
            vk::Format::eD32Sfloat,
            vk::Format::eD24UnormS8Uint,
        };

        return FindSupportedFormat(
            formats.begin(),
            formats.end(),
            vk::ImageTiling::eOptimal,
            vk::FormatFeatureFlagBits::eDepthStencilAttachment
        );
    }

    static bool HasStencilComponent(vk::Format format)
    {
        return format == vk::Format::eD32SfloatS8Uint || format == vk::Format::eD24UnormS8Uint;
    }

    void CreateDepthResources()
    {
        const vk::Format depthFormat = FindDepthFormat();

        CreateImage(
            m_SwapChainExtent.width,
            m_SwapChainExtent.height,
            depthFormat,
            vk::ImageTiling::eOptimal,
            vk::ImageUsageFlagBits::eDepthStencilAttachment,
            vk::MemoryPropertyFlagBits::eDeviceLocal,
            m_DepthImage,
            m_DepthImageMemory
        );

        m_DepthImageView = CreateImageView(m_DepthImage, depthFormat, vk::ImageAspectFlagBits::eDepth);
    }

    void CreateTextureImage()
    {
        const std::vector<std::uint8_t> fileBytes = ReadFile(GetResourcePath("Textures/Missing_Raw.tga").c_str());
        const Tga::File texture = Tga::File::CreateFrom(fileBytes);
        const vk::DeviceSize textureSize = texture.GetSize();

        vk::Buffer stagingBuffer;
        vk::DeviceMemory stagingBufferMemory;
        CreateBuffer(
            textureSize,
            vk::BufferUsageFlagBits::eTransferSrc,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
            stagingBuffer,
            stagingBufferMemory
        );

        if (void *const data = m_Device.mapMemory(stagingBufferMemory, 0, textureSize))
        {
            std::memcpy(data, texture.Pixels.data(), texture.Pixels.size());
        }
        else
        {
            throw std::runtime_error("Failed to map texture memory");
        }

        CreateImage(
            texture.Width,
            texture.Height,
            vk::Format::eB8G8R8A8Srgb,
            vk::ImageTiling::eOptimal,
            vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled,
            vk::MemoryPropertyFlagBits::eDeviceLocal,
            m_TextureImage,
            m_TextureImageMemory
        );

        TransitionImageLayout(
            m_TextureImage,
            vk::ImageLayout::eUndefined,
            vk::ImageLayout::eTransferDstOptimal
        );

        CopyBufferToImage(stagingBuffer, m_TextureImage, texture.Width, texture.Height);

        TransitionImageLayout(
            m_TextureImage,
            vk::ImageLayout::eTransferDstOptimal,
            vk::ImageLayout::eShaderReadOnlyOptimal
        );

        m_Device.destroy(stagingBuffer);
        m_Device.free(stagingBufferMemory);
    }

    void CreateTextureImageView()
    {
        m_TextureImageView = CreateImageView(
            m_TextureImage,
            vk::Format::eB8G8R8A8Srgb,
            vk::ImageAspectFlagBits::eColor
        );
    }

    void CreateTextureSampler()
    {
        const vk::PhysicalDeviceProperties properties = m_PhysicalDevice.getProperties();

        const vk::SamplerCreateInfo createInfo = {
            .magFilter               = vk::Filter::eNearest,
            .minFilter               = vk::Filter::eNearest,
            .mipmapMode              = vk::SamplerMipmapMode::eLinear,
            .addressModeU            = vk::SamplerAddressMode::eRepeat,
            .addressModeV            = vk::SamplerAddressMode::eRepeat,
            .addressModeW            = vk::SamplerAddressMode::eRepeat,
            .mipLodBias              = 0.0F,
            .anisotropyEnable        = VK_TRUE,
            .maxAnisotropy           = properties.limits.maxSamplerAnisotropy,
            .compareEnable           = VK_FALSE,
            .compareOp               = vk::CompareOp::eAlways,
            .minLod                  = 0.0F,
            .maxLod                  = 0.0F,
            .borderColor             = vk::BorderColor::eIntOpaqueBlack,
            .unnormalizedCoordinates = VK_FALSE,
        };

        m_TextureSampler = m_Device.createSampler(createInfo);
    }

    void CreateVertexBuffer()
    {
        constexpr vk::DeviceSize triangleVerticesSize =
            sizeof(decltype(Vertices)::value_type) * Vertices.size();

        vk::Buffer stagingBuffer;
        vk::DeviceMemory stagingBufferMemory;
        CreateBuffer(
            triangleVerticesSize,
            vk::BufferUsageFlagBits::eTransferSrc,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
            stagingBuffer,
            stagingBufferMemory
        );

        if (void *data = m_Device.mapMemory(stagingBufferMemory, 0, triangleVerticesSize))
        {
            std::memcpy(data, Vertices.data(), triangleVerticesSize);
            m_Device.unmapMemory(stagingBufferMemory);
        }
        else
        {
            throw std::runtime_error("Failed to map staging buffer memory for vertices");
        }

        CreateBuffer(
            triangleVerticesSize,
            vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eVertexBuffer,
            vk::MemoryPropertyFlagBits::eDeviceLocal,
            m_VertexBuffer,
            m_VertexBufferMemory
        );

        CopyBuffer(stagingBuffer, m_VertexBuffer, triangleVerticesSize);
        m_Device.destroy(stagingBuffer);
        m_Device.free(stagingBufferMemory);
    }

    void CreateIndexBuffer()
    {
        constexpr vk::DeviceSize triangleIndicesSize =
            sizeof(decltype(Indices)::value_type) * Indices.size();

        vk::Buffer stagingBuffer;
        vk::DeviceMemory stagingBufferMemory;
        CreateBuffer(
            triangleIndicesSize,
            vk::BufferUsageFlagBits::eTransferSrc,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
            stagingBuffer,
            stagingBufferMemory
        );

        if (void *data = m_Device.mapMemory(stagingBufferMemory, 0, triangleIndicesSize))
        {
            std::memcpy(data, Indices.data(), triangleIndicesSize);
            m_Device.unmapMemory(stagingBufferMemory);
        }
        else
        {
            throw std::runtime_error("Failed to map staging buffer memory for indices");
        }

        CreateBuffer(
            triangleIndicesSize,
            vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eIndexBuffer,
            vk::MemoryPropertyFlagBits::eDeviceLocal,
            m_IndexBuffer,
            m_IndexBufferMemory
        );

        CopyBuffer(stagingBuffer, m_IndexBuffer, triangleIndicesSize);
        m_Device.destroy(stagingBuffer);
        m_Device.free(stagingBufferMemory);
    }

    void CreateUniformBuffers()
    {
        constexpr vk::DeviceSize bufferSize = sizeof(UniformBufferObject);

        m_UniformBuffers.resize(MaxFramesInFlight);
        m_UniformBuffersMemory.resize(MaxFramesInFlight);

        for (std::size_t i = 0; i < MaxFramesInFlight; ++i)
        {
            CreateBuffer(
                bufferSize,
                vk::BufferUsageFlagBits::eUniformBuffer,
                vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
                m_UniformBuffers[i],
                m_UniformBuffersMemory[i]
            );
        }
    }

    void CreateDescriptorPool()
    {
        const auto poolSizes = std::to_array<vk::DescriptorPoolSize>({
            {
                .type            = vk::DescriptorType::eUniformBuffer,
                .descriptorCount = MaxFramesInFlight,
            },
            {
                .type            = vk::DescriptorType::eCombinedImageSampler,
                .descriptorCount = MaxFramesInFlight,
            },
        });

        const vk::DescriptorPoolCreateInfo poolInfo = {
            .maxSets       = MaxFramesInFlight,
            .poolSizeCount = static_cast<std::uint32_t>(poolSizes.size()),
            .pPoolSizes    = poolSizes.data(),
        };

        m_DescriptorPool = m_Device.createDescriptorPool(poolInfo);
    }

    void CreateDescriptorSets()
    {
        const std::vector<vk::DescriptorSetLayout> layouts(MaxFramesInFlight, m_DescriptorSetLayout);

        const vk::DescriptorSetAllocateInfo allocInfo = {
            .descriptorPool     = m_DescriptorPool,
            .descriptorSetCount = MaxFramesInFlight,
            .pSetLayouts        = layouts.data(),
        };

        m_DescriptorSets = m_Device.allocateDescriptorSets(allocInfo);

        for (std::size_t i = 0; i < MaxFramesInFlight; ++i)
        {
            const vk::DescriptorBufferInfo bufferInfo = {
                .buffer = m_UniformBuffers[i],
                .offset = 0,
                .range  = sizeof(::UniformBufferObject),
            };

            const vk::DescriptorImageInfo imageInfo = {
                .sampler     = m_TextureSampler,
                .imageView   = m_TextureImageView,
                .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
            };

            const auto descriptorWrites = std::to_array<vk::WriteDescriptorSet>({
                {
                    .dstSet          = m_DescriptorSets[i],
                    .dstBinding      = 0,
                    .dstArrayElement = 0,
                    .descriptorCount = 1,
                    .descriptorType  = vk::DescriptorType::eUniformBuffer,
                    .pBufferInfo     = &bufferInfo,
                },
                {
                    .dstSet          = m_DescriptorSets[i],
                    .dstBinding      = 1,
                    .dstArrayElement = 0,
                    .descriptorCount = 1,
                    .descriptorType  = vk::DescriptorType::eCombinedImageSampler,
                    .pImageInfo      = &imageInfo,
                },
            });

            m_Device.updateDescriptorSets(descriptorWrites, { });
        }
    }

    std::uint32_t FindMemoryType(std::uint32_t typeFilter, vk::MemoryPropertyFlags properties)
    {
        constexpr auto filterBitCount = std::numeric_limits<decltype(typeFilter)>::digits;
        const std::bitset<filterBitCount> eligibleTypes(typeFilter);

        const auto isTypePresent = [&eligibleTypes](std::uint32_t index)
        {
            return eligibleTypes.test(index);
        };

        const vk::PhysicalDeviceMemoryProperties memoryProperties = m_PhysicalDevice.getMemoryProperties();
        const auto arePropertiesPresent = [&properties](const vk::MemoryType &memoryType)
        {
            return (memoryType.propertyFlags & properties) == properties;
        };

        for (std::uint32_t i = 0; i < memoryProperties.memoryTypeCount; ++i)
        {
            const auto &candidateType = memoryProperties.memoryTypes[i];
            if (isTypePresent(i)
                && arePropertiesPresent(candidateType))
            {
                return i;
            }
        }

        throw std::runtime_error("Failed to find a suitable memory type");
    }

    void CreateCommandBuffers()
    {
        const auto count = MaxFramesInFlight;

        const vk::CommandBufferAllocateInfo allocInfo = {
            .commandPool        = m_CommandPool,
            .level              = vk::CommandBufferLevel::ePrimary,
            .commandBufferCount = count,
        };

        m_CommandBuffers = m_Device.allocateCommandBuffers(allocInfo);
    }

    vk::CommandBuffer BeginOneTimeCommands()
    {
        const vk::CommandBufferAllocateInfo allocInfo = {
            .commandPool        = m_CommandPool,
            .level              = vk::CommandBufferLevel::ePrimary,
            .commandBufferCount = 1,
        };

        vk::CommandBuffer commandBuffer;
        (void)m_Device.allocateCommandBuffers(&allocInfo, &commandBuffer);

        const vk::CommandBufferBeginInfo beginInfo = {
            .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit,
        };

        commandBuffer.begin(beginInfo);

        return commandBuffer;
    }

    void EndOneTimeCommands(vk::CommandBuffer commandBuffer)
    {
        vkEndCommandBuffer(commandBuffer);

        const vk::SubmitInfo submitInfo = {
            .commandBufferCount = 1,
            .pCommandBuffers    = &commandBuffer,
        };

        m_GraphicsQueue.submit({ submitInfo }, VK_NULL_HANDLE);
        vkQueueWaitIdle(m_GraphicsQueue);

        m_Device.free(m_CommandPool, { commandBuffer });
    }

    void RecordCommandBuffer(vk::CommandBuffer commandBuffer, std::uint32_t imageIndex)
    {
        const vk::CommandBufferBeginInfo beginInfo = { };
        commandBuffer.begin(beginInfo);

        constexpr std::array color = { 0.01F, 0.01F, 0.01F, 0.01F };
        const auto clearValues = std::to_array<vk::ClearValue>({
            { .color = { color } },
            { .depthStencil = { 0.0F, 0 } },
        });

        const vk::Framebuffer framebuffer = m_SwapChainFramebuffers[imageIndex];
        const vk::RenderPassBeginInfo renderPassBeginInfo = {
            .renderPass      = m_RenderPass,
            .framebuffer     = framebuffer,
            .renderArea      = {
                .offset      = {
                    .x       = 0,
                    .y       = 0,
                },
                .extent      = m_SwapChainExtent,
            },
            .clearValueCount = static_cast<std::uint32_t>(clearValues.size()),
            .pClearValues    = clearValues.data(),
        };

        const vk::Viewport viewport = {
            .x        = 0.0F,
            .y        = 0.0F,
            .width    = static_cast<float>(m_SwapChainExtent.width),
            .height   = static_cast<float>(m_SwapChainExtent.height),
            .minDepth = 0.0F,
            .maxDepth = 1.0F,
        };

        const vk::Rect2D scissor = {
            .offset = {
                .x  = 0,
                .y  = 0,
            },
            .extent = m_SwapChainExtent,
        };

        const std::array vertexBuffers = {
            m_VertexBuffer,
        };

        const std::array<vk::DeviceSize, 1> offsets = {
            0,
        };

        commandBuffer.beginRenderPass(renderPassBeginInfo, vk::SubpassContents::eInline);
        {
            commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, m_GraphicsPipeline);
            commandBuffer.setViewport(0, 1, &viewport);
            commandBuffer.setScissor(0, 1, &scissor);
            commandBuffer.bindVertexBuffers(0, vertexBuffers, offsets);
            commandBuffer.bindIndexBuffer(m_IndexBuffer, 0, vk::IndexType::eUint16);
            commandBuffer.bindDescriptorSets(
                vk::PipelineBindPoint::eGraphics,
                m_PipelineLayout,
                0,
                { m_DescriptorSets[m_CurrentFrameIndex] },
                { }
            );
            commandBuffer.drawIndexed(static_cast<std::uint32_t>(Indices.size()), 1, 0, 0, 0);
        }
        commandBuffer.endRenderPass();

        commandBuffer.end();
    }

    void CreateSyncObjects()
    {
        m_ImageAvailableSemaphores.reserve(MaxFramesInFlight);
        m_RenderFinishedSemaphores.reserve(MaxFramesInFlight);
        m_InFlightFences.reserve(MaxFramesInFlight);

        const vk::FenceCreateInfo fenceInfo = {
            .flags = vk::FenceCreateFlagBits::eSignaled,
        };

        for (std::uint32_t i = 0; i < MaxFramesInFlight; ++i)
        {
            m_ImageAvailableSemaphores.push_back(m_Device.createSemaphore({ }));
            m_RenderFinishedSemaphores.push_back(m_Device.createSemaphore({ }));
            m_InFlightFences.push_back(m_Device.createFence(fenceInfo));
        }
    }

    vk::ShaderModule CreateShaderModule(std::span<const std::uint8_t> code)
    {
        const vk::ShaderModuleCreateInfo createInfo = {
            .codeSize = static_cast<std::uint32_t>(code.size()),
            .pCode    = reinterpret_cast<const std::uint32_t *>(code.data()),
        };

        return m_Device.createShaderModule(createInfo);
    }

    void EnterMainLoop()
    {
        while (!glfwWindowShouldClose(m_Window))
        {
            glfwPollEvents();
            DrawFrame();
        }

        m_Device.waitIdle();
    }

    void DrawFrame()
    {
        (void)m_Device.waitForFences(
            { m_InFlightFences[m_CurrentFrameIndex] },
            VK_TRUE,
            std::numeric_limits<std::uint64_t>::max()
        );

        std::uint32_t imageIndex = 0;
        switch (
            m_Device.acquireNextImageKHR(
                m_SwapChain,
                std::numeric_limits<std::uint64_t>::max(),
                m_ImageAvailableSemaphores[m_CurrentFrameIndex],
                nullptr,
                &imageIndex
            )
        )
        {
        case vk::Result::eSuccess:
        case vk::Result::eSuboptimalKHR:
            break;
        case vk::Result::eErrorOutOfDateKHR:
            RecreateSwapChain();
            return;
        default:
            throw std::runtime_error("Failed to acquire swapchain image");
        }

        UpdateUniformBuffer(m_CurrentFrameIndex);

        m_Device.resetFences({ m_InFlightFences[m_CurrentFrameIndex] });

        const vk::CommandBuffer commandBuffer = m_CommandBuffers[m_CurrentFrameIndex];
        commandBuffer.reset();
        RecordCommandBuffer(commandBuffer, imageIndex);

        const std::array waitSemaphores = { m_ImageAvailableSemaphores[m_CurrentFrameIndex] };

        const std::array<vk::PipelineStageFlags, 1> waitStages = { vk::PipelineStageFlagBits::eColorAttachmentOutput };

        const std::array commandBuffers = { commandBuffer };

        const std::array signalSemaphores = { m_RenderFinishedSemaphores[m_CurrentFrameIndex] };

        const vk::SubmitInfo submitInfo = {
            .waitSemaphoreCount   = static_cast<std::uint32_t>(waitSemaphores.size()),
            .pWaitSemaphores      = waitSemaphores.data(),
            .pWaitDstStageMask    = waitStages.data(),
            .commandBufferCount   = static_cast<std::uint32_t>(commandBuffers.size()),
            .pCommandBuffers      = commandBuffers.data(),
            .signalSemaphoreCount = static_cast<std::uint32_t>(signalSemaphores.size()),
            .pSignalSemaphores    = signalSemaphores.data(),
        };

        m_GraphicsQueue.submit(submitInfo, m_InFlightFences[m_CurrentFrameIndex]);

        const std::array swapchains = { m_SwapChain };

        const vk::PresentInfoKHR presentInfo = {
            .waitSemaphoreCount = static_cast<std::uint32_t>(signalSemaphores.size()),
            .pWaitSemaphores    = signalSemaphores.data(),
            .swapchainCount     = static_cast<std::uint32_t>(swapchains.size()),
            .pSwapchains        = swapchains.data(),
            .pImageIndices      = &imageIndex,
        };

        const vk::Result presentResult = m_PresentQueue.presentKHR(&presentInfo);
        if (presentResult == vk::Result::eSuboptimalKHR
            || presentResult == vk::Result::eErrorOutOfDateKHR
            || m_IsFramebufferResized)
        {
            RecreateSwapChain();
        }
        else if (presentResult != vk::Result::eSuccess)
        {
            throw std::runtime_error("Failed to present swapchain image");
        }

        m_CurrentFrameIndex = (m_CurrentFrameIndex + 1) % MaxFramesInFlight;
    }

    void TransitionImageLayout(vk::Image image, vk::ImageLayout oldLayout, vk::ImageLayout newLayout)
    {
        const vk::CommandBuffer commandBuffer = BeginOneTimeCommands();

        vk::ImageMemoryBarrier barrier = {
            .oldLayout           = oldLayout,
            .newLayout           = newLayout,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image               = image,
            .subresourceRange    = {
                .aspectMask      = vk::ImageAspectFlagBits::eColor,
                .baseMipLevel    = 0,
                .levelCount      = 1,
                .baseArrayLayer  = 0,
                .layerCount      = 1,
            },
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

        EndOneTimeCommands(commandBuffer);
    }

    void UpdateUniformBuffer(std::uint32_t currentImage) const
    {
        const float aspectRatio =
            static_cast<float>(m_SwapChainExtent.width) / static_cast<float>(m_SwapChainExtent.height);

        const UniformBufferObject ubo = {
            .Model = glm::mat4(1.0F),
            .View  = glm::lookAt(glm::vec3(0.0F), glm::vec3(0.0F, 0.0F, 1.0F), glm::vec3(0.0F, -1.0F, 0.0F)),
            .Proj  = vkm::perspective(glm::radians<float>(80.0F), aspectRatio, 0.1F),
        };

        if (void *const data = m_Device.mapMemory(m_UniformBuffersMemory[currentImage], 0, sizeof(ubo)))
        {
            std::memcpy(data, &ubo, sizeof(ubo));
            m_Device.unmapMemory(m_UniformBuffersMemory[currentImage]);
        }
        else
        {
            throw std::runtime_error("Failed to map memory for uniform buffer");
        }
    }

    void RecreateSwapChain()
    {
        m_IsFramebufferResized = false;

        int width = 0;
        int height = 0;
        glfwGetFramebufferSize(m_Window, &width, &height);
        if (width == 0 || height == 0)
        {
            return;
        }

        m_Device.waitIdle();

        CleanupSwapChain();

        CreateSwapChain();
        CreateImageViews();
        CreateDepthResources();
        CreateFramebuffers();
    }

    void CleanupSwapChain()
    {
        m_Device.destroy(m_DepthImageView);
        m_Device.destroy(m_DepthImage);
        m_Device.free(m_DepthImageMemory);

        for (auto framebuffer : m_SwapChainFramebuffers)
        {
            m_Device.destroy(framebuffer);
        }
        m_SwapChainFramebuffers.clear();

        for (auto imageView : m_SwapChainImageViews)
        {
            m_Device.destroy(imageView);
        }
        m_SwapChainImageViews.clear();

        m_Device.destroy(m_OldSwapChain);
    }

    void Cleanup()
    {
        CleanupSwapChain();
        m_Device.destroy(m_SwapChain);

        m_Device.destroy(m_TextureSampler);
        m_Device.destroy(m_TextureImageView);
        m_Device.destroy(m_TextureImage);
        m_Device.free(m_TextureImageMemory);

        for (std::size_t i = 0; i < MaxFramesInFlight; ++i)
        {
            m_Device.destroy(m_UniformBuffers[i]);
            m_Device.free(m_UniformBuffersMemory[i]);
        }

        m_Device.destroy(m_DescriptorPool);
        m_Device.destroy(m_DescriptorSetLayout);

        m_Device.destroy(m_RenderPass);

        m_Device.destroy(m_VertexBuffer);
        m_Device.free(m_VertexBufferMemory);
        m_Device.destroy(m_IndexBuffer);
        m_Device.free(m_IndexBufferMemory);

        for (std::uint32_t i = 0; i < MaxFramesInFlight; ++i)
        {
            m_Device.destroy(m_ImageAvailableSemaphores[i]);
            m_Device.destroy(m_RenderFinishedSemaphores[i]);
            m_Device.destroy(m_InFlightFences[i]);
        }

        m_Device.destroy(m_CommandPool);

        m_Device.destroy(m_GraphicsPipeline);
        m_Device.destroy(m_PipelineLayout);

        m_Device.destroy();

        m_Instance.destroy(m_Surface);

        m_Instance.destroy(m_DebugMessenger);
        m_Instance.destroy();
        glfwDestroyWindow(m_Window);
        glfwTerminate();
    }

    static void FramebufferSizeCallback(GLFWwindow *window, [[maybe_unused]] int width, [[maybe_unused]] int height)
    {
        auto *app = static_cast<::Application *>(glfwGetWindowUserPointer(window));
        app->m_IsFramebufferResized = true;
    }

    static void KeyCallback(GLFWwindow *window, int key, int scancode, int action, int modifiers)
    {
        (void)scancode;
        (void)modifiers;

        if (key == GLFW_KEY_F11 && action == GLFW_PRESS)
        {
            auto *app = static_cast<::Application *>(glfwGetWindowUserPointer(window));

            if (app->m_IsFullscreen)
            {
                glfwSetWindowMonitor(
                    window,
                    nullptr,
                    app->m_WindowLastXPosition,
                    app->m_WindowLastYPosition,
                    WindowWidth,
                    WindowHeight,
                    GLFW_DONT_CARE
                );
            }
            else
            {
                GLFWmonitor *monitor = glfwGetPrimaryMonitor();
                const GLFWvidmode *mode = glfwGetVideoMode(monitor);
                glfwGetWindowPos(window, &app->m_WindowLastXPosition, &app->m_WindowLastYPosition);
                glfwSetWindowMonitor(window, monitor, 0, 0, mode->width, mode->height, GLFW_DONT_CARE);
            }

            app->m_IsFullscreen = !app->m_IsFullscreen;
        }
    }

    static bool IsRequiredExtensionsSupported(const vk::PhysicalDevice &device)
    {
        auto extensions = device.enumerateDeviceExtensionProperties();
        std::ranges::sort(extensions, { }, &vk::ExtensionProperties::extensionName);

        return std::ranges::includes(
            extensions,
            DeviceExtensions,
            { },
            [](const vk::ExtensionProperties &extension)
            {
                return static_cast<std::string_view>(extension.extensionName);
            }
        );
    }

    static vk::SurfaceFormatKHR SelectSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR> &surfaceFormats)
    {
        const auto candidate = std::ranges::find_if(surfaceFormats, [](const vk::SurfaceFormatKHR &surfaceFormat)
        {
            return surfaceFormat.format == vk::Format::eB8G8R8A8Srgb
                   && surfaceFormat.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear;
        });

        if (candidate == surfaceFormats.end())
        {
            return surfaceFormats.front();
        }

        return *candidate;
    }

    static vk::PresentModeKHR SelectSwapPresentMode(const std::vector<vk::PresentModeKHR> &presentModes)
    {
        const auto candidate = std::ranges::find(presentModes, vk::PresentModeKHR::eMailbox);

        if (candidate == presentModes.end())
        {
            return vk::PresentModeKHR::eFifo;
        }

        return *candidate;
    }

private:
    static constexpr std::uint32_t MaxFramesInFlight = 2;

private:
    std::filesystem::path m_ResourcesPath;

    vk::DynamicLoader m_Loader;

    bool m_IsFullscreen = false;
    bool m_IsFramebufferResized = false;
    int m_WindowLastXPosition = 0;
    int m_WindowLastYPosition = 0;
    GLFWwindow *m_Window = nullptr;

    vk::Instance m_Instance;
    vk::DebugUtilsMessengerEXT m_DebugMessenger;
    vk::SurfaceKHR m_Surface;
    vk::PhysicalDevice m_PhysicalDevice;
    vk::Device m_Device;
    vk::Queue m_GraphicsQueue;
    vk::Queue m_PresentQueue;

    vk::Format m_SwapChainImageFormat = vk::Format::eUndefined;
    vk::Extent2D m_SwapChainExtent;
    vk::SwapchainKHR m_SwapChain;
    vk::SwapchainKHR m_OldSwapChain;
    std::vector<vk::Image> m_SwapChainImages;
    std::vector<vk::ImageView> m_SwapChainImageViews;

    vk::RenderPass m_RenderPass;

    vk::DescriptorSetLayout m_DescriptorSetLayout;
    vk::PipelineLayout m_PipelineLayout;
    vk::Pipeline m_GraphicsPipeline;

    std::vector<vk::Framebuffer> m_SwapChainFramebuffers;

    vk::CommandPool m_CommandPool;

    vk::Image m_DepthImage;
    vk::DeviceMemory m_DepthImageMemory;
    vk::ImageView m_DepthImageView;

    vk::Image m_TextureImage;
    vk::DeviceMemory m_TextureImageMemory;
    vk::ImageView m_TextureImageView;
    vk::Sampler m_TextureSampler;

    vk::Buffer m_VertexBuffer;
    vk::DeviceMemory m_VertexBufferMemory;

    vk::Buffer m_IndexBuffer;
    vk::DeviceMemory m_IndexBufferMemory;

    std::vector<vk::Buffer> m_UniformBuffers;
    std::vector<vk::DeviceMemory> m_UniformBuffersMemory;

    vk::DescriptorPool m_DescriptorPool;
    std::vector<vk::DescriptorSet> m_DescriptorSets;

    std::uint32_t m_CurrentFrameIndex = 0;
    std::vector<vk::CommandBuffer> m_CommandBuffers;
    std::vector<vk::Semaphore> m_ImageAvailableSemaphores;
    std::vector<vk::Semaphore> m_RenderFinishedSemaphores;
    std::vector<vk::Fence> m_InFlightFences;
};

constexpr std::string_view ResourcesPathFlag = "--resources-path";
constexpr char FlagDelimiter = '=';

int main(int argc, char *argv[])
{
    --argc;
    ++argv;

    Application app;

    for (int i = 0; i < argc; ++i)
    {
        const std::string_view arg = argv[i];
        if (arg.starts_with(ResourcesPathFlag))
        {
            if (const std::size_t delimiterIndex = arg.find(FlagDelimiter); delimiterIndex != std::string_view::npos)
            {
                if (const std::string_view path = arg.substr(delimiterIndex + 1); !path.empty())
                {
                    app.SetResourcesPath(path.data());
                }
            }
            else
            {
                std::cerr << "\"" << ResourcesPathFlag << "\" was specified, but no value was given\n";
                std::cerr << "Usage: " << ResourcesPathFlag << FlagDelimiter << "{ABSOLUTE_PATH_TO_DIR}\n";
                return EXIT_FAILURE;
            }
        }
        else
        {
            std::cerr << "Unrecognised argument: \"" << arg << "\" - ignoring\n";
        }
    }

    try
    {
        app.Run();
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << '\n';
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
