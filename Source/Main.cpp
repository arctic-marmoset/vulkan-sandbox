#include "Camera.hpp"
#include "Device.hpp"
#include "Model.hpp"
#include "Tga.hpp"
#include "Utility.hpp"

#include <fmt/format.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <spdlog/spdlog.h>
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
#include <limits>
#include <numeric>
#include <stdexcept>

VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

constexpr std::array<const char *, 1> DeviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME,
};

constexpr std::uint32_t WindowWidth = 1280;
constexpr std::uint32_t WindowHeight = 720;

constexpr std::array<const char *, 1> ValidationLayers = {
    "VK_LAYER_KHRONOS_validation",
};

constexpr bool IsDebugMode = true;

constexpr std::array ArenaVertices = {
    // Ground
    Vertex{ .Position = { -10.0F,  1.8F, -10.0F }, .TexCoord = {  0.0F,  0.0F  } },
    Vertex{ .Position = {  10.0F,  1.8F, -10.0F }, .TexCoord = { 10.0F,  0.0F  } },
    Vertex{ .Position = {  10.0F,  1.8F,  10.0F }, .TexCoord = { 10.0F, 10.0F  } },
    Vertex{ .Position = { -10.0F,  1.8F,  10.0F }, .TexCoord = {  0.0F, 10.0F  } },

    // Front Wall
    Vertex{ .Position = { -10.0F,  1.8F,  10.0F }, .TexCoord = {  0.0F,  0.0F  } },
    Vertex{ .Position = {  10.0F,  1.8F,  10.0F }, .TexCoord = { 10.0F,  0.0F  } },
    Vertex{ .Position = {  10.0F, -0.7F,  10.0F }, .TexCoord = { 10.0F,  1.25F } },
    Vertex{ .Position = { -10.0F, -0.7F,  10.0F }, .TexCoord = {  0.0F,  1.25F } },

    // Right Wall
    Vertex{ .Position = {  10.0F,  1.8F,  10.0F }, .TexCoord = {  0.0F,  0.0F  } },
    Vertex{ .Position = {  10.0F,  1.8F, -10.0F }, .TexCoord = { 10.0F,  0.0F  } },
    Vertex{ .Position = {  10.0F, -0.7F, -10.0F }, .TexCoord = { 10.0F,  1.25F } },
    Vertex{ .Position = {  10.0F, -0.7F,  10.0F }, .TexCoord = {  0.0F,  1.25F } },

    // Back Wall
    Vertex{ .Position = {  10.0F,  1.8F, -10.0F }, .TexCoord = {  0.0F,  0.0F  } },
    Vertex{ .Position = { -10.0F,  1.8F, -10.0F }, .TexCoord = { 10.0F,  0.0F  } },
    Vertex{ .Position = { -10.0F, -0.7F, -10.0F }, .TexCoord = { 10.0F,  1.25F } },
    Vertex{ .Position = {  10.0F, -0.7F, -10.0F }, .TexCoord = {  0.0F,  1.25F } },

    // Left Wall
    Vertex{ .Position = { -10.0F,  1.8F, -10.0F }, .TexCoord = {  0.0F,  0.0F  } },
    Vertex{ .Position = { -10.0F,  1.8F,  10.0F }, .TexCoord = { 10.0F,  0.0F  } },
    Vertex{ .Position = { -10.0F, -0.7F,  10.0F }, .TexCoord = { 10.0F,  1.25F } },
    Vertex{ .Position = { -10.0F, -0.7F, -10.0F }, .TexCoord = {  0.0F,  1.25F } },
};

constexpr std::array ArenaIndices =
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
        spdlog::trace("[VULKAN] {}", data->pMessage);
        break;
    case VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT:
        spdlog::debug("[VULKAN] {}", data->pMessage);
        break;
    case VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT:
        spdlog::warn("[VULKAN] {}", data->pMessage);
        break;
    default:
        spdlog::warn("[VULKAN] Unknown validation layer debug message severity - assuming ERROR level");
        [[fallthrough]];
    case VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT:
        spdlog::error("[VULKAN] {}", data->pMessage);
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

        std::memcpy(m_Title.data(), DefaultTitle, std::size(DefaultTitle));
        m_Window = glfwCreateWindow(WindowWidth, WindowHeight, m_Title.data(), nullptr, nullptr);

        if (!m_Window)
        {
            throw std::runtime_error("Failed to create window");
        }

        glfwSetWindowUserPointer(m_Window, this);
        glfwSetFramebufferSizeCallback(m_Window, FramebufferSizeCallback);
        glfwSetKeyCallback(m_Window, KeyCallback);
        glfwSetCursorPosCallback(m_Window, CursorPositionCallback);

        glfwSetInputMode(m_Window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
        if (glfwRawMouseMotionSupported())
        {
            glfwSetInputMode(m_Window, GLFW_RAW_MOUSE_MOTION, GLFW_TRUE);
        }
    }

    void InitVulkan()
    {
        CreateInstance();
        AttachDebugMessenger();
        CreateSurface();
        const auto [physicalDevice, queueFamilyIndices] = SelectPhysicalDevice(m_Instance);
        m_Device.Init(physicalDevice, queueFamilyIndices, DeviceExtensions);
        CreateSwapChain();
        CreateImageViews();
        CreateRenderPass();
        CreateDescriptorSetLayout();
        CreateGraphicsPipeline();
        CreateCommandPool();
        CreateDepthResources();
        CreateFramebuffers();
        CreateTextureSampler();
        CreateDescriptorPool();
        CreateResources();
        CreateUniformBuffers();
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

    [[nodiscard]]
    std::pair<vk::PhysicalDevice, Device::QueueFamilyIndices> SelectPhysicalDevice(vk::Instance instance)
    {
        const auto devices = instance.enumeratePhysicalDevices();

        if (devices.empty())
        {
            throw std::runtime_error("Failed to find a GPU with Vulkan support");
        }

        Device::QueueFamilyIndices indices;
        SwapChainSupportDetails details;
        for (const vk::PhysicalDevice &device : devices)
        {
            indices = Device::QueueFamilyIndices::Find(device, m_Surface);
            details = QuerySwapChainSupport(device, m_Surface);

            if (IsRequiredExtensionsSupported(device)
                && indices.IsComplete()
                && (!details.Formats.empty() && !details.PresentModes.empty()))
            {
                return { device, indices };
            }
        }

        throw std::runtime_error("Failed to find a GPU suitable for this application");
    }

    [[nodiscard]]
    static SwapChainSupportDetails QuerySwapChainSupport(const vk::PhysicalDevice &device, vk::SurfaceKHR surface)
    {
        return {
            .Capabilities = device.getSurfaceCapabilitiesKHR(surface),
            .Formats      = device.getSurfaceFormatsKHR(surface),
            .PresentModes = device.getSurfacePresentModesKHR(surface),
        };
    }

    void CreateSwapChain()
    {
        const auto swapChainSupport = QuerySwapChainSupport(m_Device.GetPhysicalDevice(), m_Surface);

        const auto surfaceFormat = SelectSwapSurfaceFormat(swapChainSupport.Formats);
        const auto presentMode = SelectSwapPresentMode(swapChainSupport.PresentModes);
        const auto extent = SelectSwapExtent(swapChainSupport.Capabilities);

        const std::uint32_t desiredImageCount = swapChainSupport.Capabilities.minImageCount + 1;
        const std::uint32_t maxImageCount = swapChainSupport.Capabilities.maxImageCount;

        const auto imageCount = maxImageCount == 0
            ? desiredImageCount
            : std::min(desiredImageCount, maxImageCount);

        const std::array queueFamilyIndices = m_Device.GetQueueFamilyIndices().ToArray();

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

            if (m_Device.GetQueueFamilyIndices().Graphics != m_Device.GetQueueFamilyIndices().Present)
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

        m_SwapChain = m_Device.GetHandle().createSwapchainKHR(createInfo);
        m_SwapChainImages = m_Device.GetHandle().getSwapchainImagesKHR(m_SwapChain);
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

    void CreateImageViews()
    {
        m_SwapChainImageViews.reserve(m_SwapChainImages.size());

        for (const auto &image : m_SwapChainImages)
        {
            const auto imageView = m_Device.CreateImageView(
                image,
                m_SwapChainImageFormat,
                vk::ImageAspectFlagBits::eColor
            );

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

        m_RenderPass = m_Device.GetHandle().createRenderPass(renderPassCreateInfo);
    }

    void CreateDescriptorSetLayout()
    {
        const vk::DescriptorSetLayoutBinding uboLayoutBinding = {
            .binding         = 0, // layout(set = 0, binding = 0) uniform UniformBufferObject.
            .descriptorType  = vk::DescriptorType::eUniformBuffer,
            .descriptorCount = 1,
            .stageFlags      = vk::ShaderStageFlagBits::eVertex,
        };

        const vk::DescriptorSetLayoutCreateInfo uboLayoutInfo = {
            .bindingCount = 1,
            .pBindings    = &uboLayoutBinding,
        };

        m_UboDescriptorSetLayout = m_Device.GetHandle().createDescriptorSetLayout(uboLayoutInfo);

        const vk::DescriptorSetLayoutBinding samplerLayoutBinding = {
            .binding         = 0, // layout(set = 1, binding = 0) uniform sampler2D texSampler.
            .descriptorType  = vk::DescriptorType::eCombinedImageSampler,
            .descriptorCount = 1,
            .stageFlags      = vk::ShaderStageFlagBits::eFragment,
        };

        const vk::DescriptorSetLayoutCreateInfo samplerLayoutInfo = {
            .bindingCount = 1,
            .pBindings    = &samplerLayoutBinding,
        };

        m_SamplerDescriptorSetLayout = m_Device.GetHandle().createDescriptorSetLayout(samplerLayoutInfo);
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

        const std::array descriptorSetLayouts = { m_UboDescriptorSetLayout, m_SamplerDescriptorSetLayout };

        const vk::PipelineLayoutCreateInfo pipelineLayoutCreateInfo = {
            .setLayoutCount = static_cast<std::uint32_t>(descriptorSetLayouts.size()),
            .pSetLayouts    = descriptorSetLayouts.data(),
        };

        m_PipelineLayout = m_Device.GetHandle().createPipelineLayout(pipelineLayoutCreateInfo);

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

        const auto [result, pipeline] = m_Device.GetHandle().createGraphicsPipeline({ }, pipelineCreateInfo);

        if (result != vk::Result::eSuccess)
        {
            throw std::runtime_error("Failed to create graphics pipeline");
        }

        m_GraphicsPipeline = pipeline;

        m_Device.GetHandle().destroy(vsModule);
        m_Device.GetHandle().destroy(fsModule);
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

            const auto framebuffer = m_Device.GetHandle().createFramebuffer(createInfo);
            m_SwapChainFramebuffers.push_back(framebuffer);
        }
    }

    void CreateCommandPool()
    {
        const vk::CommandPoolCreateInfo createInfo = {
            .flags            = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
            .queueFamilyIndex = m_Device.GetQueueFamilyIndices().Graphics,
        };

        m_CommandPool = m_Device.GetHandle().createCommandPool(createInfo);
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
            const vk::FormatProperties properties = m_Device.GetPhysicalDevice().getFormatProperties(*it);
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

        m_DepthImage = m_Device.CreateImage(
            m_SwapChainExtent.width,
            m_SwapChainExtent.height,
            depthFormat,
            vk::ImageTiling::eOptimal,
            vk::ImageUsageFlagBits::eDepthStencilAttachment,
            vk::MemoryPropertyFlagBits::eDeviceLocal
        );

        m_DepthImageView = m_Device.CreateImageView(m_DepthImage.Handle, depthFormat, vk::ImageAspectFlagBits::eDepth);
    }

    void CreateResources()
    {
        const vk::CommandBuffer loadResourcesCommands = BeginOneTimeCommands();

        // Transfer arena vertices to GPU memory.

        constexpr vk::DeviceSize arenaVerticesSize = SizeInBytes(ArenaVertices);

        Buffer arenaVerticesStagingBuffer = m_Device.CreateBuffer(
            arenaVerticesSize,
            vk::BufferUsageFlagBits::eTransferSrc,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent
        );

        if (void *data = m_Device.GetHandle().mapMemory(arenaVerticesStagingBuffer.Memory, 0, arenaVerticesSize))
        {
            std::memcpy(data, ArenaVertices.data(), arenaVerticesSize);
            m_Device.GetHandle().unmapMemory(arenaVerticesStagingBuffer.Memory);
        }
        else
        {
            throw std::runtime_error("Failed to map staging buffer memory for vertices");
        }

        m_ArenaVertexBuffer = m_Device.CreateBuffer(
            arenaVerticesSize,
            vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eVertexBuffer,
            vk::MemoryPropertyFlagBits::eDeviceLocal
        );

        CopyBuffer(
            loadResourcesCommands,
            arenaVerticesStagingBuffer.Handle,
            m_ArenaVertexBuffer.Handle,
            arenaVerticesSize
        );

        // Transfer arena indices to GPU memory.

        constexpr vk::DeviceSize arenaIndicesSize = SizeInBytes(ArenaIndices);

        Buffer arenaIndicesStagingBuffer = m_Device.CreateBuffer(
            arenaIndicesSize,
            vk::BufferUsageFlagBits::eTransferSrc,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent
        );

        WithMappedMemory(m_Device.GetHandle(), arenaIndicesStagingBuffer.Memory, 0, arenaIndicesSize, [&](void *destination)
        {
            std::memcpy(destination, ArenaIndices.data(), arenaIndicesSize);
        });

        m_ArenaIndexBuffer = m_Device.CreateBuffer(
            arenaIndicesSize,
            vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eIndexBuffer,
            vk::MemoryPropertyFlagBits::eDeviceLocal
        );

        CopyBuffer(
            loadResourcesCommands,
            arenaIndicesStagingBuffer.Handle,
            m_ArenaIndexBuffer.Handle,
            arenaIndicesSize
        );

        // Copy textures to GPU memory.

        const Tga::Image missingTexture = Tga::Image::Load(GetResourcePath("Textures/Missing_Raw.tga").c_str());

        Buffer missingTextureStagingBuffer = m_Device.CreateBuffer(
            missingTexture.GetSize(),
            vk::BufferUsageFlagBits::eTransferSrc,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent
        );

        WithMappedMemory(m_Device.GetHandle(), missingTextureStagingBuffer.Memory, 0, missingTexture.GetSize(), [&](void *destination)
        {
            std::memcpy(destination, missingTexture.Pixels.data(), missingTexture.Pixels.size());
        });

        m_MissingTextureImage = m_Device.CreateImage(
            missingTexture.Width,
            missingTexture.Height,
            vk::Format::eB8G8R8A8Srgb,
            vk::ImageTiling::eOptimal,
            vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled,
            vk::MemoryPropertyFlagBits::eDeviceLocal
        );

        m_MissingTextureImageView = m_Device.CreateImageView(
            m_MissingTextureImage.Handle,
            vk::Format::eB8G8R8A8Srgb,
            vk::ImageAspectFlagBits::eColor
        );

        TransitionImageLayout(
            loadResourcesCommands,
            m_MissingTextureImage.Handle,
            vk::ImageLayout::eUndefined,
            vk::ImageLayout::eTransferDstOptimal
        );

        CopyBufferToImage(
            loadResourcesCommands,
            missingTextureStagingBuffer.Handle,
            m_MissingTextureImage.Handle,
            missingTexture.Width,
            missingTexture.Height
        );

        TransitionImageLayout(
            loadResourcesCommands,
            m_MissingTextureImage.Handle,
            vk::ImageLayout::eTransferDstOptimal,
            vk::ImageLayout::eShaderReadOnlyOptimal
        );

        // Load dragon mesh.

        m_DragonModel.Load(
            GetResourcePath("Meshes/StanfordDragon.obj").c_str(),
            GetResourcePath("Textures/StanfordDragon_Albedo_Raw.tga").c_str()
        );

        std::array stagingBuffers = m_DragonModel.Init(
            m_Device,
            m_SamplerDescriptorSetLayout,
            m_DescriptorPool,
            m_TextureSampler,
            loadResourcesCommands
        );

        EndOneTimeCommands(loadResourcesCommands);

        missingTextureStagingBuffer.Destroy(m_Device.GetHandle());
        arenaIndicesStagingBuffer.Destroy(m_Device.GetHandle());
        arenaVerticesStagingBuffer.Destroy(m_Device.GetHandle());

        for (Buffer &buffer : stagingBuffers)
        {
            buffer.Destroy(m_Device.GetHandle());
        }
    }

    void CreateTextureSampler()
    {
        const vk::PhysicalDeviceProperties &properties = m_Device.GetProperties();

        const vk::SamplerCreateInfo createInfo = {
            .magFilter               = vk::Filter::eLinear,
            .minFilter               = vk::Filter::eLinear,
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

        m_TextureSampler = m_Device.GetHandle().createSampler(createInfo);
    }

    void CreateUniformBuffers()
    {
        constexpr vk::DeviceSize bufferSize = sizeof(UniformBufferObject);

        m_UniformBuffers.resize(MaxFramesInFlight);

        for (std::size_t i = 0; i < MaxFramesInFlight; ++i)
        {
            m_UniformBuffers[i] = m_Device.CreateBuffer(
                bufferSize,
                vk::BufferUsageFlagBits::eUniformBuffer,
                vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent
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
                .descriptorCount = TextureCount,
            },
        });

        const vk::DescriptorPoolCreateInfo poolInfo = {
            .maxSets       = MaxFramesInFlight + TextureCount, // One UBO per frame, and one set per texture.
            .poolSizeCount = static_cast<std::uint32_t>(poolSizes.size()),
            .pPoolSizes    = poolSizes.data(),
        };

        m_DescriptorPool = m_Device.GetHandle().createDescriptorPool(poolInfo);
    }

    void CreateDescriptorSets()
    {
        const std::vector<vk::DescriptorSetLayout> uboLayouts(MaxFramesInFlight, m_UboDescriptorSetLayout);

        const vk::DescriptorSetAllocateInfo uboAllocInfo = {
            .descriptorPool     = m_DescriptorPool,
            .descriptorSetCount = MaxFramesInFlight,
            .pSetLayouts        = uboLayouts.data(),
        };

        m_UboDescriptorSets = m_Device.GetHandle().allocateDescriptorSets(uboAllocInfo);

        for (std::size_t i = 0; i < MaxFramesInFlight; ++i)
        {
            const vk::DescriptorBufferInfo bufferInfo = {
                .buffer = m_UniformBuffers[i].Handle,
                .offset = 0,
                .range  = sizeof(UniformBufferObject),
            };

            const vk::WriteDescriptorSet descriptorWrite = {
                .dstSet          = m_UboDescriptorSets[i],
                .dstBinding      = 0,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType  = vk::DescriptorType::eUniformBuffer,
                .pBufferInfo     = &bufferInfo,
            };

            m_Device.GetHandle().updateDescriptorSets({ descriptorWrite }, { });
        }

        const vk::DescriptorSetAllocateInfo textureAllocInfo = {
            .descriptorPool     = m_DescriptorPool,
            .descriptorSetCount = 1,
            .pSetLayouts        = &m_SamplerDescriptorSetLayout,
        };

        (void)m_Device.GetHandle().allocateDescriptorSets(&textureAllocInfo, &m_MissingTextureDescriptorSet);

        const vk::DescriptorImageInfo missingTextureImageInfo = {
            .sampler     = m_TextureSampler,
            .imageView   = m_MissingTextureImageView,
            .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
        };

        const vk::WriteDescriptorSet missingTextureDescriptorWrite = {
            .dstSet          = m_MissingTextureDescriptorSet,
            .dstBinding      = 0,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType  = vk::DescriptorType::eCombinedImageSampler,
            .pImageInfo      = &missingTextureImageInfo,
        };

        m_Device.GetHandle().updateDescriptorSets({ missingTextureDescriptorWrite }, { });
    }

    void CreateCommandBuffers()
    {
        const auto count = MaxFramesInFlight;

        const vk::CommandBufferAllocateInfo allocInfo = {
            .commandPool        = m_CommandPool,
            .level              = vk::CommandBufferLevel::ePrimary,
            .commandBufferCount = count,
        };

        m_CommandBuffers = m_Device.GetHandle().allocateCommandBuffers(allocInfo);
    }

    vk::CommandBuffer BeginOneTimeCommands()
    {
        const vk::CommandBufferAllocateInfo allocInfo = {
            .commandPool        = m_CommandPool,
            .level              = vk::CommandBufferLevel::ePrimary,
            .commandBufferCount = 1,
        };

        vk::CommandBuffer commandBuffer;
        (void)m_Device.GetHandle().allocateCommandBuffers(&allocInfo, &commandBuffer);

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

        m_Device.GetGraphicsQueue().submit({ submitInfo }, VK_NULL_HANDLE);
        m_Device.GetGraphicsQueue().waitIdle();

        m_Device.GetHandle().free(m_CommandPool, { commandBuffer });
    }

    void RecordCommandBuffer(vk::CommandBuffer commandBuffer, std::uint32_t imageIndex)
    {
        const vk::CommandBufferBeginInfo beginInfo = { };
        commandBuffer.begin(beginInfo);

        constexpr std::array darkGrey = { 0.01F, 0.01F, 0.01F, 1.0F };
        constexpr std::array skyBlue = { 0.576F, 0.827F, 0.929F, 1.0F };
        const auto clearValues = std::to_array<vk::ClearValue>({
            { .color = { skyBlue } },
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

        commandBuffer.beginRenderPass(renderPassBeginInfo, vk::SubpassContents::eInline);
        {
            commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, m_GraphicsPipeline);
            commandBuffer.setViewport(0, 1, &viewport);
            commandBuffer.setScissor(0, 1, &scissor);

            commandBuffer.bindDescriptorSets(
                vk::PipelineBindPoint::eGraphics,
                m_PipelineLayout,
                0,
                { m_UboDescriptorSets[m_CurrentFrameIndex] },
                { }
            );

            commandBuffer.bindDescriptorSets(
                vk::PipelineBindPoint::eGraphics,
                m_PipelineLayout,
                1,
                { m_MissingTextureDescriptorSet },
                { }
            );
            commandBuffer.bindVertexBuffers(0, { m_ArenaVertexBuffer.Handle }, { 0 });
            commandBuffer.bindIndexBuffer(m_ArenaIndexBuffer.Handle, 0, vk::IndexType::eUint16);
            commandBuffer.drawIndexed(static_cast<std::uint32_t>(ArenaIndices.size()), 1, 0, 0, 0);

            commandBuffer.bindDescriptorSets(
                vk::PipelineBindPoint::eGraphics,
                m_PipelineLayout,
                1,
                { m_DragonModel.Texture.DescriptorSet },
                { }
            );
            commandBuffer.bindVertexBuffers(0, { m_DragonModel.Vertices.Buffer.Handle }, { 0 });
            commandBuffer.bindIndexBuffer(m_DragonModel.Indices.Buffer.Handle, 0, vk::IndexType::eUint32);
            commandBuffer.drawIndexed(m_DragonModel.Indices.Count, 1, 0, 0, 0);
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
            m_ImageAvailableSemaphores.push_back(m_Device.GetHandle().createSemaphore({ }));
            m_RenderFinishedSemaphores.push_back(m_Device.GetHandle().createSemaphore({ }));
            m_InFlightFences.push_back(m_Device.GetHandle().createFence(fenceInfo));
        }
    }

    vk::ShaderModule CreateShaderModule(std::span<const std::uint8_t> code)
    {
        const vk::ShaderModuleCreateInfo createInfo = {
            .codeSize = static_cast<std::uint32_t>(code.size()),
            .pCode    = reinterpret_cast<const std::uint32_t *>(code.data()),
        };

        return m_Device.GetHandle().createShaderModule(createInfo);
    }

    void EnterMainLoop()
    {
        m_CurrentTick = std::chrono::steady_clock::now();
        m_LastTick = m_CurrentTick - std::chrono::milliseconds(33);

        while (!glfwWindowShouldClose(m_Window))
        {
            m_FrameTimeSum -= m_FrameTimeSamples[m_FrameTimeSampleIndex];
            m_FrameTimeSamples[m_FrameTimeSampleIndex] = std::chrono::duration<float>(m_CurrentTick - m_LastTick).count();

            const float deltaTime = m_FrameTimeSamples[m_FrameTimeSampleIndex];
            m_FrameTimeSampleIndex = (m_FrameTimeSampleIndex + 1) % m_FrameTimeSamples.size();
            m_FrameTimeSum += deltaTime;
            m_SmoothedFrameTime = m_FrameTimeSum / (float)m_FrameTimeSamples.size();

            HandleInput();
            UpdateState(deltaTime);
            DrawFrame();
            glfwPollEvents();

            m_LastTick = m_CurrentTick;
            m_CurrentTick = std::chrono::steady_clock::now();
        }

        m_Device.GetHandle().waitIdle();
    }

    void HandleInput()
    {
        if (glfwGetKey(m_Window, GLFW_KEY_R) == GLFW_PRESS)
        {
            m_Camera.ResetAsync();
        }
        if (glfwGetKey(m_Window, GLFW_KEY_W) == GLFW_PRESS)
        {
            m_Camera.MoveForwardAsync();
        }
        if (glfwGetKey(m_Window, GLFW_KEY_S) == GLFW_PRESS)
        {
            m_Camera.MoveBackwardAsync();
        }
        if (glfwGetKey(m_Window, GLFW_KEY_A) == GLFW_PRESS)
        {
            m_Camera.StrafeLeftAsync();
        }
        if (glfwGetKey(m_Window, GLFW_KEY_D) == GLFW_PRESS)
        {
            m_Camera.StrafeRightAsync();
        }
        if (glfwGetKey(m_Window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS)
        {
            m_Camera.SetShiftSpeedAsync();
        }
        if (glfwGetKey(m_Window, GLFW_KEY_LEFT_ALT) == GLFW_PRESS)
        {
            m_Camera.SetAltSpeedAsync();
        }
    }

    void UpdateState(float deltaTime)
    {
        const auto deltaX = static_cast<float>(m_MousePos.XPos - m_LastMousePos.XPos);
        const auto deltaY = static_cast<float>(m_MousePos.YPos - m_LastMousePos.YPos);

        m_Camera.Update(deltaTime, deltaX, deltaY);
        UpdateTitle();

        m_LastMousePos = m_MousePos;
    }

    void UpdateTitle()
    {
        const auto result = fmt::format_to_n(
            m_Title.begin(),
            m_Title.size(),
            "{} | Frame Time: {:>7.3f} ms | Position: ({:.2f}, {:.2f}, {:.2f}) | Heading: ({:.2f}, {:.2f}, {:.2f})",
            DefaultTitle,
            m_SmoothedFrameTime * 1000.0F,
            m_Camera.GetPosition().x,
            m_Camera.GetPosition().y,
            m_Camera.GetPosition().z,
            m_Camera.GetLookDirection().x,
            m_Camera.GetLookDirection().y,
            m_Camera.GetLookDirection().z
        );

        if (std::size_t size = m_Title.size(); size < result.size)
        {
            m_Title[size - 3] = '.';
            m_Title[size - 2] = '.';
            m_Title[size - 1] = '\0';
        }
        else
        {
            *result.out = '\0';
        }

        glfwSetWindowTitle(m_Window, m_Title.data());
    }

    void DrawFrame()
    {
        (void)m_Device.GetHandle().waitForFences(
            { m_InFlightFences[m_CurrentFrameIndex] },
            VK_TRUE,
            std::numeric_limits<std::uint64_t>::max()
        );

        std::uint32_t imageIndex = 0;
        switch (
            m_Device.GetHandle().acquireNextImageKHR(
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

        m_Device.GetHandle().resetFences({ m_InFlightFences[m_CurrentFrameIndex] });

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

        m_Device.GetGraphicsQueue().submit(submitInfo, m_InFlightFences[m_CurrentFrameIndex]);

        const std::array swapchains = { m_SwapChain };

        const vk::PresentInfoKHR presentInfo = {
            .waitSemaphoreCount = static_cast<std::uint32_t>(signalSemaphores.size()),
            .pWaitSemaphores    = signalSemaphores.data(),
            .swapchainCount     = static_cast<std::uint32_t>(swapchains.size()),
            .pSwapchains        = swapchains.data(),
            .pImageIndices      = &imageIndex,
        };

        const vk::Result presentResult = m_Device.GetPresentQueue().presentKHR(&presentInfo);
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

    void UpdateUniformBuffer(std::uint32_t currentImage) const
    {
        const float aspectRatio =
            static_cast<float>(m_SwapChainExtent.width) / static_cast<float>(m_SwapChainExtent.height);

        const UniformBufferObject ubo = {
            .Model = glm::mat4(1.0F),
            .View  = m_Camera.GetViewMatrix(),
            .Proj  = vkm::perspective(glm::radians<float>(80.0F), aspectRatio, 0.01F),
        };

        if (void *const data = m_Device.GetHandle().mapMemory(m_UniformBuffers[currentImage].Memory, 0, sizeof(ubo)))
        {
            std::memcpy(data, &ubo, sizeof(ubo));
            m_Device.GetHandle().unmapMemory(m_UniformBuffers[currentImage].Memory);
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

        m_Device.GetHandle().waitIdle();

        CleanupSwapChain();

        CreateSwapChain();
        CreateImageViews();
        CreateDepthResources();
        CreateFramebuffers();
    }

    void CleanupSwapChain()
    {
        m_Device.GetHandle().destroy(m_DepthImageView);
        m_DepthImage.Destroy(m_Device.GetHandle());

        for (auto framebuffer : m_SwapChainFramebuffers)
        {
            m_Device.GetHandle().destroy(framebuffer);
        }
        m_SwapChainFramebuffers.clear();

        for (auto imageView : m_SwapChainImageViews)
        {
            m_Device.GetHandle().destroy(imageView);
        }
        m_SwapChainImageViews.clear();

        m_Device.GetHandle().destroy(m_OldSwapChain);
    }

    void Cleanup()
    {
        CleanupSwapChain();
        m_Device.GetHandle().destroy(m_SwapChain);

        m_Device.GetHandle().destroy(m_TextureSampler);

        m_DragonModel.Destroy(m_Device.GetHandle());

        m_Device.GetHandle().destroy(m_MissingTextureImageView);
        m_MissingTextureImage.Destroy(m_Device.GetHandle());

        for (std::size_t i = 0; i < MaxFramesInFlight; ++i)
        {
            m_UniformBuffers[i].Destroy(m_Device.GetHandle());
        }

        m_Device.GetHandle().destroy(m_DescriptorPool);
        m_Device.GetHandle().destroy(m_SamplerDescriptorSetLayout);
        m_Device.GetHandle().destroy(m_UboDescriptorSetLayout);

        m_Device.GetHandle().destroy(m_RenderPass);

        m_ArenaVertexBuffer.Destroy(m_Device.GetHandle());
        m_ArenaIndexBuffer.Destroy(m_Device.GetHandle());

        for (std::uint32_t i = 0; i < MaxFramesInFlight; ++i)
        {
            m_Device.GetHandle().destroy(m_ImageAvailableSemaphores[i]);
            m_Device.GetHandle().destroy(m_RenderFinishedSemaphores[i]);
            m_Device.GetHandle().destroy(m_InFlightFences[i]);
        }

        m_Device.GetHandle().destroy(m_CommandPool);

        m_Device.GetHandle().destroy(m_GraphicsPipeline);
        m_Device.GetHandle().destroy(m_PipelineLayout);

        m_Device.Destroy();

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

    static void CursorPositionCallback(GLFWwindow *window, double xPos, double yPos)
    {
        auto *app = static_cast<::Application *>(glfwGetWindowUserPointer(window));
        app->m_MousePos = {
            .XPos = xPos,
            .YPos = yPos,
        };
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

    static constexpr char DefaultTitle[] = "Vulkan Renderer";
    static constexpr std::size_t MaxTitleLength = 256;
    static_assert(std::size(DefaultTitle) <= MaxTitleLength);

    static constexpr std::size_t FrameTimeSampleCount = 64;

    // TODO: This should be determined at runtime.
    static constexpr std::size_t TextureCount = 2;

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

    Device m_Device;

    vk::Format m_SwapChainImageFormat = vk::Format::eUndefined;
    vk::Extent2D m_SwapChainExtent;
    vk::SwapchainKHR m_SwapChain;
    vk::SwapchainKHR m_OldSwapChain;
    std::vector<vk::Image> m_SwapChainImages;
    std::vector<vk::ImageView> m_SwapChainImageViews;

    vk::RenderPass m_RenderPass;

    vk::DescriptorSetLayout m_UboDescriptorSetLayout;
    vk::DescriptorSetLayout m_SamplerDescriptorSetLayout;
    vk::PipelineLayout m_PipelineLayout;
    vk::Pipeline m_GraphicsPipeline;

    std::vector<vk::Framebuffer> m_SwapChainFramebuffers;

    vk::CommandPool m_CommandPool;

    Image m_DepthImage;
    vk::ImageView m_DepthImageView;

    vk::Sampler m_TextureSampler;

    vk::DescriptorPool m_DescriptorPool;

    Buffer m_ArenaVertexBuffer;
    Buffer m_ArenaIndexBuffer;
    Image m_MissingTextureImage;
    vk::ImageView m_MissingTextureImageView;
    vk::DescriptorSet m_MissingTextureDescriptorSet;

    Model m_DragonModel;

    std::vector<Buffer> m_UniformBuffers;
    std::vector<vk::DescriptorSet> m_UboDescriptorSets;

    std::uint32_t m_CurrentFrameIndex = 0;
    std::vector<vk::CommandBuffer> m_CommandBuffers;
    std::vector<vk::Semaphore> m_ImageAvailableSemaphores;
    std::vector<vk::Semaphore> m_RenderFinishedSemaphores;
    std::vector<vk::Fence> m_InFlightFences;

    std::chrono::steady_clock::time_point m_LastTick;
    std::chrono::steady_clock::time_point m_CurrentTick;
    float m_FrameTimeSum = 0.0F;
    float m_SmoothedFrameTime = 0.0F;
    std::size_t m_FrameTimeSampleIndex = 0;
    std::array<float, FrameTimeSampleCount> m_FrameTimeSamples = { };

    struct Mouse
    {
        double XPos;
        double YPos;
    };

    Mouse m_LastMousePos = { };
    Mouse m_MousePos = { };

    Camera m_Camera;

    std::array<char, MaxTitleLength> m_Title = { };
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
                std::printf("\"%s\" was specified, but no value was given\n", ResourcesPathFlag.data());
                std::fprintf(stderr, "Usage: %s%c{ABSOLUTE_PATH_TO_DIR}\n", ResourcesPathFlag.data(), FlagDelimiter);
                return EXIT_FAILURE;
            }
        }
        else
        {
            std::fprintf(stderr, "Unrecognised argument: %s - ignoring", arg.data());
        }
    }

    try
    {
        app.Run();
    }
    catch (const std::exception &e)
    {
        spdlog::error("Fatal error: {}", e.what());
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
