#include "Device.hpp"
#include "Utility.hpp"

#include <fmt/format.h>
#include <GLFW/glfw3.h>
#include <glm/gtc/matrix_transform.hpp>
#include <tiny_obj_loader.h>
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
#include <numeric>
#include <optional>
#include <set>
#include <stdexcept>

VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

constexpr std::array<const char *, 1> DeviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME,
};

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

struct Model
{
    struct
    {
        vk::Buffer Buffer;
        vk::DeviceMemory Memory;
    } Vertices;

    struct
    {
        std::uint32_t Count = 0;
        vk::Buffer Buffer;
        vk::DeviceMemory Memory;
    } Indices;

    struct
    {
        vk::Image Image;
        vk::DeviceMemory Memory;
        vk::ImageView View;
    } Texture;

    void Init(vk::Device device, const char *meshPath, const char *texturePath)
    {

    }

    void Destroy(vk::Device device)
    {

    }
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
        CreateResources();
        CreateTextureSampler();
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

        return m_Device.GetHandle().createImageView(createInfo);
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

        buffer = m_Device.GetHandle().createBuffer(bufferInfo);

        const vk::MemoryRequirements memoryRequirements = m_Device.GetHandle().getBufferMemoryRequirements(buffer);

        const std::uint32_t memoryTypeIndex = FindMemoryType(memoryRequirements.memoryTypeBits, properties);

        const vk::MemoryAllocateInfo allocateInfo = {
            .allocationSize  = memoryRequirements.size,
            .memoryTypeIndex = memoryTypeIndex,
        };

        bufferMemory = m_Device.GetHandle().allocateMemory(allocateInfo);
        m_Device.GetHandle().bindBufferMemory(buffer, bufferMemory, 0);
    }

    void CopyBuffer(vk::CommandBuffer commandBuffer, vk::Buffer source, vk::Buffer destination, vk::DeviceSize size)
    {
        const vk::BufferCopy copyRegion = {
            .srcOffset = 0,
            .dstOffset = 0,
            .size      = size,
        };

        commandBuffer.copyBuffer(source, destination, copyRegion);
    }

    void CopyBufferToImage(
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

        image = m_Device.GetHandle().createImage(createInfo);

        const vk::MemoryRequirements memoryRequirements = m_Device.GetHandle().getImageMemoryRequirements(image);

        const vk::MemoryAllocateInfo allocInfo = {
            .allocationSize = memoryRequirements.size,
            .memoryTypeIndex = FindMemoryType(memoryRequirements.memoryTypeBits, properties),
        };

        imageMemory = m_Device.GetHandle().allocateMemory(allocInfo);
        m_Device.GetHandle().bindImageMemory(image, imageMemory, 0);
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

    void CreateResources()
    {
        const vk::CommandBuffer loadResourcesCommands = BeginOneTimeCommands();

        // Transfer arena vertices to GPU memory.

        constexpr vk::DeviceSize arenaVerticesSize =
            sizeof(decltype(ArenaVertices)::value_type) * ArenaVertices.size();

        vk::Buffer arenaVerticesStagingBuffer;
        vk::DeviceMemory arenaVerticesStagingBufferMemory;
        CreateBuffer(
            arenaVerticesSize,
            vk::BufferUsageFlagBits::eTransferSrc,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
            arenaVerticesStagingBuffer,
            arenaVerticesStagingBufferMemory
        );

        if (void *data = m_Device.GetHandle().mapMemory(arenaVerticesStagingBufferMemory, 0, arenaVerticesSize))
        {
            std::memcpy(data, ArenaVertices.data(), arenaVerticesSize);
            m_Device.GetHandle().unmapMemory(arenaVerticesStagingBufferMemory);
        }
        else
        {
            throw std::runtime_error("Failed to map staging buffer memory for vertices");
        }

        CreateBuffer(
            arenaVerticesSize,
            vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eVertexBuffer,
            vk::MemoryPropertyFlagBits::eDeviceLocal,
            m_ArenaVertexBuffer,
            m_ArenaVertexBufferMemory
        );

        CopyBuffer(loadResourcesCommands, arenaVerticesStagingBuffer, m_ArenaVertexBuffer, arenaVerticesSize);

        // Transfer arena indices to GPU memory.

        constexpr vk::DeviceSize arenaIndicesSize =
            sizeof(decltype(ArenaIndices)::value_type) * ArenaIndices.size();

        vk::Buffer arenaIndicesStagingBuffer;
        vk::DeviceMemory arenaIndicesStagingBufferMemory;
        CreateBuffer(
            arenaIndicesSize,
            vk::BufferUsageFlagBits::eTransferSrc,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
            arenaIndicesStagingBuffer,
            arenaIndicesStagingBufferMemory
        );

        if (void *data = m_Device.GetHandle().mapMemory(arenaIndicesStagingBufferMemory, 0, arenaIndicesSize))
        {
            std::memcpy(data, ArenaIndices.data(), arenaIndicesSize);
            m_Device.GetHandle().unmapMemory(arenaIndicesStagingBufferMemory);
        }
        else
        {
            throw std::runtime_error("Failed to map staging buffer memory for indices");
        }

        CreateBuffer(
            arenaIndicesSize,
            vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eIndexBuffer,
            vk::MemoryPropertyFlagBits::eDeviceLocal,
            m_ArenaIndexBuffer,
            m_ArenaIndexBufferMemory
        );

        CopyBuffer(loadResourcesCommands, arenaIndicesStagingBuffer, m_ArenaIndexBuffer, arenaIndicesSize);

        // Load dragon mesh.

        tinyobj::ObjReader reader;
        if (!reader.ParseFromFile(GetResourcePath("Meshes/StanfordDragon.obj")))
        {
            throw std::runtime_error("Failed to load Stanford dragon mesh");
        }

        if (!reader.Warning().empty())
        {
            std::cout << "TinyObjReader Warning: " << reader.Warning();
        }

        const tinyobj::attrib_t &attrib = reader.GetAttrib();
        const tinyobj::shape_t &shape = reader.GetShapes()[0];

        std::size_t indexOffset = 0;
        for (std::size_t faceIndex = 0; faceIndex < shape.mesh.num_face_vertices.size(); ++faceIndex)
        {
            const std::size_t faceVertexCount = shape.mesh.num_face_vertices[faceIndex];

            for (std::size_t vertexIndex = 0; vertexIndex < faceVertexCount; ++vertexIndex)
            {
                constexpr std::size_t vertexDimensionCount = 3;
                const tinyobj::index_t index = shape.mesh.indices[vertexIndex + indexOffset];
                const tinyobj::real_t xPos = attrib.vertices[vertexDimensionCount * (std::size_t)index.vertex_index + 0];
                const tinyobj::real_t yPos = attrib.vertices[vertexDimensionCount * (std::size_t)index.vertex_index + 1];
                const tinyobj::real_t zPos = attrib.vertices[vertexDimensionCount * (std::size_t)index.vertex_index + 2];

                assert("Meshes should contain texture coordinate info" && index.texcoord_index >= 0);
                constexpr std::size_t textureDimensionCount = 2;
                const tinyobj::real_t u = attrib.texcoords[textureDimensionCount * (std::size_t)index.texcoord_index + 0];
                const tinyobj::real_t v = attrib.texcoords[textureDimensionCount * (std::size_t)index.texcoord_index + 1];

                const Vertex vertex = {
                    .Position = { xPos, yPos, zPos },
                    .TexCoord = { u, v },
                };

                m_DragonVertices.push_back(vertex);
            }

            indexOffset += faceVertexCount;
        }

        m_DragonIndices.resize(m_DragonVertices.size());
        std::iota(m_DragonIndices.begin(), m_DragonIndices.end(), 0);

        // Copy dragon vertices to GPU memory.

        const vk::DeviceSize dragonVerticesSize =
            sizeof(decltype(m_DragonVertices)::value_type) * m_DragonVertices.size();

        vk::Buffer dragonVerticesStagingBuffer;
        vk::DeviceMemory dragonVerticesStagingBufferMemory;
        CreateBuffer(
            dragonVerticesSize,
            vk::BufferUsageFlagBits::eTransferSrc,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
            dragonVerticesStagingBuffer,
            dragonVerticesStagingBufferMemory
        );

        if (void *data = m_Device.GetHandle().mapMemory(dragonVerticesStagingBufferMemory, 0, dragonVerticesSize))
        {
            std::memcpy(data, m_DragonVertices.data(), dragonVerticesSize);
            m_Device.GetHandle().unmapMemory(dragonVerticesStagingBufferMemory);
        }
        else
        {
            throw std::runtime_error("Failed to map staging buffer memory for vertices");
        }

        CreateBuffer(
            dragonVerticesSize,
            vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eVertexBuffer,
            vk::MemoryPropertyFlagBits::eDeviceLocal,
            m_DragonVertexBuffer,
            m_DragonVertexBufferMemory
        );

        CopyBuffer(loadResourcesCommands, dragonVerticesStagingBuffer, m_DragonVertexBuffer, dragonVerticesSize);

        // Copy dragon indices to GPU memory.

        const vk::DeviceSize dragonIndicesSize =
            sizeof(decltype(m_DragonIndices)::value_type) * m_DragonIndices.size();

        vk::Buffer dragonIndicesStagingBuffer;
        vk::DeviceMemory dragonIndicesStagingBufferMemory;
        CreateBuffer(
            dragonIndicesSize,
            vk::BufferUsageFlagBits::eTransferSrc,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
            dragonIndicesStagingBuffer,
            dragonIndicesStagingBufferMemory
        );

        if (void *data = m_Device.GetHandle().mapMemory(dragonIndicesStagingBufferMemory, 0, dragonIndicesSize))
        {
            std::memcpy(data, m_DragonIndices.data(), dragonIndicesSize);
            m_Device.GetHandle().unmapMemory(dragonIndicesStagingBufferMemory);
        }
        else
        {
            throw std::runtime_error("Failed to map staging buffer memory for vertices");
        }

        CreateBuffer(
            dragonIndicesSize,
            vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eIndexBuffer,
            vk::MemoryPropertyFlagBits::eDeviceLocal,
            m_DragonIndexBuffer,
            m_DragonIndexBufferMemory
        );

        CopyBuffer(loadResourcesCommands, dragonIndicesStagingBuffer, m_DragonIndexBuffer, dragonIndicesSize);

        // Copy textures to GPU memory.

        const std::vector<std::uint8_t> missingTextureFileBytes =
            ReadFile(GetResourcePath("Textures/Missing_Raw.tga").c_str());

        const Tga::File missingTexture = Tga::File::CreateFrom(missingTextureFileBytes);

        vk::Buffer missingTextureStagingBuffer;
        vk::DeviceMemory missingTextureStagingBufferMemory;
        CreateBuffer(
            missingTexture.GetSize(),
            vk::BufferUsageFlagBits::eTransferSrc,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
            missingTextureStagingBuffer,
            missingTextureStagingBufferMemory
        );

        if (void *const data = m_Device.GetHandle().mapMemory(missingTextureStagingBufferMemory, 0, missingTexture.GetSize()))
        {
            std::memcpy(data, missingTexture.Pixels.data(), missingTexture.Pixels.size());
            m_Device.GetHandle().unmapMemory(missingTextureStagingBufferMemory);
        }
        else
        {
            throw std::runtime_error("Failed to map texture memory");
        }

        CreateImage(
            missingTexture.Width,
            missingTexture.Height,
            vk::Format::eB8G8R8A8Srgb,
            vk::ImageTiling::eOptimal,
            vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled,
            vk::MemoryPropertyFlagBits::eDeviceLocal,
            m_MissingTextureImage,
            m_MissingTextureImageMemory
        );

        TransitionImageLayout(
            loadResourcesCommands,
            m_MissingTextureImage,
            vk::ImageLayout::eUndefined,
            vk::ImageLayout::eTransferDstOptimal
        );

        CopyBufferToImage(
            loadResourcesCommands,
            missingTextureStagingBuffer,
            m_MissingTextureImage,
            missingTexture.Width,
            missingTexture.Height
        );

        TransitionImageLayout(
            loadResourcesCommands,
            m_MissingTextureImage,
            vk::ImageLayout::eTransferDstOptimal,
            vk::ImageLayout::eShaderReadOnlyOptimal
        );

        const std::vector<std::uint8_t> dragonTextureFileBytes =
            ReadFile(GetResourcePath("Textures/StanfordDragon_Albedo_Raw.tga").c_str());

        const Tga::File dragonTexture = Tga::File::CreateFrom(dragonTextureFileBytes);

        vk::Buffer dragonTextureStagingBuffer;
        vk::DeviceMemory dragonTextureStagingBufferMemory;
        CreateBuffer(
            dragonTexture.GetSize(),
            vk::BufferUsageFlagBits::eTransferSrc,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
            dragonTextureStagingBuffer,
            dragonTextureStagingBufferMemory
        );

        if (void *const data = m_Device.GetHandle().mapMemory(dragonTextureStagingBufferMemory, 0, dragonTexture.GetSize()))
        {
            std::memcpy(data, dragonTexture.Pixels.data(), dragonTexture.Pixels.size());
            m_Device.GetHandle().unmapMemory(dragonTextureStagingBufferMemory);
        }
        else
        {
            throw std::runtime_error("Failed to map texture memory");
        }

        CreateImage(
            dragonTexture.Width,
            dragonTexture.Height,
            vk::Format::eB8G8R8A8Srgb,
            vk::ImageTiling::eOptimal,
            vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled,
            vk::MemoryPropertyFlagBits::eDeviceLocal,
            m_DragonTextureImage,
            m_DragonTextureImageMemory
        );

        TransitionImageLayout(
            loadResourcesCommands,
            m_DragonTextureImage,
            vk::ImageLayout::eUndefined,
            vk::ImageLayout::eTransferDstOptimal
        );

        CopyBufferToImage(
            loadResourcesCommands,
            dragonTextureStagingBuffer,
            m_DragonTextureImage,
            dragonTexture.Width,
            dragonTexture.Height
        );

        TransitionImageLayout(
            loadResourcesCommands,
            m_DragonTextureImage,
            vk::ImageLayout::eTransferDstOptimal,
            vk::ImageLayout::eShaderReadOnlyOptimal
        );

        EndOneTimeCommands(loadResourcesCommands);

        m_Device.GetHandle().destroy(dragonTextureStagingBuffer);
        m_Device.GetHandle().free(dragonTextureStagingBufferMemory);

        m_Device.GetHandle().destroy(missingTextureStagingBuffer);
        m_Device.GetHandle().free(missingTextureStagingBufferMemory);

        m_Device.GetHandle().destroy(dragonIndicesStagingBuffer);
        m_Device.GetHandle().free(dragonIndicesStagingBufferMemory);

        m_Device.GetHandle().destroy(dragonVerticesStagingBuffer);
        m_Device.GetHandle().free(dragonVerticesStagingBufferMemory);

        m_Device.GetHandle().destroy(arenaIndicesStagingBuffer);
        m_Device.GetHandle().free(arenaIndicesStagingBufferMemory);

        m_Device.GetHandle().destroy(arenaVerticesStagingBuffer);
        m_Device.GetHandle().free(arenaVerticesStagingBufferMemory);

        m_MissingTextureImageView = CreateImageView(
            m_MissingTextureImage,
            vk::Format::eB8G8R8A8Srgb,
            vk::ImageAspectFlagBits::eColor
        );

        m_DragonTextureImageView = CreateImageView(
            m_DragonTextureImage,
            vk::Format::eB8G8R8A8Srgb,
            vk::ImageAspectFlagBits::eColor
        );
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
                .buffer = m_UniformBuffers[i],
                .offset = 0,
                .range  = sizeof(::UniformBufferObject),
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

        const std::vector<vk::DescriptorSetLayout> textureLayouts(TextureCount, m_SamplerDescriptorSetLayout);

        const vk::DescriptorSetAllocateInfo textureAllocInfo = {
            .descriptorPool     = m_DescriptorPool,
            .descriptorSetCount = TextureCount,
            .pSetLayouts        = textureLayouts.data(),
        };

        m_TextureDescriptorSets = m_Device.GetHandle().allocateDescriptorSets(textureAllocInfo);

        const vk::DescriptorImageInfo missingTextureImageInfo = {
            .sampler     = m_TextureSampler,
            .imageView   = m_MissingTextureImageView,
            .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
        };

        const vk::WriteDescriptorSet missingTextureDescriptorWrite = {
            .dstSet          = m_TextureDescriptorSets[MissingTexture],
            .dstBinding      = 0,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType  = vk::DescriptorType::eCombinedImageSampler,
            .pImageInfo      = &missingTextureImageInfo,
        };

        m_Device.GetHandle().updateDescriptorSets({ missingTextureDescriptorWrite }, { });

        const vk::DescriptorImageInfo dragonTextureImageInfo = {
            .sampler     = m_TextureSampler,
            .imageView   = m_DragonTextureImageView,
            .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
        };

        const vk::WriteDescriptorSet dragonTextureDescriptorWrite = {
            .dstSet          = m_TextureDescriptorSets[DragonTexture],
            .dstBinding      = 0,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType  = vk::DescriptorType::eCombinedImageSampler,
            .pImageInfo      = &dragonTextureImageInfo,
        };

        m_Device.GetHandle().updateDescriptorSets({ dragonTextureDescriptorWrite }, { });
    }

    std::uint32_t FindMemoryType(std::uint32_t typeFilter, vk::MemoryPropertyFlags properties)
    {
        constexpr auto filterBitCount = std::numeric_limits<decltype(typeFilter)>::digits;
        const std::bitset<filterBitCount> eligibleTypes(typeFilter);

        const auto isTypePresent = [&eligibleTypes](std::uint32_t index)
        {
            return eligibleTypes.test(index);
        };

        const vk::PhysicalDeviceMemoryProperties &memoryProperties = m_Device.GetMemoryProperties();
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
                { m_TextureDescriptorSets[MissingTexture] },
                { }
            );
            commandBuffer.bindVertexBuffers(0, { m_ArenaVertexBuffer }, { 0 });
            commandBuffer.bindIndexBuffer(m_ArenaIndexBuffer, 0, vk::IndexType::eUint16);
            commandBuffer.drawIndexed(static_cast<std::uint32_t>(ArenaIndices.size()), 1, 0, 0, 0);

            commandBuffer.bindDescriptorSets(
                vk::PipelineBindPoint::eGraphics,
                m_PipelineLayout,
                1,
                { m_TextureDescriptorSets[DragonTexture] },
                { }
            );
            commandBuffer.bindVertexBuffers(0, { m_DragonVertexBuffer }, { 0 });
            commandBuffer.bindIndexBuffer(m_DragonIndexBuffer, 0, vk::IndexType::eUint32);
            commandBuffer.drawIndexed(static_cast<std::uint32_t>(m_DragonIndices.size()), 1, 0, 0, 0);
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

            UpdateState(deltaTime);
            DrawFrame();
            glfwPollEvents();

            m_LastTick = m_CurrentTick;
            m_CurrentTick = std::chrono::steady_clock::now();
        }

        m_Device.GetHandle().waitIdle();
    }

    void UpdateState(float deltaTime)
    {
        const auto deltaX = static_cast<float>(m_MousePos.XPos - m_LastMousePos.XPos);
        const auto deltaY = static_cast<float>(m_MousePos.YPos - m_LastMousePos.YPos);

        UpdateCamera(deltaTime, deltaX, deltaY);
        UpdateTitle();

        m_LastMousePos = m_MousePos;
    }

    void UpdateCamera(float deltaTime, float deltaX, float deltaY)
    {
        if (glfwGetKey(m_Window, GLFW_KEY_R) == GLFW_PRESS)
        {
            m_Camera = Camera();
            return;
        }

        constexpr float lookSensitivity = 0.005F;

        constexpr float pitchLimit = glm::half_pi<float>() - 0.01F;
        const float deltaYaw = lookSensitivity * -deltaX;
        const float deltaPitch = lookSensitivity * deltaY;

        m_Camera.Yaw = glm::mod(m_Camera.Yaw + deltaYaw, glm::two_pi<float>());
        m_Camera.Pitch = glm::clamp(m_Camera.Pitch + deltaPitch, -pitchLimit, pitchLimit);

        const glm::vec3 lookDirection = {
            glm::cos(m_Camera.Yaw) * glm::cos(m_Camera.Pitch),
            glm::sin(m_Camera.Pitch),
            glm::sin(m_Camera.Yaw) * glm::cos(m_Camera.Pitch),
        };

        m_Camera.Forward = glm::normalize(lookDirection);
        m_Camera.Right = glm::normalize(glm::cross(m_Camera.Forward, glm::vec3(0.0F, -1.0F, 0.0F)));
        m_Camera.Up = glm::normalize(glm::cross(m_Camera.Right, m_Camera.Forward));

        struct
        {
            bool Forward = false;
            bool Backward = false;
            bool Left = false;
            bool Right = false;

            void Normalize()
            {
                if (Forward && Backward)
                {
                    Forward = false;
                    Backward = false;
                }

                if (Left && Right)
                {
                    Left = false;
                    Right = false;
                }
            }

            bool Any() const
            {
                return Forward || Backward || Left || Right;
            }
        } movement;

        glm::vec3 moveDirection = { };
        if (glfwGetKey(m_Window, GLFW_KEY_W) == GLFW_PRESS)
        {
            movement.Forward = true;
            moveDirection += m_Camera.Forward;
        }
        if (glfwGetKey(m_Window, GLFW_KEY_A) == GLFW_PRESS)
        {
            movement.Left = true;
            moveDirection -= m_Camera.Right;
        }
        if (glfwGetKey(m_Window, GLFW_KEY_S) == GLFW_PRESS)
        {
            movement.Backward = true;
            moveDirection -= m_Camera.Forward;
        }
        if (glfwGetKey(m_Window, GLFW_KEY_D) == GLFW_PRESS)
        {
            movement.Right = true;
            moveDirection += m_Camera.Right;
        }

        movement.Normalize();

        if (movement.Any())
        {
            constexpr float moveSpeed = 5.0F;
            constexpr float shiftMoveSpeed = 10.0F;
            constexpr float altMoveSpeed = 2.0F;

            const float moveAmount = glfwGetKey(m_Window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS
                ? deltaTime * shiftMoveSpeed
                : glfwGetKey(m_Window, GLFW_KEY_LEFT_ALT) == GLFW_PRESS
                    ? deltaTime * altMoveSpeed
                    : deltaTime * moveSpeed;

            m_Camera.Position += moveAmount * glm::normalize(moveDirection);
        }
    }

    void UpdateTitle()
    {
        const auto result = fmt::format_to_n(
            m_Title.begin(),
            m_Title.size(),
            "{} | Frame Time: {:>7.3f} ms | Position: ({:.2f}, {:.2f}, {:.2f}) | Heading: ({:.2f}, {:.2f}, {:.2f})",
            DefaultTitle,
            m_SmoothedFrameTime * 1000.0F,
            m_Camera.Position.x,
            m_Camera.Position.y,
            m_Camera.Position.z,
            m_Camera.Forward.x,
            m_Camera.Forward.y,
            m_Camera.Forward.z
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

    [[nodiscard]]
    glm::mat4 GetViewMatrix() const
    {
        return glm::lookAt(m_Camera.Position, m_Camera.Position + m_Camera.Forward, m_Camera.Up);
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

    void TransitionImageLayout(
        vk::CommandBuffer commandBuffer,
        vk::Image image,
        vk::ImageLayout oldLayout,
        vk::ImageLayout newLayout
    )
    {
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
    }

    void UpdateUniformBuffer(std::uint32_t currentImage) const
    {
        const float aspectRatio =
            static_cast<float>(m_SwapChainExtent.width) / static_cast<float>(m_SwapChainExtent.height);

        const UniformBufferObject ubo = {
            .Model = glm::mat4(1.0F),
            .View  = GetViewMatrix(),
            .Proj  = vkm::perspective(glm::radians<float>(80.0F), aspectRatio, 0.01F),
        };

        if (void *const data = m_Device.GetHandle().mapMemory(m_UniformBuffersMemory[currentImage], 0, sizeof(ubo)))
        {
            std::memcpy(data, &ubo, sizeof(ubo));
            m_Device.GetHandle().unmapMemory(m_UniformBuffersMemory[currentImage]);
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
        m_Device.GetHandle().destroy(m_DepthImage);
        m_Device.GetHandle().free(m_DepthImageMemory);

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

        m_Device.GetHandle().destroy(m_DragonTextureImageView);
        m_Device.GetHandle().destroy(m_DragonTextureImage);
        m_Device.GetHandle().free(m_DragonTextureImageMemory);

        m_Device.GetHandle().destroy(m_MissingTextureImageView);
        m_Device.GetHandle().destroy(m_MissingTextureImage);
        m_Device.GetHandle().free(m_MissingTextureImageMemory);

        for (std::size_t i = 0; i < MaxFramesInFlight; ++i)
        {
            m_Device.GetHandle().destroy(m_UniformBuffers[i]);
            m_Device.GetHandle().free(m_UniformBuffersMemory[i]);
        }

        m_Device.GetHandle().destroy(m_DescriptorPool);
        m_Device.GetHandle().destroy(m_SamplerDescriptorSetLayout);
        m_Device.GetHandle().destroy(m_UboDescriptorSetLayout);

        m_Device.GetHandle().destroy(m_RenderPass);

        m_Device.GetHandle().destroy(m_DragonVertexBuffer);
        m_Device.GetHandle().free(m_DragonVertexBufferMemory);
        m_Device.GetHandle().destroy(m_DragonIndexBuffer);
        m_Device.GetHandle().free(m_DragonIndexBufferMemory);

        m_Device.GetHandle().destroy(m_ArenaVertexBuffer);
        m_Device.GetHandle().free(m_ArenaVertexBufferMemory);
        m_Device.GetHandle().destroy(m_ArenaIndexBuffer);
        m_Device.GetHandle().free(m_ArenaIndexBufferMemory);

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

    static constexpr std::size_t FrameTimeSampleCount = 64;

    enum : std::size_t
    {
        MissingTexture,
        DragonTexture,
        TextureCount,
    };

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

    vk::Image m_DepthImage;
    vk::DeviceMemory m_DepthImageMemory;
    vk::ImageView m_DepthImageView;

    vk::Image m_MissingTextureImage;
    vk::DeviceMemory m_MissingTextureImageMemory;
    vk::ImageView m_MissingTextureImageView;

    vk::Image m_DragonTextureImage;
    vk::DeviceMemory m_DragonTextureImageMemory;
    vk::ImageView m_DragonTextureImageView;

    std::vector<Vertex> m_DragonVertices;
    std::vector<std::uint32_t> m_DragonIndices;

    vk::Sampler m_TextureSampler;

    vk::Buffer m_ArenaVertexBuffer;
    vk::DeviceMemory m_ArenaVertexBufferMemory;
    vk::Buffer m_ArenaIndexBuffer;
    vk::DeviceMemory m_ArenaIndexBufferMemory;

    vk::Buffer m_DragonVertexBuffer;
    vk::DeviceMemory m_DragonVertexBufferMemory;
    vk::Buffer m_DragonIndexBuffer;
    vk::DeviceMemory m_DragonIndexBufferMemory;

    std::vector<vk::Buffer> m_UniformBuffers;
    std::vector<vk::DeviceMemory> m_UniformBuffersMemory;

    vk::DescriptorPool m_DescriptorPool;
    std::vector<vk::DescriptorSet> m_UboDescriptorSets;
    std::vector<vk::DescriptorSet> m_TextureDescriptorSets;

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

    struct Camera
    {
        glm::vec3 Forward = { 0.0F, 0.0F, 1.0F };
        glm::vec3 Right = { 1.0F, 0.0F, 0.0F };
        glm::vec3 Up = { 0.0F, -1.0F, 0.0F };

        glm::vec3 Position = { 0.0F, 0.0F, -1.0F };

        float Yaw   = glm::half_pi<float>();
        float Pitch = 0.0F;
    };

    Camera m_Camera;

    std::array<char, MaxTitleLength> m_Title;
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
