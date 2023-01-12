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

constexpr const char *VulkanPortabilitySubsetExtensionName = "VK_KHR_portability_subset";

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

bool IsValidationLayersSupported(std::span<const vk::LayerProperties> instanceLayers)
{
    return std::ranges::includes(
        instanceLayers,
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

constexpr auto ConsoleDebugMessengerInfo = vk::DebugUtilsMessengerCreateInfoEXT()
    .setMessageSeverity(
        vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose
        | vk::DebugUtilsMessageSeverityFlagBitsEXT::eInfo
        | vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning
        | vk::DebugUtilsMessageSeverityFlagBitsEXT::eError
    )
    .setMessageType(
        vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral
        | vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation
        | vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance
    )
    .setPfnUserCallback(VulkanDebugMessengerCallback);

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
        const Device::InitInfo deviceInfo = SelectPhysicalDevice(m_Instance);
        m_Device.Init(deviceInfo);
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

        std::vector<vk::LayerProperties> instanceLayers = vk::enumerateInstanceLayerProperties();
        std::ranges::sort(instanceLayers, { }, &vk::LayerProperties::layerName);

        if constexpr (IsDebugMode)
        {
            if (!IsValidationLayersSupported(instanceLayers))
            {
                throw std::runtime_error("Validation layers requested but not supported");
            }
        }

        const auto appInfo = vk::ApplicationInfo()
            .setApiVersion(VK_API_VERSION_1_3);

        vk::StructureChain<vk::InstanceCreateInfo, vk::DebugUtilsMessengerCreateInfoEXT> chain;
        auto &createInfo = chain.get<vk::InstanceCreateInfo>();
        auto &debugCreateInfo = chain.get<vk::DebugUtilsMessengerCreateInfoEXT>();

        vk::InstanceCreateFlags instanceFlags = { };
        std::vector<const char *> requiredExtensions = GetRequiredExtensions();
        std::vector<vk::ExtensionProperties> instanceExtensions = vk::enumerateInstanceExtensionProperties();
        std::ranges::sort(instanceExtensions, { }, &vk::ExtensionProperties::extensionName);
        if (IsPortabilityEnumerationRequired(instanceExtensions))
        {
            instanceFlags |= vk::InstanceCreateFlagBits::eEnumeratePortabilityKHR;
            requiredExtensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
            requiredExtensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
        }

        createInfo = vk::InstanceCreateInfo()
            .setFlags(instanceFlags)
            .setPApplicationInfo(&appInfo)
            .setPEnabledExtensionNames(requiredExtensions);

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

    static bool IsPortabilityEnumerationRequired(std::span<const vk::ExtensionProperties> instanceExtensions)
    {
        return std::ranges::find_if(
            instanceExtensions,
            [](const vk::ExtensionProperties &properties)
            {
                return properties.extensionName == std::string_view(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
            }
        ) != instanceExtensions.end();
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
    Device::InitInfo SelectPhysicalDevice(vk::Instance instance)
    {
        const std::vector<vk::PhysicalDevice> devices = instance.enumeratePhysicalDevices();

        if (devices.empty())
        {
            throw std::runtime_error("Failed to find a GPU with Vulkan support");
        }

        for (const vk::PhysicalDevice &device : devices)
        {
            const Device::QueueFamilyIndices indices = Device::QueueFamilyIndices::Find(device, m_Surface);
            const SwapChainSupportDetails details = QuerySwapChainSupport(device, m_Surface);

            auto deviceExtensions = device.enumerateDeviceExtensionProperties();
            std::ranges::sort(deviceExtensions, { }, &vk::ExtensionProperties::extensionName);

            if (IsRequiredExtensionsSupported(deviceExtensions)
                && indices.IsComplete()
                && (!details.Formats.empty() && !details.PresentModes.empty()))
            {
                std::vector requiredExtensions(DeviceExtensions.begin(), DeviceExtensions.end());

                if (IsPortabilitySubsetRequired(deviceExtensions))
                {
                    requiredExtensions.push_back(VulkanPortabilitySubsetExtensionName);
                }

                return {
                    .PhysicalDevice = device,
                    .QueueFamilyIndices = indices,
                    .RequiredExtensions = std::move(requiredExtensions),
                };
            }
        }

        throw std::runtime_error("Failed to find a GPU suitable for this application");
    }

    bool IsPortabilitySubsetRequired(std::span<const vk::ExtensionProperties> deviceExtensions)
    {
        return std::ranges::find_if(
            deviceExtensions,
            [](const vk::ExtensionProperties &properties)
            {
            return properties.extensionName == std::string_view(VulkanPortabilitySubsetExtensionName);
            }
        ) != deviceExtensions.end();
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
        const SwapChainSupportDetails swapChainSupport = QuerySwapChainSupport(m_Device.GetPhysicalDevice(), m_Surface);

        const vk::SurfaceFormatKHR surfaceFormat = SelectSwapSurfaceFormat(swapChainSupport.Formats);
        const vk::PresentModeKHR presentMode = SelectSwapPresentMode(swapChainSupport.PresentModes);
        const vk::Extent2D extent = SelectSwapExtent(swapChainSupport.Capabilities);

        const std::uint32_t desiredImageCount = swapChainSupport.Capabilities.minImageCount + 1;
        const std::uint32_t maxImageCount = swapChainSupport.Capabilities.maxImageCount;

        const std::uint32_t imageCount = maxImageCount == 0
            ? desiredImageCount
            : std::min(desiredImageCount, maxImageCount);

        const std::array queueFamilyIndices = m_Device.GetQueueFamilyIndices().ToArray();

        m_OldSwapChain = m_SwapChain;
        const vk::SwapchainCreateInfoKHR createInfo = ([&]
        {
            auto result = vk::SwapchainCreateInfoKHR()
                .setSurface(m_Surface)
                .setMinImageCount(imageCount)
                .setImageFormat(surfaceFormat.format)
                .setImageColorSpace(surfaceFormat.colorSpace)
                .setImageExtent(extent)
                .setImageArrayLayers(1)
                .setImageUsage(vk::ImageUsageFlagBits::eColorAttachment)
                .setPreTransform(swapChainSupport.Capabilities.currentTransform)
                .setPresentMode(presentMode)
                .setClipped(VK_TRUE)
                .setOldSwapchain(m_OldSwapChain);

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
            std::clamp(width, minWidth, maxWidth),
            std::clamp(height, minHeight, maxHeight)
        };
    }

    void CreateImageViews()
    {
        m_SwapChainImageViews.reserve(m_SwapChainImages.size());

        for (vk::Image image : m_SwapChainImages)
        {
            const vk::ImageView imageView = m_Device.CreateImageView(
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
        auto &colorAttachment = attachments[ColorAttachmentIndex];
        auto &depthAttachment = attachments[DepthAttachmentIndex];

        colorAttachment = vk::AttachmentDescription()
            .setFormat(m_SwapChainImageFormat)
            .setSamples(vk::SampleCountFlagBits::e1)
            .setLoadOp(vk::AttachmentLoadOp::eClear)
            .setStoreOp(vk::AttachmentStoreOp::eStore)
            .setStencilLoadOp(vk::AttachmentLoadOp::eDontCare)
            .setStencilStoreOp(vk::AttachmentStoreOp::eDontCare)
            .setInitialLayout(vk::ImageLayout::eUndefined)
            .setFinalLayout(vk::ImageLayout::ePresentSrcKHR);

        depthAttachment = vk::AttachmentDescription()
            .setFormat(FindDepthFormat())
            .setSamples(vk::SampleCountFlagBits::e1)
            .setLoadOp(vk::AttachmentLoadOp::eClear)
            .setStoreOp(vk::AttachmentStoreOp::eDontCare)
            .setStencilLoadOp(vk::AttachmentLoadOp::eDontCare)
            .setStencilStoreOp(vk::AttachmentStoreOp::eDontCare)
            .setInitialLayout(vk::ImageLayout::eUndefined)
            .setFinalLayout(vk::ImageLayout::eDepthStencilAttachmentOptimal);

        const std::array colorAttachmentReferences = {
            vk::AttachmentReference()
                .setAttachment(ColorAttachmentIndex)
                .setLayout(vk::ImageLayout::eColorAttachmentOptimal),
        };

        const auto depthAttachmentReference = vk::AttachmentReference()
            .setAttachment(DepthAttachmentIndex)
            .setLayout(vk::ImageLayout::eDepthStencilAttachmentOptimal);

        enum : std::uint32_t
        {
            ColorSubpassIndex,

            SubpassCount,
        };

        std::array<vk::SubpassDescription, SubpassCount> subpassDescriptions;
        auto &colorSubpass = subpassDescriptions[ColorSubpassIndex];
        colorSubpass = vk::SubpassDescription()
            .setPipelineBindPoint(vk::PipelineBindPoint::eGraphics)
            .setColorAttachments(colorAttachmentReferences)
            .setPDepthStencilAttachment(&depthAttachmentReference);

        enum : std::uint32_t
        {
            ColorTransitionDependencyIndex,

            SubpassDependencyCount,
        };

        std::array<vk::SubpassDependency, SubpassDependencyCount> dependencies;
        auto &colorTransitionDependency = dependencies[ColorTransitionDependencyIndex];
        colorTransitionDependency = vk::SubpassDependency()
            .setSrcSubpass(VK_SUBPASS_EXTERNAL)
            .setDstSubpass(ColorSubpassIndex)
            .setSrcStageMask(
                vk::PipelineStageFlagBits::eColorAttachmentOutput
                | vk::PipelineStageFlagBits::eEarlyFragmentTests
            )
            .setDstStageMask(
                vk::PipelineStageFlagBits::eColorAttachmentOutput
                | vk::PipelineStageFlagBits::eEarlyFragmentTests
            )
            .setSrcAccessMask(vk::AccessFlagBits::eNoneKHR)
            .setDstAccessMask(
                vk::AccessFlagBits::eColorAttachmentWrite
                | vk::AccessFlagBits::eDepthStencilAttachmentWrite
            );

        const auto renderPassCreateInfo = vk::RenderPassCreateInfo()
            .setAttachments(attachments)
            .setSubpasses(subpassDescriptions)
            .setDependencies(dependencies);

        m_RenderPass = m_Device.GetHandle().createRenderPass(renderPassCreateInfo);
    }

    void CreateDescriptorSetLayout()
    {
        const std::array uboBindings = {
            vk::DescriptorSetLayoutBinding()
                .setBinding(0) // layout(set = 0, binding = 0) uniform UniformBufferObject.
                .setDescriptorType(vk::DescriptorType::eUniformBuffer)
                .setDescriptorCount(1)
                .setStageFlags(vk::ShaderStageFlagBits::eVertex),
        };

        const auto uboLayoutInfo = vk::DescriptorSetLayoutCreateInfo()
            .setBindings(uboBindings);

        m_UboDescriptorSetLayout = m_Device.GetHandle().createDescriptorSetLayout(uboLayoutInfo);

        const std::array samplerBindings = {
            vk::DescriptorSetLayoutBinding()
                .setBinding(0) // layout(set = 1, binding = 0) uniform sampler2D texSampler.
                .setDescriptorType(vk::DescriptorType::eCombinedImageSampler)
                .setDescriptorCount(1)
                .setStageFlags(vk::ShaderStageFlagBits::eFragment),
        };

        const auto samplerLayoutInfo = vk::DescriptorSetLayoutCreateInfo()
            .setBindings(samplerBindings);

        m_SamplerDescriptorSetLayout = m_Device.GetHandle().createDescriptorSetLayout(samplerLayoutInfo);
    }

    void CreateGraphicsPipeline()
    {
        const auto modelVsBytecode = ReadFile(GetResourcePath(TEXTURE_VERT_SHADER_RELATIVE_PATH).c_str());
        const auto modelFsBytecode = ReadFile(GetResourcePath(TEXTURE_FRAG_SHADER_RELATIVE_PATH).c_str());

        const vk::ShaderModule modelVsModule = CreateShaderModule(modelVsBytecode);
        const vk::ShaderModule modelFsModule = CreateShaderModule(modelFsBytecode);

        std::array shaderStages = {
            vk::PipelineShaderStageCreateInfo()
                .setStage(vk::ShaderStageFlagBits::eVertex)
                .setModule(modelVsModule)
                .setPName("main"),
            vk::PipelineShaderStageCreateInfo()
                .setStage(vk::ShaderStageFlagBits::eFragment)
                .setModule(modelFsModule)
                .setPName("main"),
        };

        const std::array bindingDescriptions = {
            Vertex::GetBindingDescription(),
        };

        const auto attributeDescriptions = Vertex::GetAttributeDescriptions();

        const auto vertexInputStateCreateInfo = vk::PipelineVertexInputStateCreateInfo()
            .setVertexBindingDescriptions(bindingDescriptions)
            .setVertexAttributeDescriptions(attributeDescriptions);

        const auto inputAssemblyStateCreateInfo = vk::PipelineInputAssemblyStateCreateInfo()
            .setTopology(vk::PrimitiveTopology::eTriangleList)
            .setPrimitiveRestartEnable(VK_FALSE);

        const auto viewportStateCreateInfo = vk::PipelineViewportStateCreateInfo()
            .setViewportCount(1)
            .setScissorCount(1);

        const auto rasterizationStateCreateInfo = vk::PipelineRasterizationStateCreateInfo()
            .setDepthClampEnable(VK_FALSE)
            .setRasterizerDiscardEnable(VK_FALSE)
            .setPolygonMode(vk::PolygonMode::eFill)
            .setCullMode(vk::CullModeFlagBits::eBack)
            .setFrontFace(vk::FrontFace::eCounterClockwise)
            .setDepthBiasEnable(VK_FALSE)
            .setLineWidth(1.0F);

        const auto multisampleStateCreateInfo = vk::PipelineMultisampleStateCreateInfo()
            .setRasterizationSamples(vk::SampleCountFlagBits::e1)
            .setSampleShadingEnable(VK_FALSE);

        const auto depthStencilStateCreateInfo = vk::PipelineDepthStencilStateCreateInfo()
            .setDepthTestEnable(VK_TRUE)
            .setDepthWriteEnable(VK_TRUE)
            .setDepthCompareOp(vk::CompareOp::eGreaterOrEqual)
            .setMinDepthBounds(0.0F)
            .setMaxDepthBounds(1.0F);

        const std::array blendAttachments = {
            vk::PipelineColorBlendAttachmentState()
                .setBlendEnable(VK_FALSE)
                .setColorWriteMask(
                    vk::ColorComponentFlagBits::eR
                    | vk::ColorComponentFlagBits::eG
                    | vk::ColorComponentFlagBits::eB
                    | vk::ColorComponentFlagBits::eA
                ),
        };

        const auto colorBlendStateCreateInfo = vk::PipelineColorBlendStateCreateInfo()
            .setLogicOpEnable(VK_FALSE)
            .setAttachments(blendAttachments);

        const std::array descriptorSetLayouts = { m_UboDescriptorSetLayout, m_SamplerDescriptorSetLayout };

        const auto pipelineLayoutCreateInfo = vk::PipelineLayoutCreateInfo()
            .setSetLayouts(descriptorSetLayouts);

        m_PipelineLayout = m_Device.GetHandle().createPipelineLayout(pipelineLayoutCreateInfo);

        const auto dynamicStateCreateInfo = vk::PipelineDynamicStateCreateInfo()
            .setDynamicStates(DynamicStates);

        const auto pipelineCreateInfo = vk::GraphicsPipelineCreateInfo()
            .setStages(shaderStages)
            .setPVertexInputState(&vertexInputStateCreateInfo)
            .setPInputAssemblyState(&inputAssemblyStateCreateInfo)
            .setPViewportState(&viewportStateCreateInfo)
            .setPRasterizationState(&rasterizationStateCreateInfo)
            .setPMultisampleState(&multisampleStateCreateInfo)
            .setPDepthStencilState(&depthStencilStateCreateInfo)
            .setPColorBlendState(&colorBlendStateCreateInfo)
            .setPDynamicState(&dynamicStateCreateInfo)
            .setLayout(m_PipelineLayout)
            .setRenderPass(m_RenderPass)
            .setSubpass(0);

        vk::Result result = { };

        std::tie(result, m_ModelPipeline) = m_Device.GetHandle().createGraphicsPipeline({ }, pipelineCreateInfo);

        if (result != vk::Result::eSuccess)
        {
            throw std::runtime_error("Failed to create model pipeline");
        }

        m_Device.GetHandle().destroy(modelVsModule);
        m_Device.GetHandle().destroy(modelFsModule);

        const auto skyboxVsBytecode = ReadFile(GetResourcePath(SKYBOX_VERT_SHADER_RELATIVE_PATH).c_str());
        const auto skyboxFsBytecode = ReadFile(GetResourcePath(SKYBOX_FRAG_SHADER_RELATIVE_PATH).c_str());

        const vk::ShaderModule skyboxVsModule = CreateShaderModule(skyboxVsBytecode);
        const vk::ShaderModule skyboxFsModule = CreateShaderModule(skyboxFsBytecode);

        shaderStages[0].module = skyboxVsModule;
        shaderStages[1].module = skyboxFsModule;

        std::tie(result, m_SkyboxPipeline) = m_Device.GetHandle().createGraphicsPipeline({ }, pipelineCreateInfo);

        if (result != vk::Result::eSuccess)
        {
            throw std::runtime_error("Failed to create skybox pipeline");
        }

        m_Device.GetHandle().destroy(skyboxVsModule);
        m_Device.GetHandle().destroy(skyboxFsModule);
    }

    void CreateFramebuffers()
    {
        m_SwapChainFramebuffers.reserve(m_SwapChainImageViews.size());

        for (vk::ImageView imageView : m_SwapChainImageViews)
        {
            const std::array attachments = {
                imageView,
                m_DepthImageView,
            };

            const auto createInfo = vk::FramebufferCreateInfo()
                .setRenderPass(m_RenderPass)
                .setAttachments(attachments)
                .setWidth(m_SwapChainExtent.width)
                .setHeight(m_SwapChainExtent.height)
                .setLayers(1);

            const vk::Framebuffer framebuffer = m_Device.GetHandle().createFramebuffer(createInfo);
            m_SwapChainFramebuffers.push_back(framebuffer);
        }
    }

    void CreateCommandPool()
    {
        const auto createInfo = vk::CommandPoolCreateInfo()
            .setFlags(vk::CommandPoolCreateFlagBits::eResetCommandBuffer)
            .setQueueFamilyIndex(m_Device.GetQueueFamilyIndices().Graphics);

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

        std::array dragonStagingBuffers = m_DragonModel.Init(
            m_Device,
            m_SamplerDescriptorSetLayout,
            m_DescriptorPool,
            m_TextureSampler,
            loadResourcesCommands
        );

        // Load skybox mesh.

        m_Skybox.Load(
            GetResourcePath("Meshes/Skybox.obj").c_str(),
            nullptr
        );

        std::array skyboxStagingBuffers = m_Skybox.Init(
            m_Device,
            nullptr,
            nullptr,
            nullptr,
            loadResourcesCommands
        );

        // Load skybox texture.

        const std::array<std::string, 6> filepaths = {
            GetResourcePath("Textures/Sky/Clouds/Clouds_0Back_Raw.tga"),
            GetResourcePath("Textures/Sky/Clouds/Clouds_1Front_Raw.tga"),
            GetResourcePath("Textures/Sky/Clouds/Clouds_2Down_Raw.tga"),
            GetResourcePath("Textures/Sky/Clouds/Clouds_3Up_Raw.tga"),
            GetResourcePath("Textures/Sky/Clouds/Clouds_4Right_Raw.tga"),
            GetResourcePath("Textures/Sky/Clouds/Clouds_5Left_Raw.tga"),
        };

        std::array<const char *, 6> cFilepaths = { };
        std::ranges::transform(
            filepaths,
            cFilepaths.begin(),
            [](const std::string &path)
            {
                return path.c_str();
            }
        );

        auto [cubeMapTexture, cubeMapStagingBuffer] = LoadCubeMap(cFilepaths, m_Device, loadResourcesCommands);
        m_SkyboxTexture = cubeMapTexture;

        EndOneTimeCommands(loadResourcesCommands);

        missingTextureStagingBuffer.Destroy(m_Device.GetHandle());
        arenaIndicesStagingBuffer.Destroy(m_Device.GetHandle());
        arenaVerticesStagingBuffer.Destroy(m_Device.GetHandle());

        for (Buffer &buffer : dragonStagingBuffers)
        {
            buffer.Destroy(m_Device.GetHandle());
        }

        for (Buffer &buffer : skyboxStagingBuffers)
        {
            buffer.Destroy(m_Device.GetHandle());
        }

        cubeMapStagingBuffer.Destroy(m_Device.GetHandle());

        m_SkyboxView = m_Device.GetHandle().createImageView(
            vk::ImageViewCreateInfo()
                .setImage(m_SkyboxTexture.Handle)
                .setViewType(vk::ImageViewType::eCube)
                .setFormat(vk::Format::eB8G8R8A8Srgb)
                .setSubresourceRange(
                    vk::ImageSubresourceRange()
                        .setAspectMask(vk::ImageAspectFlagBits::eColor)
                        .setLevelCount(1)
                        .setLayerCount(6)
                )
        );
    }

    void CreateTextureSampler()
    {
        const vk::PhysicalDeviceProperties &properties = m_Device.GetProperties();

        const auto createInfo = vk::SamplerCreateInfo()
            .setMagFilter(vk::Filter::eLinear)
            .setMinFilter(vk::Filter::eLinear)
            .setMipmapMode(vk::SamplerMipmapMode::eLinear)
            .setAddressModeU(vk::SamplerAddressMode::eRepeat)
            .setAddressModeV(vk::SamplerAddressMode::eRepeat)
            .setAddressModeW(vk::SamplerAddressMode::eRepeat)
            .setMipLodBias(0.0F)
            .setAnisotropyEnable(VK_TRUE)
            .setMaxAnisotropy(properties.limits.maxSamplerAnisotropy)
            .setCompareEnable(VK_FALSE)
            .setCompareOp(vk::CompareOp::eAlways)
            .setMinLod(0.0F)
            .setMaxLod(0.0F)
            .setBorderColor(vk::BorderColor::eIntOpaqueBlack)
            .setUnnormalizedCoordinates(VK_FALSE);

        m_TextureSampler = m_Device.GetHandle().createSampler(createInfo);
    }

    void CreateUniformBuffers()
    {
        constexpr vk::DeviceSize bufferSize = sizeof(UniformBufferObject);

        m_UniformBuffers.resize(MaxFramesInFlight);
        m_UniformBuffersMapped.resize(MaxFramesInFlight);

        for (std::size_t i = 0; i < MaxFramesInFlight; ++i)
        {
            m_UniformBuffers[i] = m_Device.CreateBuffer(
                bufferSize,
                vk::BufferUsageFlagBits::eUniformBuffer,
                vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent
            );

            m_UniformBuffersMapped[i] = m_Device.GetHandle().mapMemory(m_UniformBuffers[i].Memory, 0, bufferSize);
            if (!m_UniformBuffersMapped[i])
            {
                throw std::runtime_error("Failed to map UBO memory");
            }
        }
    }

    void CreateDescriptorPool()
    {
        constexpr std::array poolSizes = {
            vk::DescriptorPoolSize()
                .setType(vk::DescriptorType::eUniformBuffer)
                .setDescriptorCount(MaxFramesInFlight),
            vk::DescriptorPoolSize()
                .setType(vk::DescriptorType::eCombinedImageSampler)
                .setDescriptorCount(TextureCount + 1),
        };

        const auto poolInfo = vk::DescriptorPoolCreateInfo()
            .setMaxSets(MaxFramesInFlight + TextureCount + 1) // One UBO per frame, one set per texture, one for skybox.
            .setPoolSizes(poolSizes);

        m_DescriptorPool = m_Device.GetHandle().createDescriptorPool(poolInfo);
    }

    void CreateDescriptorSets()
    {
        const std::vector<vk::DescriptorSetLayout> uboLayouts(MaxFramesInFlight, m_UboDescriptorSetLayout);

        const auto uboAllocInfo = vk::DescriptorSetAllocateInfo()
            .setDescriptorPool(m_DescriptorPool)
            .setSetLayouts(uboLayouts);

        m_UboDescriptorSets = m_Device.GetHandle().allocateDescriptorSets(uboAllocInfo);

        for (std::size_t i = 0; i < MaxFramesInFlight; ++i)
        {
            const auto bufferInfo = vk::DescriptorBufferInfo()
                .setBuffer(m_UniformBuffers[i].Handle)
                .setOffset(0)
                .setRange(sizeof(UniformBufferObject));

            const auto descriptorWrite = vk::WriteDescriptorSet()
                .setDstSet(m_UboDescriptorSets[i])
                .setDstBinding(0)
                .setDstArrayElement(0)
                .setDescriptorCount(1)
                .setDescriptorType(vk::DescriptorType::eUniformBuffer)
                .setPBufferInfo(&bufferInfo);

            m_Device.GetHandle().updateDescriptorSets({ descriptorWrite }, { });
        }

        const std::array samplerDescriptorSets = { m_SamplerDescriptorSetLayout };

        const auto textureAllocInfo = vk::DescriptorSetAllocateInfo()
            .setDescriptorPool(m_DescriptorPool)
            .setSetLayouts(samplerDescriptorSets);

        (void)m_Device.GetHandle().allocateDescriptorSets(&textureAllocInfo, &m_MissingTextureDescriptorSet);
        (void)m_Device.GetHandle().allocateDescriptorSets(&textureAllocInfo, &m_SkyboxTextureDescriptorSet);

        const std::array imageInfos = {
            vk::DescriptorImageInfo()
                .setSampler(m_TextureSampler)
                .setImageView(m_MissingTextureImageView)
                .setImageLayout(vk::ImageLayout::eShaderReadOnlyOptimal),
            vk::DescriptorImageInfo()
                .setSampler(m_TextureSampler)
                .setImageView(m_SkyboxView)
                .setImageLayout(vk::ImageLayout::eShaderReadOnlyOptimal),
        };

        const std::array descriptorWrites = {
            vk::WriteDescriptorSet()
                .setDstSet(m_MissingTextureDescriptorSet)
                .setDstBinding(0)
                .setDstArrayElement(0)
                .setDescriptorCount(1)
                .setDescriptorType(vk::DescriptorType::eCombinedImageSampler)
                .setPImageInfo(&imageInfos[0]),
            vk::WriteDescriptorSet()
                .setDstSet(m_SkyboxTextureDescriptorSet)
                .setDstBinding(0)
                .setDstArrayElement(0)
                .setDescriptorCount(1)
                .setDescriptorType(vk::DescriptorType::eCombinedImageSampler)
                .setPImageInfo(&imageInfos[1]),
        };

        m_Device.GetHandle().updateDescriptorSets(descriptorWrites, { });
    }

    void CreateCommandBuffers()
    {
        const std::uint32_t count = MaxFramesInFlight;

        const auto allocInfo = vk::CommandBufferAllocateInfo()
            .setCommandPool(m_CommandPool)
            .setLevel(vk::CommandBufferLevel::ePrimary)
            .setCommandBufferCount(count);

        m_CommandBuffers = m_Device.GetHandle().allocateCommandBuffers(allocInfo);
    }

    vk::CommandBuffer BeginOneTimeCommands()
    {
        const auto allocInfo = vk::CommandBufferAllocateInfo()
            .setCommandPool(m_CommandPool)
            .setLevel(vk::CommandBufferLevel::ePrimary)
            .setCommandBufferCount(1);

        vk::CommandBuffer commandBuffer;
        (void)m_Device.GetHandle().allocateCommandBuffers(&allocInfo, &commandBuffer);

        const auto beginInfo = vk::CommandBufferBeginInfo()
            .setFlags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);

        commandBuffer.begin(beginInfo);

        return commandBuffer;
    }

    void EndOneTimeCommands(vk::CommandBuffer commandBuffer)
    {
        vkEndCommandBuffer(commandBuffer);

        const std::array commandBuffers = { commandBuffer };

        const auto submitInfo = vk::SubmitInfo()
            .setCommandBuffers(commandBuffers);

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
        const std::array clearValues = {
            vk::ClearValue(),
            vk::ClearValue(),
        };

        const vk::Framebuffer framebuffer = m_SwapChainFramebuffers[imageIndex];
        const auto renderPassBeginInfo = vk::RenderPassBeginInfo()
            .setRenderPass(m_RenderPass)
            .setFramebuffer(framebuffer)
            .setRenderArea(
                vk::Rect2D()
                    .setExtent(m_SwapChainExtent)
            )
            .setClearValues(clearValues);

        const auto viewport = vk::Viewport()
            .setWidth(static_cast<float>(m_SwapChainExtent.width))
            .setHeight(static_cast<float>(m_SwapChainExtent.height))
            .setMinDepth(0.0F)
            .setMaxDepth(1.0F);

        const auto scissor = vk::Rect2D()
            .setExtent(m_SwapChainExtent);

        commandBuffer.beginRenderPass(renderPassBeginInfo, vk::SubpassContents::eInline);
        {
            commandBuffer.setViewport(0, 1, &viewport);
            commandBuffer.setScissor(0, 1, &scissor);

            commandBuffer.bindDescriptorSets(
                vk::PipelineBindPoint::eGraphics,
                m_PipelineLayout,
                0,
                { m_UboDescriptorSets[m_CurrentFrameIndex] },
                { }
            );

            commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, m_ModelPipeline);

            // TODO: This will eventually become a loop over all models to be drawn.
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

            commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, m_SkyboxPipeline);
            commandBuffer.bindDescriptorSets(
                vk::PipelineBindPoint::eGraphics,
                m_PipelineLayout,
                1,
                { m_SkyboxTextureDescriptorSet },
                { }
            );
            commandBuffer.bindVertexBuffers(0, { m_Skybox.Vertices.Buffer.Handle }, { 0 });
            commandBuffer.bindIndexBuffer(m_Skybox.Indices.Buffer.Handle, 0, vk::IndexType::eUint32);
            commandBuffer.drawIndexed(m_Skybox.Indices.Count, 1, 0, 0, 0);
        }
        commandBuffer.endRenderPass();

        commandBuffer.end();
    }

    void CreateSyncObjects()
    {
        m_ImageAvailableSemaphores.reserve(MaxFramesInFlight);
        m_RenderFinishedSemaphores.reserve(MaxFramesInFlight);
        m_InFlightFences.reserve(MaxFramesInFlight);

        const auto fenceInfo = vk::FenceCreateInfo()
            .setFlags(vk::FenceCreateFlagBits::eSignaled);

        for (std::uint32_t i = 0; i < MaxFramesInFlight; ++i)
        {
            m_ImageAvailableSemaphores.push_back(m_Device.GetHandle().createSemaphore({ }));
            m_RenderFinishedSemaphores.push_back(m_Device.GetHandle().createSemaphore({ }));
            m_InFlightFences.push_back(m_Device.GetHandle().createFence(fenceInfo));
        }
    }

    vk::ShaderModule CreateShaderModule(std::span<const std::uint8_t> code)
    {
        const auto createInfo = vk::ShaderModuleCreateInfo()
            .setCodeSize(static_cast<std::uint32_t>(code.size()))
            .setPCode(reinterpret_cast<const std::uint32_t *>(code.data()));

        return m_Device.GetHandle().createShaderModule(createInfo);
    }

    void EnterMainLoop()
    {
        constexpr auto initialFrameTime = std::chrono::duration<float>(1.0F / 30.0F);

        m_CurrentTick = std::chrono::steady_clock::now();
        m_LastTick = m_CurrentTick - std::chrono::duration_cast<std::chrono::steady_clock::duration>(initialFrameTime);

        while (!glfwWindowShouldClose(m_Window))
        {
            m_FrameTimeSum -= m_FrameTimeSamples[m_FrameTimeSampleIndex];
            m_FrameTimeSamples[m_FrameTimeSampleIndex] = m_CurrentTick - m_LastTick;
            m_FrameTimeSum += m_FrameTimeSamples[m_FrameTimeSampleIndex];
            m_SmoothedFrameTime = m_FrameTimeSum / m_FrameTimeSamples.size();
            const float deltaTime = std::chrono::duration<float>(m_FrameTimeSamples[m_FrameTimeSampleIndex]).count();
            m_FrameTimeSampleIndex = (m_FrameTimeSampleIndex + 1) % m_FrameTimeSamples.size();

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
            std::chrono::duration<float, std::milli>(m_SmoothedFrameTime).count(),
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

        const auto submitInfo = vk::SubmitInfo()
            .setWaitSemaphores(waitSemaphores)
            .setWaitDstStageMask(waitStages)
            .setCommandBuffers(commandBuffers)
            .setSignalSemaphores(signalSemaphores);

        m_Device.GetGraphicsQueue().submit(submitInfo, m_InFlightFences[m_CurrentFrameIndex]);

        const std::array swapchains = { m_SwapChain };

        const auto presentInfo = vk::PresentInfoKHR()
            .setWaitSemaphores(signalSemaphores)
            .setSwapchains(swapchains)
            .setPImageIndices(&imageIndex);

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

        void *destination = m_UniformBuffersMapped[currentImage];
        std::memcpy(destination, &ubo, sizeof(ubo));
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

        for (vk::Framebuffer framebuffer : m_SwapChainFramebuffers)
        {
            m_Device.GetHandle().destroy(framebuffer);
        }
        m_SwapChainFramebuffers.clear();

        for (vk::ImageView imageView : m_SwapChainImageViews)
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

        m_Device.GetHandle().destroyImageView(m_SkyboxView);
        m_SkyboxTexture.Destroy(m_Device.GetHandle());

        m_Skybox.Destroy(m_Device.GetHandle());
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

        m_Device.GetHandle().destroy(m_SkyboxPipeline);
        m_Device.GetHandle().destroy(m_ModelPipeline);
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

        if (app->m_IsFirstMouseMovement)
        {
            app->m_IsFirstMouseMovement = false;
            app->m_LastMousePos = app->m_MousePos;
        }
    }

    static bool IsRequiredExtensionsSupported(std::span<const vk::ExtensionProperties> deviceExtensions)
    {
        return std::ranges::includes(
            deviceExtensions,
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
        const auto candidate = std::ranges::find_if(
            surfaceFormats, [](const vk::SurfaceFormatKHR &surfaceFormat)
            {
                return surfaceFormat.format == vk::Format::eB8G8R8A8Srgb
                       && surfaceFormat.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear;
            }
        );

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
    //  Textures should probably all be stored in the same container and referenced by models via their index. The
    //  application would then be able to determine the total number of textures by the size of the container.
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
    vk::Pipeline m_ModelPipeline;
    vk::Pipeline m_SkyboxPipeline;

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
    vk::DescriptorSet m_SkyboxTextureDescriptorSet;

    Model m_DragonModel;
    Model m_Skybox;

    Image m_SkyboxTexture;
    vk::ImageView m_SkyboxView;

    std::vector<Buffer> m_UniformBuffers;
    std::vector<void *> m_UniformBuffersMapped;
    std::vector<vk::DescriptorSet> m_UboDescriptorSets;

    std::uint32_t m_CurrentFrameIndex = 0;
    std::vector<vk::CommandBuffer> m_CommandBuffers;
    std::vector<vk::Semaphore> m_ImageAvailableSemaphores;
    std::vector<vk::Semaphore> m_RenderFinishedSemaphores;
    std::vector<vk::Fence> m_InFlightFences;

    std::chrono::steady_clock::time_point m_LastTick = { };
    std::chrono::steady_clock::time_point m_CurrentTick = { };
    std::chrono::steady_clock::duration m_FrameTimeSum = { };
    std::chrono::steady_clock::duration m_SmoothedFrameTime = { };
    std::size_t m_FrameTimeSampleIndex = 0;
    std::array<std::chrono::steady_clock::duration, FrameTimeSampleCount> m_FrameTimeSamples = { };

    struct Mouse
    {
        double XPos;
        double YPos;
    };

    Mouse m_LastMousePos = { };
    Mouse m_MousePos = { };
    bool m_IsFirstMouseMovement = true;

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
