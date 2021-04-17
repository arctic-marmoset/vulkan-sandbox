#include "utility.hpp"

#include <GLFW/glfw3.h>
#include <vulkan/vulkan.hpp>

#include <algorithm>
#include <array>
#include <bitset>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <functional>
#include <iostream>
#include <limits>
#include <optional>
#include <set>
#include <stdexcept>
#include <string_view>

constexpr std::uint32_t window_width  = 1280;
constexpr std::uint32_t window_height = 720;

constexpr std::array validation_layers = {
    "VK_LAYER_KHRONOS_validation",
};

constexpr std::array device_extensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME,
};

#ifndef NDEBUG
constexpr bool debug_mode = true;
#else
constexpr bool debug_mode = false;
#endif

constexpr auto triangle_vertices = std::to_array<::vertex>({
    { {  0.0f, -0.5f }, { 1.0f, 0.0f, 0.0f } },
    { {  0.5f,  0.5f }, { 0.0f, 1.0f, 0.0f } },
    { { -0.5f,  0.5f }, { 0.0f, 0.0f, 1.0f } },
});

VKAPI_ATTR vk::Bool32 VKAPI_CALL debug_callback(VkDebugUtilsMessageSeverityFlagBitsEXT message_severity,
                                                [[maybe_unused]] VkDebugUtilsMessageTypeFlagsEXT message_type,
                                                const VkDebugUtilsMessengerCallbackDataEXT *data,
                                                [[maybe_unused]] void *user_data)
{
    switch (message_severity) {
    case VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT:
    case VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT:
        std::cout << "[DEBUG_CALLBACK] " << data->pMessage << '\n';
        break;

    case VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT:
    case VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT:
    case VK_DEBUG_UTILS_MESSAGE_SEVERITY_FLAG_BITS_MAX_ENUM_EXT:
        std::cerr << "[DEBUG_CALLBACK] " << data->pMessage << '\n';
        break;
    }

    return VK_FALSE;
}

bool validation_layers_supported()
{
    auto layers = vk::enumerateInstanceLayerProperties();
    std::ranges::sort(layers, { }, &vk::LayerProperties::layerName);

    const auto name_projection = [](const vk::LayerProperties &properties) {
        return static_cast<std::string_view>(properties.layerName);
    };

    return std::ranges::includes(layers, validation_layers, { }, name_projection);
}

std::vector<const char *> required_extensions()
{
    std::uint32_t glfw_extension_count = 0;
    const char **glfw_extensions = glfwGetRequiredInstanceExtensions(&glfw_extension_count);

    std::vector extensions(glfw_extensions, glfw_extensions + glfw_extension_count);

    if constexpr (debug_mode)
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

    return extensions;
}

void init_debug_messenger_create_info(vk::DebugUtilsMessengerCreateInfoEXT &create_info)
{
    create_info = {
        .messageSeverity = vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose
                         | vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning
                         | vk::DebugUtilsMessageSeverityFlagBitsEXT::eError,

        .messageType     = vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral
                         | vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation
                         | vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance,

        .pfnUserCallback = debug_callback,
    };
}

class application
{
public:
    static constexpr std::array dynamic_states = {
        vk::DynamicState::eScissor,
        vk::DynamicState::eViewport,
    };

public:
    void run()
    {
        init_window();
        init_vulkan();
        main_loop();
        cleanup();
    }

private:
    void init_window()
    {
        if (!glfwInit())
            throw std::runtime_error("Failed to initialize GLFW!");

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

        window_ = glfwCreateWindow(window_width, window_height, "Vulkan Renderer", nullptr, nullptr);

        if (!window_)
            throw std::runtime_error("Failed to create window!");

        glfwSetWindowUserPointer(window_, this);
        glfwSetFramebufferSizeCallback(window_, framebuffer_size_callback);
    }

    void init_vulkan()
    {
        create_instance();
        attach_debug_messenger();
        create_surface();
        select_physical_device();
        create_logical_device();
        create_swapchain();
        create_image_views();
        create_render_pass();
        create_graphics_pipeline();
        create_framebuffers();
        create_command_pool();
        create_vertex_buffer();
        create_command_buffers();
        create_sync_objects();
    }

    void create_instance()
    {
        const vk::DynamicLoader loader;

        // NOLINTNEXTLINE(readability-identifier-naming): Vulkan-specific function name
        auto *vkGetInstanceProcAddr = loader.getProcAddress<PFN_vkGetInstanceProcAddr>("vkGetInstanceProcAddr");
        VULKAN_HPP_DEFAULT_DISPATCHER.init(vkGetInstanceProcAddr);

        if constexpr (debug_mode) {
            if (!validation_layers_supported())
                throw std::runtime_error("Validation layers requested but not supported!");
        }

        const vk::ApplicationInfo app_info = {
            .pApplicationName   = "Hello Triangle",
            .applicationVersion = VK_MAKE_VERSION(0, 1, 0),
            .pEngineName        = "Sandbox",
            .engineVersion      = VK_MAKE_VERSION(0, 1, 0),
            .apiVersion         = VK_API_VERSION_1_0,
        };

        vk::StructureChain<vk::InstanceCreateInfo, vk::DebugUtilsMessengerCreateInfoEXT> chain;
        auto &create_info       = chain.get<vk::InstanceCreateInfo>();
        auto &debug_create_info = chain.get<vk::DebugUtilsMessengerCreateInfoEXT>();

        const auto extensions = required_extensions();

        create_info = {
            .pApplicationInfo        = &app_info,
            .enabledExtensionCount   = static_cast<std::uint32_t>(extensions.size()),
            .ppEnabledExtensionNames = extensions.data(),
        };

        if constexpr (debug_mode) {
            init_debug_messenger_create_info(debug_create_info);
            create_info.enabledLayerCount   = static_cast<std::uint32_t>(validation_layers.size());
            create_info.ppEnabledLayerNames = validation_layers.data();
        } else {
            chain.unlink<vk::DebugUtilsMessengerCreateInfoEXT>();
        }

        instance_ = vk::createInstance(create_info);
        VULKAN_HPP_DEFAULT_DISPATCHER.init(instance_);
    }

    void attach_debug_messenger()
    {
        if constexpr (!debug_mode)
            return;

        vk::DebugUtilsMessengerCreateInfoEXT create_info;
        init_debug_messenger_create_info(create_info);

#ifndef NDEBUG
        debug_messenger_ = instance_.createDebugUtilsMessengerEXT(create_info);
#endif
    }

    void create_surface()
    {
        VkSurfaceKHR surface = VK_NULL_HANDLE;

        if (glfwCreateWindowSurface(instance_, window_, nullptr, &surface) != VK_SUCCESS)
            throw std::runtime_error("Failed to window surface!");

        surface_ = surface;
    }

    void select_physical_device()
    {
        const auto devices = instance_.enumeratePhysicalDevices();

        if (devices.empty())
            throw std::runtime_error("Failed to find a GPU with Vulkan support!");

        const auto candidate = std::ranges::find_if(devices, [this](const vk::PhysicalDevice &device) {
            return is_device_suitable(device);
        });

        if (candidate == devices.end())
            throw std::runtime_error("Failed to find a GPU suitable for this application!");

        physical_device_ = *candidate;
    }

    bool is_device_suitable(const vk::PhysicalDevice &device)
    {
        const auto indices = find_queue_families(device);

        const bool supports_extensions = device_supports_extensions(device);

        const bool swapchain_adequate = supports_extensions && std::invoke([&] {
            const auto details = query_swapchain_support(device);
            return !details.formats.empty() && !details.present_modes.empty();
        });

        return indices.complete()
            && supports_extensions
            && swapchain_adequate;
    }

    ::queue_family_indices find_queue_families(const vk::PhysicalDevice &device)
    {
        ::queue_family_indices indices;

        const auto queue_families = device.getQueueFamilyProperties();

        for (std::uint32_t i = 0; i < queue_families.size(); ++i) {
            const auto &queue_family = queue_families[i];

            if (queue_family.queueFlags & vk::QueueFlagBits::eGraphics)
                indices.graphics_family = i;

            const vk::Bool32 present_support = device.getSurfaceSupportKHR(i, surface_);

            if (present_support)
                indices.present_family = i;

            if (indices.complete())
                break;
        }

        return indices;
    }

    void create_logical_device()
    {
        const ::queue_family_indices indices = find_queue_families(physical_device_);

        const float queue_priority = 1.0f;

        const std::set unique_queue_families = {
            indices.graphics_family.value(),
            indices.present_family.value(),
        };

        std::vector<vk::DeviceQueueCreateInfo> queue_create_infos;
        queue_create_infos.reserve(unique_queue_families.size());

        for (auto queue_family : unique_queue_families) {
            queue_create_infos.push_back({
                .queueFamilyIndex = queue_family,
                .queueCount       = 1,
                .pQueuePriorities = &queue_priority,
            });
        }

        const vk::PhysicalDeviceFeatures device_features = { };

        const vk::DeviceCreateInfo create_info = {
            .queueCreateInfoCount    = static_cast<std::uint32_t>(queue_create_infos.size()),
            .pQueueCreateInfos       = queue_create_infos.data(),
            .enabledExtensionCount   = static_cast<std::uint32_t>(device_extensions.size()),
            .ppEnabledExtensionNames = device_extensions.data(),
            .pEnabledFeatures        = &device_features,
        };

        device_ = physical_device_.createDevice(create_info);
        VULKAN_HPP_DEFAULT_DISPATCHER.init(device_);

        graphics_queue_ = device_.getQueue(indices.graphics_family.value(), 0);
        present_queue_  = device_.getQueue(indices.present_family.value(), 0);
    }

    ::swapchain_support_details query_swapchain_support(const vk::PhysicalDevice &device)
    {
        return {
            .capabilities  = device.getSurfaceCapabilitiesKHR(surface_),
            .formats       = device.getSurfaceFormatsKHR(surface_),
            .present_modes = device.getSurfacePresentModesKHR(surface_),
        };
    }

    void create_swapchain()
    {
        const auto swapchain_support = query_swapchain_support(physical_device_);

        const auto surface_format = select_swap_surface_format(swapchain_support.formats);
        const auto present_mode   = select_swap_present_mode(swapchain_support.present_modes);
        const auto extent         = select_swap_extent(swapchain_support.capabilities);

        const std::uint32_t desired_image_count = swapchain_support.capabilities.minImageCount + 1;
        const std::uint32_t max_image_count     = swapchain_support.capabilities.maxImageCount;

        const auto image_count = max_image_count == 0
                               ? desired_image_count
                               : std::min(desired_image_count, max_image_count);

        const auto indices = find_queue_families(physical_device_);

        const std::array queue_family_indices = {
            indices.graphics_family.value(),
            indices.present_family.value(),
        };

        old_swapchain_ = swapchain_;

        const auto create_info = std::invoke([&] {
            vk::SwapchainCreateInfoKHR result = {
                .surface          = surface_,
                .minImageCount    = image_count,
                .imageFormat      = surface_format.format,
                .imageColorSpace  = surface_format.colorSpace,
                .imageExtent      = extent,
                .imageArrayLayers = 1,
                .imageUsage       = vk::ImageUsageFlagBits::eColorAttachment,
                .preTransform     = swapchain_support.capabilities.currentTransform,
                .presentMode      = present_mode,
                .clipped          = VK_TRUE,
                .oldSwapchain     = old_swapchain_,
            };

            if (indices.graphics_family != indices.present_family) {
                result.imageSharingMode      = vk::SharingMode::eConcurrent;
                result.queueFamilyIndexCount = 2;
                result.pQueueFamilyIndices   = queue_family_indices.data();
            } else {
                result.imageSharingMode = vk::SharingMode::eExclusive;
            }

            return result;
        });

        swapchain_image_format_ = surface_format.format;
        swapchain_extent_       = extent;

        swapchain_        = device_.createSwapchainKHR(create_info);
        swapchain_images_ = device_.getSwapchainImagesKHR(swapchain_);
    }

    vk::Extent2D select_swap_extent(const vk::SurfaceCapabilitiesKHR &capabilities)
    {
        if (capabilities.currentExtent.width != std::numeric_limits<std::uint32_t>::max())
            return capabilities.currentExtent;

        int frame_width  = 0;
        int frame_height = 0;

        glfwGetFramebufferSize(window_, &frame_width, &frame_height);

        const auto width  = static_cast<std::uint32_t>(frame_width);
        const auto height = static_cast<std::uint32_t>(frame_height);

        const auto [min_width, min_height] = capabilities.minImageExtent;
        const auto [max_width, max_height] = capabilities.maxImageExtent;

        return {
            .width  = std::clamp(width, min_width, max_width),
            .height = std::clamp(height, min_height, max_height),
        };
    }

    void create_image_views()
    {
        swapchain_image_views_.reserve(swapchain_images_.size());

        for (const auto &image : swapchain_images_) {
            const vk::ImageViewCreateInfo create_info = {
                .image              = image,
                .viewType           = vk::ImageViewType::e2D,
                .format             = swapchain_image_format_,
                .subresourceRange   = {
                    .aspectMask     = vk::ImageAspectFlagBits::eColor,
                    .baseMipLevel   = 0,
                    .levelCount     = 1,
                    .baseArrayLayer = 0,
                    .layerCount     = 1,
                },
            };

            const auto image_view = device_.createImageView(create_info);
            swapchain_image_views_.push_back(image_view);
        }
    }

    void create_render_pass()
    {
        const vk::AttachmentDescription color_attachment = {
            .format         = swapchain_image_format_,
            .samples        = vk::SampleCountFlagBits::e1,
            .loadOp         = vk::AttachmentLoadOp::eClear,
            .storeOp        = vk::AttachmentStoreOp::eStore,
            .stencilLoadOp  = vk::AttachmentLoadOp::eDontCare,
            .stencilStoreOp = vk::AttachmentStoreOp::eDontCare,
            .initialLayout  = vk::ImageLayout::eUndefined,
            .finalLayout    = vk::ImageLayout::ePresentSrcKHR,
        };

        const vk::AttachmentReference color_attachment_reference = {
            .attachment = 0,
            .layout     = vk::ImageLayout::eColorAttachmentOptimal,
        };

        const vk::SubpassDescription subpass_description = {
            .colorAttachmentCount = 1,
            .pColorAttachments    = &color_attachment_reference,
        };

        const vk::SubpassDependency dependency = {
            .srcSubpass    = VK_SUBPASS_EXTERNAL,
            .dstSubpass    = 0,
            .srcStageMask  = vk::PipelineStageFlagBits::eColorAttachmentOutput,
            .dstStageMask  = vk::PipelineStageFlagBits::eColorAttachmentOutput,
            .srcAccessMask = vk::AccessFlagBits::eNoneKHR,
            .dstAccessMask = vk::AccessFlagBits::eColorAttachmentWrite,
        };

        const vk::RenderPassCreateInfo render_pass_create_info = {
            .attachmentCount = 1,
            .pAttachments    = &color_attachment,
            .subpassCount    = 1,
            .pSubpasses      = &subpass_description,
            .dependencyCount = 1,
            .pDependencies   = &dependency,
        };

        render_pass_ = device_.createRenderPass(render_pass_create_info);
    }

    void create_graphics_pipeline()
    {
        const auto vertex_shader_bytecode   = read_file("resources/shaders/triangle/triangle.vert.spv");
        const auto fragment_shader_bytecode = read_file("resources/shaders/triangle/triangle.frag.spv");

        const auto vertex_shader_module   = create_shader_module(vertex_shader_bytecode);
        const auto fragment_shader_module = create_shader_module(fragment_shader_bytecode);

        const vk::PipelineShaderStageCreateInfo vertex_shader_stage_create_info = {
            .stage  = vk::ShaderStageFlagBits::eVertex,
            .module = vertex_shader_module,
            .pName  = "main",
        };

        const vk::PipelineShaderStageCreateInfo fragment_shader_stage_create_info = {
            .stage  = vk::ShaderStageFlagBits::eFragment,
            .module = fragment_shader_module,
            .pName  = "main",
        };

        const std::array shader_stages = {
            vertex_shader_stage_create_info,
            fragment_shader_stage_create_info,
        };

        const auto binding_description    = ::vertex::binding_description();
        const auto attribute_descriptions = ::vertex::attribute_descriptions();

        const vk::PipelineVertexInputStateCreateInfo vertex_input_state_create_info = {
            .vertexBindingDescriptionCount   = 1,
            .pVertexBindingDescriptions      = &binding_description,
            .vertexAttributeDescriptionCount = static_cast<std::uint32_t>(attribute_descriptions.size()),
            .pVertexAttributeDescriptions    = attribute_descriptions.data(),
        };

        const vk::PipelineInputAssemblyStateCreateInfo input_assembly_state_create_info = {
            .topology               = vk::PrimitiveTopology::eTriangleList,
            .primitiveRestartEnable = VK_FALSE,
        };

        const vk::PipelineViewportStateCreateInfo viewport_state_create_info = {
            .viewportCount = 1,
            .pViewports    = nullptr, // Dynamic
            .scissorCount  = 1,
            .pScissors     = nullptr, // Dynamic
        };

        const vk::PipelineRasterizationStateCreateInfo rasterization_state_create_info = {
            .depthClampEnable        = VK_FALSE,
            .rasterizerDiscardEnable = VK_FALSE,
            .polygonMode             = vk::PolygonMode::eFill,
            .cullMode                = vk::CullModeFlagBits::eBack,
            .frontFace               = vk::FrontFace::eClockwise,
            .depthBiasEnable         = VK_FALSE,
            .lineWidth               = 1.0f,
        };

        const vk::PipelineMultisampleStateCreateInfo multisample_state_create_info = {
            .rasterizationSamples = vk::SampleCountFlagBits::e1,
            .sampleShadingEnable  = VK_FALSE,
        };

        const vk::PipelineColorBlendAttachmentState color_blend_attachment_state = {
            .blendEnable    = VK_FALSE,
            .colorWriteMask = vk::ColorComponentFlagBits::eR
                            | vk::ColorComponentFlagBits::eG
                            | vk::ColorComponentFlagBits::eB
                            | vk::ColorComponentFlagBits::eA,
        };

        const vk::PipelineColorBlendStateCreateInfo color_blend_state_create_info = {
            .logicOpEnable   = VK_FALSE,
            .attachmentCount = 1,
            .pAttachments    = &color_blend_attachment_state,
        };

        const vk::PipelineLayoutCreateInfo pipeline_layout_create_info = { };

        pipeline_layout_ = device_.createPipelineLayout(pipeline_layout_create_info);

        const vk::PipelineDynamicStateCreateInfo dynamic_state_create_info = {
            .dynamicStateCount = static_cast<std::uint32_t>(dynamic_states.size()),
            .pDynamicStates    = dynamic_states.data(),
        };

        const vk::GraphicsPipelineCreateInfo pipeline_create_info = {
            .stageCount          = 2,
            .pStages             = shader_stages.data(),
            .pVertexInputState   = &vertex_input_state_create_info,
            .pInputAssemblyState = &input_assembly_state_create_info,
            .pViewportState      = &viewport_state_create_info,
            .pRasterizationState = &rasterization_state_create_info,
            .pMultisampleState   = &multisample_state_create_info,
            .pDepthStencilState  = nullptr,
            .pColorBlendState    = &color_blend_state_create_info,
            .pDynamicState       = &dynamic_state_create_info,
            .layout              = pipeline_layout_,
            .renderPass          = render_pass_,
            .subpass             = 0,
        };

        const auto [result, pipeline] = device_.createGraphicsPipeline({ }, pipeline_create_info);

        if (result != vk::Result::eSuccess)
            throw std::runtime_error("Failed to create graphics pipeline!");

        graphics_pipeline_ = pipeline;

        device_.destroy(vertex_shader_module);
        device_.destroy(fragment_shader_module);
    }

    void create_framebuffers()
    {
        swapchain_framebuffers_.reserve(swapchain_image_views_.size());

        for (const auto &image_view : swapchain_image_views_) {
            const vk::FramebufferCreateInfo create_info = {
                .renderPass      = render_pass_,
                .attachmentCount = 1,
                .pAttachments    = &image_view,
                .width           = swapchain_extent_.width,
                .height          = swapchain_extent_.height,
                .layers          = 1,
            };

            const auto framebuffer = device_.createFramebuffer(create_info);
            swapchain_framebuffers_.push_back(framebuffer);
        }
    }

    void create_command_pool()
    {
        const ::queue_family_indices indices = find_queue_families(physical_device_);

        const vk::CommandPoolCreateInfo create_info = {
            .queueFamilyIndex = indices.graphics_family.value(),
        };

        command_pool_ = device_.createCommandPool(create_info);
    }

    void create_buffer(vk::DeviceSize size,
                       vk::BufferUsageFlags usage,
                       vk::MemoryPropertyFlags properties,
                       vk::Buffer &buffer,
                       vk::DeviceMemory &buffer_memory)
    {
        const vk::BufferCreateInfo buffer_info = {
            .size        = size,
            .usage       = usage,
            .sharingMode = vk::SharingMode::eExclusive,
        };

        buffer = device_.createBuffer(buffer_info);

        const vk::MemoryRequirements memory_requirements = device_.getBufferMemoryRequirements(buffer);

        const std::uint32_t memory_type_index = find_memory_type(memory_requirements.memoryTypeBits, properties);

        const vk::MemoryAllocateInfo allocate_info = {
            .allocationSize  = memory_requirements.size,
            .memoryTypeIndex = memory_type_index,
        };

        buffer_memory = device_.allocateMemory(allocate_info);
        device_.bindBufferMemory(buffer, buffer_memory, 0);
    }

    void copy_buffer(vk::Buffer source, vk::Buffer destination, vk::DeviceSize size)
    {
        const vk::CommandBufferAllocateInfo allocate_info = {
            .commandPool        = command_pool_,
            .level              = vk::CommandBufferLevel::ePrimary,
            .commandBufferCount = 1,
        };

        vk::CommandBuffer command_buffer;
        device_.allocateCommandBuffers(&allocate_info, &command_buffer);

        const vk::CommandBufferBeginInfo begin_info = {
            .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit,
        };

        command_buffer.begin(begin_info);

        const vk::BufferCopy copy_region = {
            .srcOffset = 0,
            .dstOffset = 0,
            .size      = size,
        };

        command_buffer.copyBuffer(source, destination, copy_region);

        command_buffer.end();

        const vk::SubmitInfo submit_info = {
            .commandBufferCount = 1,
            .pCommandBuffers    = &command_buffer,
        };

        graphics_queue_.submit(submit_info);
        graphics_queue_.waitIdle();
        device_.free(command_pool_, command_buffer);
    }

    void create_vertex_buffer()
    {
        constexpr auto triangle_vertices_size =
            static_cast<vk::DeviceSize>(sizeof(decltype(triangle_vertices)::value_type) * triangle_vertices.size());

        vk::Buffer staging_buffer;
        vk::DeviceMemory staging_buffer_memory;
        create_buffer(triangle_vertices_size,
                      vk::BufferUsageFlagBits::eTransferSrc,
                      vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
                      staging_buffer,
                      staging_buffer_memory);

        if (void *data = device_.mapMemory(staging_buffer_memory, 0, triangle_vertices_size)) {
            std::memcpy(data, triangle_vertices.data(), triangle_vertices_size);
            device_.unmapMemory(staging_buffer_memory);
        }

        create_buffer(triangle_vertices_size,
                      vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eVertexBuffer,
                      vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
                      vertex_buffer_,
                      vertex_buffer_memory_);

        copy_buffer(staging_buffer, vertex_buffer_, triangle_vertices_size);
        device_.destroy(staging_buffer);
        device_.free(staging_buffer_memory);
    }

    std::uint32_t find_memory_type(std::uint32_t type_filter, vk::MemoryPropertyFlags properties)
    {
        constexpr auto filter_bitcount = std::numeric_limits<decltype(type_filter)>::digits;
        const std::bitset<filter_bitcount> eligible_types(type_filter);

        constexpr auto type_present = [](std::bitset<filter_bitcount> types, std::uint32_t index) {
            return types.test(index);
        };

        constexpr auto properties_present = [](vk::MemoryPropertyFlags superset, vk::MemoryPropertyFlags subset) {
            return (superset & subset) == subset;
        };

        const vk::PhysicalDeviceMemoryProperties memory_properties = physical_device_.getMemoryProperties();
        for (std::uint32_t i = 0; i < memory_properties.memoryTypeCount; ++i) {
            const auto candidate_properties = memory_properties.memoryTypes[i].propertyFlags;
            if (type_present(eligible_types, i)
                && properties_present(candidate_properties, properties)) {
                return i;
            }
        }

        throw std::runtime_error("Failed to find a suitable memory type!");
    }

    void create_command_buffers()
    {
        const auto count = static_cast<std::uint32_t>(swapchain_framebuffers_.size());

        const vk::CommandBufferAllocateInfo alloc_info = {
            .commandPool        = command_pool_,
            .level              = vk::CommandBufferLevel::ePrimary,
            .commandBufferCount = count,
        };

        command_buffers_ = device_.allocateCommandBuffers(alloc_info);

        for (std::size_t i = 0; i < command_buffers_.size(); ++i) {
            const auto &command_buffer = command_buffers_[i];
            const auto &framebuffer    = swapchain_framebuffers_[i];

            const vk::CommandBufferBeginInfo begin_info = { };
            command_buffer.begin(begin_info);

            const std::array color = { 0.0f, 0.0f, 0.0f, 1.0f };
            const vk::ClearValue clear_color = { color };

            const vk::RenderPassBeginInfo render_pass_begin_info = {
                .renderPass      = render_pass_,
                .framebuffer     = framebuffer,
                .renderArea      = {
                    .offset      = {
                        .x       = 0,
                        .y       = 0,
                    },
                    .extent      = swapchain_extent_,
                },
                .clearValueCount = 1,
                .pClearValues    = &clear_color,
            };

            const vk::Viewport viewport = {
                .x        = 0.0f,
                .y        = 0.0f,
                .width    = static_cast<float>(swapchain_extent_.width),
                .height   = static_cast<float>(swapchain_extent_.height),
                .minDepth = 0.0f,
                .maxDepth = 1.0f,
            };

            const vk::Rect2D scissor = {
                .offset = {
                    .x  = 0,
                    .y  = 0,
                },
                .extent = swapchain_extent_,
            };

            const std::array vertex_buffers = {
                vertex_buffer_,
            };

            const auto offsets = std::to_array<vk::DeviceSize>({
                0,
            });

            command_buffer.beginRenderPass(render_pass_begin_info, vk::SubpassContents::eInline);
            {
                command_buffer.bindPipeline(vk::PipelineBindPoint::eGraphics, graphics_pipeline_);
                command_buffer.setViewport(0, 1, &viewport);
                command_buffer.setScissor(0, 1, &scissor);
                command_buffer.bindVertexBuffers(0, vertex_buffers, offsets);
                command_buffer.draw(static_cast<std::uint32_t>(triangle_vertices.size()), 1, 0, 0);
            }
            command_buffer.endRenderPass();

            command_buffer.end();
        }
    }

    void create_sync_objects()
    {
        image_available_semaphores_.reserve(max_frames_in_flight_);
        render_finished_semaphores_.reserve(max_frames_in_flight_);
        in_flight_fences_.reserve(max_frames_in_flight_);
        images_in_flight_.resize(swapchain_images_.size());

        const vk::FenceCreateInfo fence_info = {
            .flags = vk::FenceCreateFlagBits::eSignaled,
        };

        for (std::uint32_t i = 0; i < max_frames_in_flight_; ++i) {
            image_available_semaphores_.push_back(device_.createSemaphore({ }));
            render_finished_semaphores_.push_back(device_.createSemaphore({ }));
            in_flight_fences_.push_back(device_.createFence(fence_info));
        }
    }

    vk::ShaderModule create_shader_module(const std::vector<std::byte> &code)
    {
        const vk::ShaderModuleCreateInfo create_info = {
            .codeSize = static_cast<std::uint32_t>(code.size()),
            .pCode    = reinterpret_cast<const std::uint32_t *>(code.data()),
        };

        return device_.createShaderModule(create_info);
    }

    void main_loop()
    {
        while (!glfwWindowShouldClose(window_)) {
            glfwPollEvents();
            draw_frame();
        }

        device_.waitIdle();
    }

    void draw_frame()
    {
        [[maybe_unused]]
        auto wait_result = device_.waitForFences(1,
                                                 &in_flight_fences_[current_frame_],
                                                 VK_TRUE,
                                                 std::numeric_limits<std::uint64_t>::max());

        std::uint32_t image_index = 0;

        // Have to use the C interface since the C++ one annoyingly throws an exception for eErrorOutOfDateKHR
        const auto acquire_next_image_result = static_cast<vk::Result>(
            vkAcquireNextImageKHR(device_,
                                  swapchain_,
                                  std::numeric_limits<std::uint64_t>::max(),
                                  image_available_semaphores_[current_frame_],
                                  nullptr,
                                  &image_index));

        switch (acquire_next_image_result) {
        case vk::Result::eSuccess:
        case vk::Result::eSuboptimalKHR:
            break;
        case vk::Result::eErrorOutOfDateKHR:
            recreate_swapchain();
            return;
        default:
            throw std::runtime_error("Failed to acquire swapchain image!");
        }

        if (images_in_flight_[image_index]) {
            [[maybe_unused]]
            auto unused = device_.waitForFences(1,
                                                &images_in_flight_[image_index],
                                                VK_TRUE,
                                                std::numeric_limits<std::uint64_t>::max());
        }

        images_in_flight_[image_index] = in_flight_fences_[current_frame_];

        const std::array wait_semaphores = {
            image_available_semaphores_[current_frame_],
        };

        const auto wait_stages = std::to_array<vk::PipelineStageFlags>({
            vk::PipelineStageFlagBits::eColorAttachmentOutput,
        });

        const std::array signal_semaphores = {
            render_finished_semaphores_[current_frame_],
        };

        const vk::SubmitInfo submit_info = {
            .waitSemaphoreCount   = static_cast<std::uint32_t>(wait_semaphores.size()),
            .pWaitSemaphores      = wait_semaphores.data(),
            .pWaitDstStageMask    = wait_stages.data(),
            .commandBufferCount   = 1,
            .pCommandBuffers      = &command_buffers_[image_index],
            .signalSemaphoreCount = static_cast<std::uint32_t>(signal_semaphores.size()),
            .pSignalSemaphores    = signal_semaphores.data(),
        };

        [[maybe_unused]]
        auto reset_result = device_.resetFences(1, &in_flight_fences_[current_frame_]);

        graphics_queue_.submit(submit_info, in_flight_fences_[current_frame_]);

        const std::array swapchains = {
            swapchain_,
        };

        const vk::PresentInfoKHR present_info = {
            .waitSemaphoreCount = static_cast<std::uint32_t>(signal_semaphores.size()),
            .pWaitSemaphores    = signal_semaphores.data(),
            .swapchainCount     = static_cast<std::uint32_t>(swapchains.size()),
            .pSwapchains        = swapchains.data(),
            .pImageIndices      = &image_index,
        };

        const auto c_present_info = static_cast<VkPresentInfoKHR>(present_info);

        // Have to use the C interface since the C++ one annoyingly throws an exception for eErrorOutOfDateKHR
        const auto present_result = static_cast<vk::Result>(vkQueuePresentKHR(present_queue_, &c_present_info));

        if (present_result == vk::Result::eSuboptimalKHR
            || present_result == vk::Result::eErrorOutOfDateKHR
            || framebuffer_resized_) {
            recreate_swapchain();
        } else if (present_result != vk::Result::eSuccess) {
            throw std::runtime_error("Failed to present swapchain image!");
        }

        current_frame_ = (current_frame_ + 1) % max_frames_in_flight_;
    }

    void recreate_swapchain()
    {
        framebuffer_resized_ = false;

        device_.waitIdle();

        cleanup_swapchain();

        create_swapchain();
        create_image_views();
        create_render_pass();
        create_framebuffers();
        create_command_buffers();
    }

    void cleanup_swapchain()
    {
        for (auto framebuffer : swapchain_framebuffers_)
            device_.destroy(framebuffer);

        swapchain_framebuffers_.clear();

        device_.freeCommandBuffers(command_pool_, command_buffers_);
        command_buffers_.clear();

        device_.destroy(render_pass_);

        for (auto image_view : swapchain_image_views_)
            device_.destroy(image_view);

        swapchain_image_views_.clear();

        device_.destroy(old_swapchain_);
    }

    void cleanup()
    {
        cleanup_swapchain();
        device_.destroy(swapchain_);

        device_.destroy(vertex_buffer_);
        device_.free(vertex_buffer_memory_);

        for (std::uint32_t i = 0; i < max_frames_in_flight_; ++i) {
            device_.destroy(image_available_semaphores_[i]);
            device_.destroy(render_finished_semaphores_[i]);
            device_.destroy(in_flight_fences_[i]);
        }

        device_.destroy(command_pool_);

        device_.destroy(graphics_pipeline_);
        device_.destroy(pipeline_layout_);

        device_.destroy();

        instance_.destroy(surface_);

#ifndef NDEBUG
        instance_.destroy(debug_messenger_);
#endif

        instance_.destroy();
        glfwDestroyWindow(window_);
        glfwTerminate();
    }

    static void framebuffer_size_callback(GLFWwindow *window, [[maybe_unused]] int width, [[maybe_unused]] int height)
    {
        auto *self = static_cast<::application *>(glfwGetWindowUserPointer(window));
        self->framebuffer_resized_ = true;
    }

    static bool device_supports_extensions(const vk::PhysicalDevice &device)
    {
        auto extensions = device.enumerateDeviceExtensionProperties();
        std::ranges::sort(extensions, { }, &vk::ExtensionProperties::extensionName);

        const auto name_projection = [](const vk::ExtensionProperties &extension) {
            return static_cast<std::string_view>(extension.extensionName);
        };

        return std::ranges::includes(extensions, device_extensions, { }, name_projection);
    }

    static vk::SurfaceFormatKHR select_swap_surface_format(const std::vector<vk::SurfaceFormatKHR> &surface_formats)
    {
        const auto candidate = std::ranges::find_if(surface_formats, [](const vk::SurfaceFormatKHR &surface_format) {
            return surface_format.format == vk::Format::eB8G8R8A8Srgb
                && surface_format.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear;
        });

        if (candidate == surface_formats.end())
            return surface_formats.front();

        return *candidate;
    }

    static vk::PresentModeKHR select_swap_present_mode(const std::vector<vk::PresentModeKHR> &present_modes)
    {
        const auto candidate = std::ranges::find_if(present_modes, [](const vk::PresentModeKHR &present_mode) {
            return present_mode == vk::PresentModeKHR::eMailbox;
        });

        if (candidate == present_modes.end())
            return vk::PresentModeKHR::eFifo;

        return *candidate;
    }

private:
    bool framebuffer_resized_ = false;
    std::uint32_t max_frames_in_flight_ = 2;
    std::size_t current_frame_ = 0;
    GLFWwindow *window_ = nullptr;
    vk::Instance instance_;
#ifndef NDEBUG
    vk::DebugUtilsMessengerEXT debug_messenger_;
#endif
    vk::SurfaceKHR surface_;
    vk::PhysicalDevice physical_device_;
    vk::Device device_;
    vk::Queue graphics_queue_;
    vk::Queue present_queue_;
    vk::Format swapchain_image_format_ = vk::Format::eUndefined;
    vk::Extent2D swapchain_extent_;
    vk::SwapchainKHR swapchain_;
    vk::SwapchainKHR old_swapchain_;
    std::vector<vk::Image> swapchain_images_;
    std::vector<vk::ImageView> swapchain_image_views_;
    vk::RenderPass render_pass_;
    vk::PipelineLayout pipeline_layout_;
    vk::Pipeline graphics_pipeline_;
    std::vector<vk::Framebuffer> swapchain_framebuffers_;
    vk::CommandPool command_pool_;
    vk::Buffer vertex_buffer_;
    vk::DeviceMemory vertex_buffer_memory_;
    std::vector<vk::CommandBuffer> command_buffers_;
    std::vector<vk::Semaphore> image_available_semaphores_;
    std::vector<vk::Semaphore> render_finished_semaphores_;
    std::vector<vk::Fence> in_flight_fences_;
    std::vector<vk::Fence> images_in_flight_;
};

int main()
{
    ::application app;

    try {
        app.run();
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
