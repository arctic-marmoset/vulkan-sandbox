#include <vulkan/vulkan.hpp>
#include <GLFW/glfw3.h>

#include <algorithm>
#include <cstdlib>
#include <exception>
#include <functional>
#include <iostream>
#include <limits>
#include <optional>
#include <set>
#include <stdexcept>
#include <string_view>

struct queue_family_indices {
    std::optional<std::uint32_t> graphics_family;
    std::optional<std::uint32_t> present_family;

    [[nodiscard]] bool complete() const
    {
        return graphics_family.has_value()
            && present_family.has_value();
    }
};

struct swapchain_support_details {
    vk::SurfaceCapabilitiesKHR capabilities;
    std::vector<vk::SurfaceFormatKHR> formats;
    std::vector<vk::PresentModeKHR> present_modes;
};

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
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

        window_ = glfwCreateWindow(window_width, window_height, "Vulkan Renderer", nullptr, nullptr);
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
    }

    void create_instance()
    {
        const vk::DynamicLoader loader;
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

        debug_messenger_ = instance_.createDebugUtilsMessengerEXT(create_info);
    }

    void create_surface()
    {
        VkSurfaceKHR surface;

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

    bool device_supports_extensions(const vk::PhysicalDevice &device)
    {
        auto extensions = device.enumerateDeviceExtensionProperties();
        std::ranges::sort(extensions, { }, &vk::ExtensionProperties::extensionName);

        const auto name_projection = [](const vk::ExtensionProperties &extension) {
            return static_cast<std::string_view>(extension.extensionName);
        };

        return std::ranges::includes(extensions, device_extensions, { }, name_projection);
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
        ::queue_family_indices indices = find_queue_families(physical_device_);

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
        };

        vk::PhysicalDeviceFeatures device_features = { };

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
        present_queue_ = device_.getQueue(indices.present_family.value(), 0);
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
        const auto present_mode = select_swap_present_mode(swapchain_support.present_modes);
        const auto extent = select_swap_extent(swapchain_support.capabilities);

        const std::uint32_t desired_image_count = swapchain_support.capabilities.minImageCount + 1;
        const std::uint32_t max_image_count = swapchain_support.capabilities.maxImageCount;

        const auto image_count = max_image_count == 0
                               ? desired_image_count
                               : std::min(desired_image_count, max_image_count);

        const auto indices = find_queue_families(physical_device_);

        const std::array queue_family_indices = {
            indices.graphics_family.value(),
            indices.present_family.value(),
        };

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
            };

            if (indices.graphics_family != indices.present_family) {
                result.imageSharingMode = vk::SharingMode::eConcurrent;
                result.queueFamilyIndexCount = 2;
                result.pQueueFamilyIndices = queue_family_indices.data();
            } else {
                result.imageSharingMode = vk::SharingMode::eExclusive;
            }

            return result;
        });

        swapchain_image_format_ = surface_format.format;
        swapchain_extent_ = extent;

        swapchain_ = device_.createSwapchainKHR(create_info);
        swapchain_images_ = device_.getSwapchainImagesKHR(swapchain_);
    }

    vk::SurfaceFormatKHR select_swap_surface_format(const std::vector<vk::SurfaceFormatKHR> &surface_formats)
    {
        const auto candidate = std::ranges::find_if(surface_formats, [](const vk::SurfaceFormatKHR &surface_format) {
            return surface_format.format == vk::Format::eB8G8R8A8Srgb
                && surface_format.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear;
        });

        if (candidate == surface_formats.end())
            return surface_formats.front();

        return *candidate;
    }

    vk::PresentModeKHR select_swap_present_mode(const std::vector<vk::PresentModeKHR> &present_modes)
    {
        const auto candidate = std::ranges::find_if(present_modes, [](const vk::PresentModeKHR &present_mode) {
            return present_mode == vk::PresentModeKHR::eMailbox;
        });

        if (candidate == present_modes.end())
            return vk::PresentModeKHR::eFifo;

        return *candidate;
    }

    vk::Extent2D select_swap_extent(const vk::SurfaceCapabilitiesKHR &capabilities)
    {
        if (capabilities.currentExtent.width != std::numeric_limits<std::uint32_t>::max())
            return capabilities.currentExtent;

        int frame_width;
        int frame_height;

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
                .image            = image,
                .viewType         = vk::ImageViewType::e2D,
                .format           = swapchain_image_format_,
                .subresourceRange = {
                    .aspectMask     = vk::ImageAspectFlagBits::eColor,
                    .baseMipLevel   = 0,
                    .levelCount     = 1,
                    .baseArrayLayer = 0,
                    .layerCount     = 1,
                },
            };

            auto image_view = device_.createImageView(create_info);
            swapchain_image_views_.push_back(image_view);
        }
    }

    void main_loop()
    {
        while (!glfwWindowShouldClose(window_)) {
            glfwPollEvents();
        }
    }

    void cleanup()
    {
        for (auto image_view : swapchain_image_views_)
            device_.destroy(image_view);

        device_.destroy(swapchain_);
        device_.destroy();

        instance_.destroy(surface_);

        if constexpr (debug_mode)
            instance_.destroy(debug_messenger_);

        instance_.destroy();
        glfwDestroyWindow(window_);
        glfwTerminate();
    }

private:
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
    vk::Format swapchain_image_format_;
    vk::Extent2D swapchain_extent_;
    vk::SwapchainKHR swapchain_;
    std::vector<vk::Image> swapchain_images_;
    std::vector<vk::ImageView> swapchain_image_views_;
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
