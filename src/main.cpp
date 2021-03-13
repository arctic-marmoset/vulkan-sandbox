#include <GLFW/glfw3.h>
#include <vulkan/vulkan.hpp>

#include <algorithm>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string_view>

struct queue_family_indices {
    std::optional<std::uint32_t> graphics_family;

    bool complete() const
    {
        return graphics_family.has_value();
    }
};

constexpr std::uint32_t window_width  = 1280;
constexpr std::uint32_t window_height = 720;

constexpr std::array validation_layers = {
    "VK_LAYER_KHRONOS_validation"
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

    const auto name_projection = [](const vk::LayerProperties &properties)
    {
        auto name = static_cast<std::string_view>(properties.layerName);
        return name;
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

void init_debug_messenger_create_info(vk::DebugUtilsMessengerCreateInfoEXT &createInfo)
{
    createInfo = {
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
        select_physical_device();
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

        const vk::ApplicationInfo appInfo = {
            .pApplicationName   = "Hello Triangle",
            .applicationVersion = VK_MAKE_VERSION(0, 1, 0),
            .pEngineName        = "Sandbox",
            .engineVersion      = VK_MAKE_VERSION(0, 1, 0),
            .apiVersion         = VK_API_VERSION_1_0,
        };

        vk::StructureChain<vk::InstanceCreateInfo, vk::DebugUtilsMessengerCreateInfoEXT> chain;
        auto &createInfo      = chain.get<vk::InstanceCreateInfo>();
        auto &debugCreateInfo = chain.get<vk::DebugUtilsMessengerCreateInfoEXT>();

        const auto extensions = required_extensions();

        createInfo = {
            .pApplicationInfo        = &appInfo,
            .enabledExtensionCount   = static_cast<std::uint32_t>(extensions.size()),
            .ppEnabledExtensionNames = extensions.data(),
        };

        if constexpr (debug_mode) {
            init_debug_messenger_create_info(debugCreateInfo);
            createInfo.enabledLayerCount   = static_cast<std::uint32_t>(validation_layers.size());
            createInfo.ppEnabledLayerNames = validation_layers.data();
        } else {
            chain.unlink<vk::DebugUtilsMessengerCreateInfoEXT>();
        }

        instance_ = vk::createInstance(createInfo);
        VULKAN_HPP_DEFAULT_DISPATCHER.init(instance_);
    }

    void attach_debug_messenger()
    {
        if constexpr (!debug_mode)
            return;

        vk::DebugUtilsMessengerCreateInfoEXT createInfo;
        init_debug_messenger_create_info(createInfo);

        debug_messenger_ = instance_.createDebugUtilsMessengerEXT(createInfo);
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
        return indices.complete();
    }

    ::queue_family_indices find_queue_families(const vk::PhysicalDevice &device)
    {
        ::queue_family_indices indices;

        const auto queue_families = device.getQueueFamilyProperties();

        for (std::uint32_t i = 0; i < queue_families.size(); ++i) {
            const auto &queue_family = queue_families[i];

            if (queue_family.queueFlags & vk::QueueFlagBits::eGraphics)
                indices.graphics_family = i;

            if (indices.complete())
                break;
        }

        return indices;
    }

    void main_loop()
    {
        while (!glfwWindowShouldClose(window_)) {
            glfwPollEvents();
        }
    }

    void cleanup()
    {
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
    vk::PhysicalDevice physical_device_;
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
