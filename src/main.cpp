#include <GLFW/glfw3.h>
#include <vulkan/vulkan.hpp>

#include <cstdlib>
#include <exception>
#include <iostream>
#include <stdexcept>

constexpr std::uint32_t window_width  = 1280;
constexpr std::uint32_t window_height = 720;

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
    }

    void create_instance()
    {
        const vk::ApplicationInfo appInfo = {
            .pApplicationName   = "Hello Triangle",
            .applicationVersion = VK_MAKE_VERSION(0, 1, 0),
            .pEngineName        = "Sandbox",
            .engineVersion      = VK_MAKE_VERSION(0, 1, 0),
            .apiVersion         = VK_API_VERSION_1_0,
        };

        std::uint32_t glfwExtensionCount = 0;
        const char **glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

        const vk::InstanceCreateInfo createInfo = {
            .pApplicationInfo        = &appInfo,
            .enabledExtensionCount   = glfwExtensionCount,
            .ppEnabledExtensionNames = glfwExtensions,
        };

        instance_ = vk::createInstance(createInfo);
    }

    void main_loop()
    {
        while (!glfwWindowShouldClose(window_)) {
            glfwPollEvents();
        }
    }

    void cleanup()
    {
        instance_.destroy();
        glfwDestroyWindow(window_);
        glfwTerminate();
    }

private:
    GLFWwindow *window_ = nullptr;
    vk::Instance instance_;
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
