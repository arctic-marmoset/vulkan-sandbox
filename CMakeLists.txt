cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(VulkanSandbox)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)

add_executable(renderer
    src/main.cpp
    src/vulkan_dispatcher.cpp
)

find_package(glfw3 REQUIRED)
find_package(Vulkan REQUIRED)

target_compile_definitions(renderer
    PRIVATE
        GLFW_INCLUDE_NONE
        VULKAN_HPP_NO_STRUCT_CONSTRUCTORS
        VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1
)

target_include_directories(renderer
    PRIVATE
        include
)

target_link_libraries(renderer
    PRIVATE
        ${CMAKE_DL_LIBS}
        glfw
        vulkan
)
