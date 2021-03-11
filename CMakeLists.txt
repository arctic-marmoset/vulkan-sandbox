cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(VulkanSandbox)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)

add_executable(renderer
    src/main.cpp
)

find_package(glfw3 REQUIRED)

target_compile_definitions(renderer
    PRIVATE
        GLFW_INCLUDE_NONE
)

target_include_directories(renderer
    PRIVATE
        include
)

target_link_libraries(renderer
    PRIVATE
        glfw
)
