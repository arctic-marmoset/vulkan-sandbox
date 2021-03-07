cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(VulkanSandbox)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)

add_executable(renderer
    src/main.cpp
)
