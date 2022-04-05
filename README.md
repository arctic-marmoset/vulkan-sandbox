# Vulkan Sandbox

This is where I will experiment with the Vulkan graphics API. The plan is to
start by following [Vulkan Tutorial](https://vulkan-tutorial.com/) to build up a
foundation. Then, after completing the tutorial, I will modify the code with the
aim to produce a basic 3D renderer. The scope of the 3D renderer will be decided
at a later point.

# Requirements

This project has only been tested on Linux and Windows, but should work on other
platforms. Vcpkg is recommended for managing dependencies.

| Dependency                                 | Version Used in Project |
|--------------------------------------------|-------------------------|
| A C++ compiler with C++20 support          | `13.0.1 (clang)`        |
| [CMake](https://cmake.org/download/)       | `3.23.0`                |
| [GLSLC](https://github.com/google/shaderc) | `2022.1`                |
| [Vulkan SDK](https://vulkan.lunarg.com/)   | `1.3.211`               |
| [GLFW](https://www.glfw.org/)              | `3.3.7 (glfw-x11)`      |
| [GLM](https://github.com/g-truc/glm)       | `0.9.9.8`               |

# Progress Preview

![Hello Triangle 2022-04-05](docs/images/2022-04-05_triangle_trimmed.gif "Hello, triangle!")
