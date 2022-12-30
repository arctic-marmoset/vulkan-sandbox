# Vulkan Sandbox

This is where I will experiment with the Vulkan graphics API. The plan is to
start by following [Vulkan Tutorial](https://vulkan-tutorial.com/) to build up a
foundation. Then, after completing the tutorial, I will modify the code with the
aim to produce a basic 3D renderer. The scope of the 3D renderer will be decided
at a later point.

# External Dependencies

This project uses vcpkg to pull in most dependencies. However, the following must
still be installed manually:

| Dependency                                         | Version Used in Project |
|----------------------------------------------------|-------------------------|
| A C++ compiler with C++20 support                  | `15.0.5 (clang)`        |
| [vcpkg](https://github.com/microsoft/vcpkg) itself | -                       |
| [CMake](https://cmake.org/download/)               | `3.24.0`                |
| [Vulkan SDK](https://vulkan.lunarg.com/)           | `1.3.236`               |

# Project Setup

This project has only been tested on Linux and Windows 10. Other platforms may or
may not also work.

## Clone the Repository
```
git clone --recurse-submodules {repo}
```

## Configure CMake
There are currently two compilers (MSVC, Clang) and two build configurations (Debug, Release) with presets.

The presets are named `{Compiler}-{Configuration}`, with each component spelled exactly as above.

```
cd {projectRoot}
cmake --preset {preset}
```

## Build
```
cmake --build build/{preset}
```

## Run
```
cd {projectRoot}/build/{preset}
./renderer
```

Note the directory change. This is necessary since the program expects resources
to be located relative to the binary directory.

# Progress Preview

![Depth 2022-04-08](docs/images/2022-04-08_depth_trimmed.gif "Depth buffering")

Reversed depth buffer visualisation:
![Depth Visualised 2022-04-08](docs/images/2022-04-08_depth_visualised.png "Depth buffer visualised")
