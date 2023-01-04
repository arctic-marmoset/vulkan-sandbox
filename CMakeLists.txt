cmake_minimum_required(VERSION 3.24 FATAL_ERROR)
project(VulkanRenderer)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_EXTENSIONS FALSE)

include(CheckIPOSupported)
check_ipo_supported(RESULT IPO_SUPPORTED)
if(IPO_SUPPORTED)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_DEBUG FALSE)
endif()
message(STATUS "IPO supported: ${IPO_SUPPORTED}")

if(MSVC)
    string(REGEX REPLACE "[-/]W[1-4]" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    set(RENDERER_REQUIRED_WARN_FLAGS /W4 /permissive-)
else()
    set(RENDERER_REQUIRED_WARN_FLAGS -Wall -Wextra -Wpedantic -Wnon-virtual-dtor)
endif()

option(RENDERER_TREAT_WARN_AS_ERROR "Treat compiler warnings as errors" FALSE)

if(RENDERER_TREAT_WARN_AS_ERROR)
    if(MSVC)
        list(APPEND RENDERER_REQUIRED_WARN_FLAGS /Wx)
    else()
        list(APPEND RENDERER_REQUIRED_WARN_FLAGS -Werror -pedantic-errors)
    endif()
endif()

# See: https://github.com/cpp-best-practices/cppbestpractices.
option(RENDERER_USE_EXTENSIVE_WARNINGS "Enable warnings recommended by C++ Best Practices" TRUE)

if(RENDERER_USE_EXTENSIVE_WARNINGS)
    if(MSVC)
        set(RENDERER_EXTRA_WARN_FLAGS
            /w14242 # possible loss of data: conversion from `type1` to `type2`
            /w14254 # possible loss of data: conversion from `type1:field_bits`
                    # to `type2:field_bits`
            /w14263 # member function does not override any base class virtual
                    # member function
            /w14265 # class has virtual functions, but destructor is not
                    # virtual; instances of this class may not be destructed
                    # correctly
            /w14287 # unsigned/negative constant mismatch
            /we4289 # loop control variable declared in the for-loop is used
                    # outside the for-loop scope
            /w14296 # expression is always [true | false]
            /w14311 # pointer truncation from `type1` to `type2`
            /w14545 # expression before comma evaluates to a function which is
                    # missing an argument list
            /w14546 # function call before comma missing argument list
            /w14547 # operator before comma has no effect; expected operator
                    # with side-effect
            /w14549 # operator before comma has no effect; did you intend
                    # 'operator'?
            /w14555 # expression has no effect; expected expression with
                    # side-effect
            /w14619 # pragma warning: there is no warning number `number`
            /w14640 # thread un-safe static member initialization
            /w14826 # conversion from 'type1' to 'type_2' is sign-extended which
                    # may cause unexpected runtime behavior
            /w14928 # illegal copy-initialization; more than one user-defined
                    # conversion has been implicitly applied
        )
    else()
        set(RENDERER_EXTRA_WARN_FLAGS
            -Wcast-align
            -Wunused
            -Woverloaded-virtual
            -Wconversion
            -Wsign-conversion
            -Wnull-dereference
            -Wmisleading-indentation
            -Wformat=2
            -Wfloat-equal
        )
    endif()

    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        list(APPEND RENDERER_EXTRA_WARN_FLAGS
            -Wduplicated-cond
            -Wduplicated-branches
            -Wuseless-cast
            -Wlogical-op
        )
    elseif(CMAKE_CXX_COMPILER_ID MATCHES ".*Clang")
        list(APPEND RENDERER_EXTRA_WARN_FLAGS
            -Wimplicit-fallthrough
            -Wshorten-64-to-32
        )
    endif()
endif()

set(RESOURCES_DIR ${CMAKE_CURRENT_SOURCE_DIR}/Resources)

find_program(GLSL_VALIDATOR glslc REQUIRED)

set(SHADER_SOURCES
    Source/Shaders/Triangle.frag
    Source/Shaders/Triangle.vert
    Source/Shaders/Texture.frag
    Source/Shaders/Texture.vert
)

foreach(RELATIVE_SOURCE_PATH ${SHADER_SOURCES})
    if(NOT RELATIVE_SOURCE_PATH MATCHES "^Source/")
        message(WARNING "${RELATIVE_SOURCE_PATH} is not in the Source directory.")
    endif()

    get_filename_component(FILE_NAME_WITH_EXT ${RELATIVE_SOURCE_PATH} NAME)
    get_filename_component(FILE_DIR ${RELATIVE_SOURCE_PATH} DIRECTORY)
    string(REGEX
        REPLACE "^Source/" ""
        RESOURCES_RELATIVE_FILE_DIR
        ${FILE_DIR}
    )
    set(RESOURCES_RELATIVE_FILE_DIR "${RESOURCES_RELATIVE_FILE_DIR}/$<CONFIG>")

    set(SOURCE_PATH "${PROJECT_SOURCE_DIR}/${RELATIVE_SOURCE_PATH}")
    set(OUTPUT_DIR "${RESOURCES_DIR}/${RESOURCES_RELATIVE_FILE_DIR}")

    set(OUTPUT_FILE_WITH_EXT "${FILE_NAME_WITH_EXT}.spv")
    set(OUTPUT_FILE_PATH "${OUTPUT_DIR}/${OUTPUT_FILE_WITH_EXT}")

    add_custom_command(
        OUTPUT ${OUTPUT_FILE_PATH}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${OUTPUT_DIR}
        COMMAND ${GLSL_VALIDATOR} ${SOURCE_PATH} -g -o ${OUTPUT_FILE_PATH}
        DEPENDS ${SOURCE_PATH}
        VERBATIM
    )

    list(APPEND SPIRV_BYTECODE ${OUTPUT_FILE_PATH})

    string(MAKE_C_IDENTIFIER ${FILE_NAME_WITH_EXT} FILE_NAME_ESCAPED)
    string(TOUPPER ${FILE_NAME_ESCAPED} FILE_NAME_UPPER)
    list(APPEND RENDERER_SHADER_BYTECODE_PATH_DEFINES
        "${FILE_NAME_UPPER}_SHADER_RELATIVE_PATH=\"${RESOURCES_RELATIVE_FILE_DIR}/${OUTPUT_FILE_WITH_EXT}\""
    )
endforeach()

add_custom_target(RendererShaders
    DEPENDS ${SPIRV_BYTECODE}
)

find_package(glfw3 CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)
find_package(tinyobjloader CONFIG REQUIRED)
find_package(Vulkan REQUIRED)

# Workaround for older versions of GLM
if(NOT TARGET glm::glm)
    get_target_property(GLM_INCLUDE_DIR glm INTERFACE_INCLUDE_DIRECTORIES)
    add_library(glm::glm INTERFACE IMPORTED)
    set_target_properties(glm::glm PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES ${GLM_INCLUDE_DIR}
    )
endif()

add_executable(Renderer
    Source/Main.cpp
    Source/TinyObjLoader.cpp
)

target_precompile_headers(Renderer
    PRIVATE
        Source/pch.hpp
)

target_include_directories(Renderer
    PRIVATE
        include
)

add_dependencies(Renderer
    RendererShaders
)

target_compile_definitions(Renderer
    PRIVATE
        GLFW_INCLUDE_VULKAN
        GLM_FORCE_DEPTH_ZERO_TO_ONE
        GLM_FORCE_RADIANS
        VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1
        VULKAN_HPP_NO_CONSTRUCTORS
        ${RENDERER_SHADER_BYTECODE_PATH_DEFINES}
)

target_link_libraries(Renderer
    PRIVATE
        ${CMAKE_DL_LIBS}
        glfw
        glm::glm
        spdlog::spdlog_header_only
        tinyobjloader::tinyobjloader
        Vulkan::Vulkan
)

target_compile_options(Renderer
    PRIVATE
        ${RENDERER_REQUIRED_WARN_FLAGS}
        ${RENDERER_EXTRA_WARN_FLAGS}
)

# CMake flags for Release builds are suboptimal.
# See: https://gitlab.kitware.com/cmake/cmake/-/issues/20812.
# See: https://stackoverflow.com/questions/28178978/how-to-generate-pdb-files-for-release-build-with-cmake-flags.
if(MSVC)
    target_compile_options(Renderer
        PRIVATE
            $<$<CONFIG:Release>:/Zc:inline>
            $<$<CONFIG:Release>:/Zi>
            $<$<CONFIG:Release>:/Gy>
    )
    target_link_options(Renderer
        PRIVATE
            $<$<CONFIG:Release>:/DEBUG>
            $<$<CONFIG:Release>:/INCREMENTAL:NO>
            $<$<CONFIG:Release>:/OPT:REF>
            $<$<CONFIG:Release>:/OPT:ICF>
    )
elseif((CMAKE_CXX_COMPILER_ID STREQUAL "GNU") OR (CMAKE_CXX_COMPILER_ID MATCHES ".*Clang"))
    target_compile_options(Renderer
        PRIVATE
            $<$<CONFIG:Release>:-g>
    )
    target_link_options(Renderer
        PRIVATE
            $<$<CONFIG:Release>:-g>
    )
    if(WIN32)
        target_compile_options(Renderer
            PRIVATE
                $<$<CONFIG:Release>:-gcodeview>
        )
    endif()
endif()
