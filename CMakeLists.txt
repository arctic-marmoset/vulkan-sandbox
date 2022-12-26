cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(VulkanSandbox)

set(CMAKE_CXX_STANDARD           20)
set(CMAKE_CXX_STANDARD_REQUIRED  ON)
set(CMAKE_CXX_EXTENSIONS        OFF)

string(TOUPPER ${CMAKE_BUILD_TYPE} CMAKE_BUILD_TYPE_UPPER)
if(CMAKE_BUILD_TYPE_UPPER STREQUAL "RELEASE")
    include(CheckIPOSupported)
    check_ipo_supported(RESULT IPO_IS_SUPPORTED)
    if(IPO_IS_SUPPORTED)
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
    endif()
endif()

option(SANDBOX_USE_EXTENSIVE_WARNINGS "Turn warnings up to 11" ON)
option(SANDBOX_TREAT_WARN_AS_ERROR "Treat compiler warnings as errors" OFF)

if(MSVC)
    string(REGEX REPLACE "[-/]W[1-4]" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    set(SANDBOX_WARN_FLAGS /W4 /permissive-)
else()
    set(SANDBOX_WARN_FLAGS -Wall -Wextra -Wpedantic)
endif()

if(SANDBOX_TREAT_WARN_AS_ERROR)
    if(MSVC)
        list(APPEND SANDBOX_WARN_FLAGS /Wx)
    else()
        list(APPEND SANDBOX_WARN_FLAGS -Werror -pedantic-errors)
    endif()
endif()

if(SANDBOX_USE_EXTENSIVE_WARNINGS)
    if(MSVC)
        list(APPEND SANDBOX_WARN_FLAGS
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
        list(APPEND SANDBOX_WARN_FLAGS
            -Wshadow
            -Wnon-virtual-dtor
            -Wold-style-cast
            -Wcast-align
            -Wunused
            -Woverloaded-virtual
            -Wconversion
            -Wsign-conversion
            -Wnull-dereference
            -Wformat=2
            -Wfloat-equal
        )
    endif()

    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        list(APPEND SANDBOX_WARN_FLAGS
            -Wduplicated-cond
            -Wduplicated-branches
            -Wlogical-op
            -Wuseless-cast
        )
    elseif(CMAKE_CXX_COMPILER_ID MATCHES ".*Clang")
        list(APPEND SANDBOX_WARN_FLAGS
            -Wshorten-64-to-32
        )
    endif()
endif()

find_package(glfw3 CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(Vulkan REQUIRED)

# Workaround for older versions of GLM
if(NOT TARGET glm::glm)
    get_target_property(GLM_INCLUDE_DIR glm INTERFACE_INCLUDE_DIRECTORIES)
    add_library(glm::glm INTERFACE IMPORTED)
    set_target_properties(glm::glm PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES ${GLM_INCLUDE_DIR}
    )
endif()

add_executable(renderer
    src/main.cpp
)

find_program(GLSL_VALIDATOR glslc REQUIRED)

set(GLSL_SOURCES
    src/shaders/triangle.frag
    src/shaders/triangle.vert
    src/shaders/texture.frag
    src/shaders/texture.vert
)

foreach(RELATIVE_SOURCE_PATH ${GLSL_SOURCES})
    if(NOT RELATIVE_SOURCE_PATH MATCHES "^src/")
        message(WARNING "${RELATIVE_SOURCE_PATH} is not in src directory.")
    endif()

    get_filename_component(FILE_NAME_WITH_EXT ${RELATIVE_SOURCE_PATH} NAME)
    get_filename_component(FILE_DIR ${RELATIVE_SOURCE_PATH} DIRECTORY)
    string(REGEX
        REPLACE "^src/" "resources/"
        RESOURCES_FILE_DIR
        ${FILE_DIR}
    )

    set(SOURCE_PATH "${PROJECT_SOURCE_DIR}/${RELATIVE_SOURCE_PATH}")
    set(OUTPUT_DIR "${PROJECT_BINARY_DIR}/${RESOURCES_FILE_DIR}")

    set(OUTPUT_FILE_WITH_EXT "${FILE_NAME_WITH_EXT}.spv")
    set(OUTPUT_FILE_PATH "${OUTPUT_DIR}/${OUTPUT_FILE_WITH_EXT}")

    add_custom_command(
        OUTPUT ${OUTPUT_FILE_PATH}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${OUTPUT_DIR}
        COMMAND ${GLSL_VALIDATOR} ${SOURCE_PATH} -o ${OUTPUT_FILE_PATH}
        DEPENDS ${SOURCE_PATH}
        VERBATIM
    )

    list(APPEND SPIRV_BYTECODE ${OUTPUT_FILE_PATH})

    string(MAKE_C_IDENTIFIER ${FILE_NAME_WITH_EXT} FILE_NAME_ESCAPED)
    string(TOUPPER ${FILE_NAME_ESCAPED} FILE_NAME_UPPER)
    list(APPEND RENDERER_SHADER_BYTECODE_PATH_DEFINES
        "${FILE_NAME_UPPER}_SHADER_RELATIVE_PATH=\"${RESOURCES_FILE_DIR}/${OUTPUT_FILE_WITH_EXT}\""
    )
endforeach()

add_custom_target(renderer_shaders
    DEPENDS ${SPIRV_BYTECODE}
)

add_custom_target(renderer_resources
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${PROJECT_SOURCE_DIR}/resources ${PROJECT_BINARY_DIR}/resources
)

add_dependencies(renderer
    renderer_shaders
    renderer_resources
)

target_compile_definitions(renderer
    PRIVATE
        GLFW_INCLUDE_VULKAN
        GLM_FORCE_DEPTH_ZERO_TO_ONE
        GLM_FORCE_RADIANS
        VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1
        VULKAN_HPP_NO_CONSTRUCTORS
        ${RENDERER_SHADER_BYTECODE_PATH_DEFINES}
)

target_include_directories(renderer
    PRIVATE
        include
)

target_link_libraries(renderer
    PRIVATE
        ${CMAKE_DL_LIBS}
        glfw
        glm::glm
        Vulkan::Vulkan
)

target_compile_options(renderer PRIVATE ${SANDBOX_WARN_FLAGS})
