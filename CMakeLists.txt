cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(VulkanSandbox)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_EXTENSIONS FALSE)

option(SANDBOX_USE_EXTENSIVE_WARNINGS "Turn warnings up to 11" TRUE)
option(SANDBOX_TREAT_WARN_AS_ERROR "Treat compiler warnings as errors" FALSE)

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

find_package(glfw3 REQUIRED)
find_package(Vulkan REQUIRED)

add_executable(renderer
    src/main.cpp
    src/vulkan_dispatcher.cpp
)

find_program(GLSL_VALIDATOR glslc REQUIRED)

set(GLSL_SOURCES
    resources/shaders/triangle/triangle.frag
    resources/shaders/triangle/triangle.vert
)

foreach(RELATIVE_SOURCE_PATH ${GLSL_SOURCES})
    get_filename_component(FILE_NAME ${RELATIVE_SOURCE_PATH} NAME)
    get_filename_component(FILE_DIR ${RELATIVE_SOURCE_PATH} DIRECTORY)
    set(SOURCE_PATH "${PROJECT_SOURCE_DIR}/${RELATIVE_SOURCE_PATH}")
    set(OUTPUT_DIR "${PROJECT_BINARY_DIR}/${FILE_DIR}")
    set(OUTPUT_FILE "${OUTPUT_DIR}/${FILE_NAME}.spv")

    add_custom_command(
        OUTPUT ${OUTPUT_FILE}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${OUTPUT_DIR}
        COMMAND ${GLSL_VALIDATOR} ${SOURCE_PATH} -o ${OUTPUT_FILE}
        DEPENDS ${SOURCE_PATH}
        VERBATIM
    )
    list(APPEND SPIRV_BYTECODE ${OUTPUT_FILE})
endforeach()

add_custom_target(renderer_shaders
    DEPENDS ${SPIRV_BYTECODE}
)

add_dependencies(renderer
    renderer_shaders
)

target_compile_definitions(renderer
    PRIVATE
        GLFW_INCLUDE_VULKAN
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

target_compile_options(renderer PRIVATE ${SANDBOX_WARN_FLAGS})
