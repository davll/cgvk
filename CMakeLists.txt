cmake_minimum_required(VERSION 3.16)

project(cgvk C CXX)

find_package(SDL2 REQUIRED)

add_library(cgvk SHARED)
target_sources(cgvk
PRIVATE
    cgvk.cc
    vk_mem_alloc.cc
    volk.c
    log.c
PUBLIC
    cgvk.h
)
target_compile_options(cgvk
PRIVATE
    "-g"
    "-Wall"
    "-fno-exceptions"
    #"-Wpointer-sign"
    #"-Wpedantic"
)
target_link_libraries(cgvk PUBLIC SDL2::SDL2)

target_compile_definitions(cgvk
PRIVATE
    VK_USE_PLATFORM_WAYLAND_KHR
    VK_USE_PLATFORM_XCB_KHR
    VK_USE_PLATFORM_XLIB_KHR
)

add_executable(ex-hello ex-hello.c)
target_link_libraries(ex-hello PRIVATE cgvk)
target_compile_options(ex-hello PRIVATE "-g")
