cmake_minimum_required(VERSION 3.16)

project(cgvk C CXX)

find_package(SDL2 CONFIG REQUIRED)
find_program(GLSLANG_VALIDATOR
    NAMES glslangValidator
    HINTS $ENV{HOME}/.local/bin "/usr/local/bin" "/usr/bin"
)

function(compile_shader fname)
    add_custom_command(
        COMMENT "Compile shader: ${fname} -> ${fname}.inc"
        OUTPUT ${fname}.inc
        COMMAND ${GLSLANG_VALIDATOR} -V -x -o ${CMAKE_CURRENT_SOURCE_DIR}/${fname}.inc ${PROJECT_SOURCE_DIR}/${fname}
        MAIN_DEPENDENCY ${PROJECT_SOURCE_DIR}/${fname}
        DEPENDS ${PROJECT_SOURCE_DIR}/${fname} ${GLSLANG_VALIDATOR}
    )
    add_custom_command(
        COMMENT "Compile shader: ${fname} -> ${fname}.spv"
        OUTPUT ${fname}.spv
        COMMAND ${GLSLANG_VALIDATOR} -V -o ${CMAKE_CURRENT_SOURCE_DIR}/${fname}.spv ${PROJECT_SOURCE_DIR}/${fname}
        MAIN_DEPENDENCY ${PROJECT_SOURCE_DIR}/${fname}
        DEPENDS ${PROJECT_SOURCE_DIR}/${fname} ${GLSLANG_VALIDATOR}
    )
endfunction()

compile_shader(triangle.vert)
compile_shader(triangle.frag)

add_library(cgvk SHARED)
target_sources(cgvk
PRIVATE
    cgvk.cc
    vk_mem_alloc.cc
    volk.c
    log.c
    xxhash.c
    triangle.vert.inc
    triangle.frag.inc
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
