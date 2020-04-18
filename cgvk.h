// CGVK: Simple C Library for 3D Graphics on Vulkan
//
// *Warning*
// The library and example codes are under active development
//

#ifndef CGVK_H_
#define CGVK_H_

#ifdef __cplusplus
#  define CGVK_API extern "C"
#else
#  define CGVK_API
#endif

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

CGVK_API void cgvk_init(const char* appname);
CGVK_API void cgvk_quit();
CGVK_API bool cgvk_pump_events();
CGVK_API void cgvk_render();

#endif // CGVK_H_
