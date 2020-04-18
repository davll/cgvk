// Roadmap
// 1) Hello world
// 2) Forward lighting
// 3) Remove global variables
// 4) Custom shading
// 5) Frame graph
// ?) cgvk.cc -> ccvk.c
//
// Coding Guidelines
// 1) Orthodox C++ (https://gist.github.com/bkaradzic/2e39896bc7d8c34e042b)
// 2) Avoid C++ standard library
// 3) No OOP
// 4) No templates

// ============================================================================

#include "cgvk.h"
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "volk.h"
#include "vk_mem_alloc.h"
#include "log.h"
#include <SDL2/SDL.h>
#include <SDL2/SDL_vulkan.h>

#define CGVK_VERSION_MAJOR 0
#define CGVK_VERSION_MINOR 1
#define CGVK_VERSION_PATCH 0

// ============================================================================

struct cgvk_Device {
    VkPhysicalDevice gpu;
    VkDevice device;
    VmaAllocator allocator;
    VkQueue main_queue;
    VkQueue present_queue;
    bool supports_KHR_swapchain;
    uint32_t main_queue_family;
    uint32_t present_queue_family;
};

// ============================================================================

static cgvk_Device dev_;

// ============================================================================

static VkInstance cgvk_get_instance();
static VkSurfaceKHR cgvk_get_surface();
static void cgvk_get_drawable_extent(VkExtent2D* extent);

// ============================================================================

static VkPhysicalDevice cgvk_choose_gpu()
{
    VkInstance instance;
    VkResult result;
    uint32_t num_gpus;

    instance = cgvk_get_instance();

    result = vkEnumeratePhysicalDevices(instance, &num_gpus, NULL);
    if (result != VK_SUCCESS) {
        log_fatal("vkEnumeratePhysicalDevices failed: %d", (int)result);
        abort();
    } else if (num_gpus == 0) {
        log_fatal("The machine does not have any GPU");
        abort();
    }

    VkPhysicalDevice gpus[num_gpus];
    result = vkEnumeratePhysicalDevices(instance, &num_gpus, gpus);
    if (result != VK_SUCCESS) {
        log_fatal("vkEnumeratePhysicalDevices failed: %d", (int)result);
        abort();
    }

    // TODO: Ask the user to select GPU
    return gpus[0];
}

static void cgvk_init_device(cgvk_Device* dev)
{
    VkResult result;

    VkInstance instance;
    VkSurfaceKHR surface;
    VkPhysicalDevice gpu;
    VkDevice device;

    uint32_t extension_count = 0;
    const char* extension_names[4];
    VkPhysicalDeviceFeatures2 features, avail_features;
    uint32_t main_queue_family = UINT32_MAX;
    uint32_t present_queue_family = UINT32_MAX;

    memset(dev, 0, sizeof(cgvk_Device));

    // Get globals
    instance = cgvk_get_instance();
    surface = cgvk_get_surface();

    // Choose GPU
    gpu = cgvk_choose_gpu();
    dev->gpu = gpu;

    // Choose features
    memset(&features, 0, sizeof(VkPhysicalDeviceFeatures2));
    memset(&avail_features, 0, sizeof(VkPhysicalDeviceFeatures2));
    features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    avail_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    vkGetPhysicalDeviceFeatures2(gpu, &avail_features);

    // Check device extensions
    {
        uint32_t available_extension_count;
        result = vkEnumerateDeviceExtensionProperties(gpu, NULL, &available_extension_count, NULL);
        if (result != VK_SUCCESS) {
            log_fatal("vkEnumerateDeviceExtensionProperties failed: %d", (int)result);
            abort();
        }
        if (available_extension_count > 0) {
            VkExtensionProperties properties[available_extension_count];
            result = vkEnumerateDeviceExtensionProperties(gpu, NULL, &available_extension_count, properties);
            if (result != VK_SUCCESS) {
                log_fatal("vkEnumerateDeviceExtensionProperties failed: %d", (int)result);
                abort();
            }
            for (uint32_t i = 0; i < available_extension_count; ++i) {
                const VkExtensionProperties* props = &properties[i];
                const char* extname = props->extensionName;
                if (strcmp(extname, VK_KHR_SWAPCHAIN_EXTENSION_NAME) == 0) {
                    dev->supports_KHR_swapchain = true;
                    extension_names[extension_count++] = VK_KHR_SWAPCHAIN_EXTENSION_NAME;
                }
            }
        }
    }

    // select queues
    {
        uint32_t queue_family_count;
        vkGetPhysicalDeviceQueueFamilyProperties(gpu, &queue_family_count, NULL);
        VkQueueFamilyProperties properties[queue_family_count];
        vkGetPhysicalDeviceQueueFamilyProperties(gpu, &queue_family_count, properties);
        VkBool32 supported[queue_family_count];
        for (uint32_t i = 0; i < queue_family_count; ++i) {
            result = vkGetPhysicalDeviceSurfaceSupportKHR(gpu, i, surface, &supported[i]);
            if (result != VK_SUCCESS) {
                log_fatal("vkGetPhysicalDeviceSurfaceSupportKHR failed: %d", (int)result);
                abort();
            }
        }
        for (uint32_t i = 0; i < queue_family_count; ++i) {
            const VkQueueFamilyProperties* p = &properties[i];
            if (p->queueCount == 0)
                continue;
            VkQueueFlags mask = VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT;
            if (main_queue_family == UINT32_MAX && (p->queueFlags & mask) == mask)
                main_queue_family = i;
            if (present_queue_family == UINT32_MAX && supported[i])
                present_queue_family = i;
        }
    }
    dev->main_queue_family = main_queue_family;
    dev->present_queue_family = present_queue_family;

    // make queue create infos
    uint32_t queue_create_info_count = (main_queue_family == present_queue_family ? 1 : 2);
    float priorities[1] = { 1.0f };
    VkDeviceQueueCreateInfo queue_create_infos[2] = {
        {
            .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            .pNext = NULL,
            .flags = 0,
            .queueFamilyIndex = main_queue_family,
            .queueCount = 1,
            .pQueuePriorities = priorities,
        },
        {
            .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            .pNext = NULL,
            .flags = 0,
            .queueFamilyIndex = present_queue_family,
            .queueCount = 1,
            .pQueuePriorities = priorities,
        },
    };

    // create device
    VkDeviceCreateInfo device_create_info = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .pNext = &features,
        .flags = 0,
        .queueCreateInfoCount = queue_create_info_count,
        .pQueueCreateInfos = queue_create_infos,
        .enabledLayerCount = 0,
        .ppEnabledLayerNames = NULL,
        .enabledExtensionCount = extension_count,
        .ppEnabledExtensionNames = extension_names,
        .pEnabledFeatures = NULL,
    };
    result = vkCreateDevice(gpu, &device_create_info, NULL, &device);
    if (result != VK_SUCCESS) {
        log_fatal("vkCreateDevice failed: %d", (int)result);
        abort();
    }
    dev->device = device;

    // Get queues
    vkGetDeviceQueue(device, main_queue_family, 0, &dev->main_queue);
    vkGetDeviceQueue(device, present_queue_family, 0, &dev->present_queue);

    // Init allocator
    VmaVulkanFunctions alloc_functions = {
        .vkGetPhysicalDeviceProperties = vkGetPhysicalDeviceProperties,
        .vkGetPhysicalDeviceMemoryProperties = vkGetPhysicalDeviceMemoryProperties,
        .vkAllocateMemory = vkAllocateMemory,
        .vkFreeMemory = vkFreeMemory,
        .vkMapMemory = vkMapMemory,
        .vkUnmapMemory = vkUnmapMemory,
        .vkFlushMappedMemoryRanges = vkFlushMappedMemoryRanges,
        .vkInvalidateMappedMemoryRanges = vkInvalidateMappedMemoryRanges,
        .vkBindBufferMemory = vkBindBufferMemory,
        .vkBindImageMemory = vkBindImageMemory,
        .vkGetBufferMemoryRequirements = vkGetBufferMemoryRequirements,
        .vkGetImageMemoryRequirements = vkGetImageMemoryRequirements,
        .vkCreateBuffer = vkCreateBuffer,
        .vkDestroyBuffer = vkDestroyBuffer,
        .vkCreateImage = vkCreateImage,
        .vkDestroyImage = vkDestroyImage,
        .vkCmdCopyBuffer = vkCmdCopyBuffer,
        .vkGetBufferMemoryRequirements2KHR = vkGetBufferMemoryRequirements2,
        .vkGetImageMemoryRequirements2KHR = vkGetImageMemoryRequirements2,
        .vkBindBufferMemory2KHR = vkBindBufferMemory2,
        .vkBindImageMemory2KHR = vkBindImageMemory2,
        .vkGetPhysicalDeviceMemoryProperties2KHR = vkGetPhysicalDeviceMemoryProperties2,
    };
    VmaAllocatorCreateInfo alloc_create_info = {
        .flags = 0,
        .physicalDevice = gpu,
        .device = device,
        .preferredLargeHeapBlockSize = 0,
        .pAllocationCallbacks = NULL,
        .pDeviceMemoryCallbacks = NULL,
        .frameInUseCount = 0,
        .pHeapSizeLimit = NULL,
        .pVulkanFunctions = &alloc_functions,
        .pRecordSettings = NULL,
        .instance = instance,
        .vulkanApiVersion = VK_API_VERSION_1_1,
    };
    result = vmaCreateAllocator(&alloc_create_info, &dev->allocator);
    if (result != VK_SUCCESS) {
        log_fatal("vmaCreateAllocator failed: %d", (int)result);
        abort();
    }
}

static void cgvk_kill_device(cgvk_Device* dev)
{
    if (dev->allocator)
        vmaDestroyAllocator(dev->allocator);

    if (dev->device)
        vkDestroyDevice(dev->device, NULL);

    memset(dev, 0, sizeof(cgvk_Device));
}

// ============================================================================

static VkInstance instance_ = VK_NULL_HANDLE;
static VkDebugUtilsMessengerEXT debug_messenger_ = VK_NULL_HANDLE;
static bool supports_EXT_debug_utils_ = false;
static bool supports_KHR_surface_ = false;
static bool supports_KHR_display_ = false;

static SDL_Window* window_ = NULL;
static VkSurfaceKHR surface_ = VK_NULL_HANDLE;

static void cgvk_init_instance(const char* appname, bool debug)
{
    assert(!instance_);

    static const char* ENGINE_NAME = "CGVK Renderer";
    static uint32_t ENGINE_VERSION = VK_MAKE_VERSION(CGVK_VERSION_MAJOR, CGVK_VERSION_MINOR, CGVK_VERSION_PATCH);

    uint32_t layer_count = 0;
    const char* layer_names[2];
    uint32_t extension_count = 0;
    const char* extension_names[16];

    // Find validation layers
    if (debug) {
        uint32_t available_layer_count;
        VkResult result = vkEnumerateInstanceLayerProperties(&available_layer_count, NULL);
        if (result != VK_SUCCESS) {
            log_fatal("vkEnumerateInstanceLayerProperties failed: %d", (int)result);
            abort();
        }
        if (available_layer_count > 0) {
            VkLayerProperties properties[available_layer_count];
            result = vkEnumerateInstanceLayerProperties(&available_layer_count, properties);
            if (result != VK_SUCCESS) {
                log_fatal("vkEnumerateInstanceLayerProperties failed: %d", (int)result);
                abort();
            }
            for (uint32_t i = 0; i < available_layer_count; ++i) {
                const VkLayerProperties* props = &properties[i];
                if (strcmp(props->layerName, "VK_LAYER_KHRONOS_validation") == 0) {
                    layer_names[layer_count++] = "VK_LAYER_KHRONOS_validation";
                } else if (strcmp(props->layerName, "VK_LAYER_LUNARG_standard_validation") == 0) {
                    layer_names[layer_count++] = "VK_LAYER_LUNARG_standard_validation";
                }
            }
        }
    }

    // Find debug extensions
    supports_EXT_debug_utils_ = false;
    if (debug) {
        for (uint32_t k = 0; k < layer_count; ++k) {
            uint32_t available_extension_count;
            VkResult result = vkEnumerateInstanceExtensionProperties(layer_names[k], &available_extension_count, NULL);
            if (result != VK_SUCCESS) {
                log_fatal("vkEnumerateInstanceExtensionProperties failed: %d", (int)result);
                abort();
            }
            if (available_extension_count > 0) {
                VkExtensionProperties properties[available_extension_count];
                result = vkEnumerateInstanceExtensionProperties(layer_names[k], &available_extension_count, properties);
                if (result != VK_SUCCESS) {
                    log_fatal("vkEnumerateInstanceExtensionProperties failed: %d", (int)result);
                    abort();
                }
                for (uint32_t i = 0; i < available_extension_count; ++i) {
                    const VkExtensionProperties* props = &properties[i];
                    if (strcmp(props->extensionName, VK_EXT_DEBUG_UTILS_EXTENSION_NAME) == 0) {
                        supports_EXT_debug_utils_ = true;
                    }
                }
            }
        }
    }

    if (supports_EXT_debug_utils_)
        extension_names[extension_count++] = VK_EXT_DEBUG_UTILS_EXTENSION_NAME;

    // Find surface extensions
    supports_KHR_surface_ = false;
    supports_KHR_display_ = false;
    {
        uint32_t available_extension_count;
        VkResult result = vkEnumerateInstanceExtensionProperties(NULL, &available_extension_count, NULL);
        if (result != VK_SUCCESS) {
            log_fatal("vkEnumerateInstanceExtensionProperties failed: %d", (int)result);
            abort();
        }
        if (available_extension_count > 0) {
            VkExtensionProperties properties[available_extension_count];
            result = vkEnumerateInstanceExtensionProperties(NULL, &available_extension_count, properties);
            if (result != VK_SUCCESS) {
                log_fatal("vkEnumerateInstanceExtensionProperties failed: %d", (int)result);
                abort();
            }
            for (uint32_t i = 0; i < available_extension_count; ++i) {
                const VkExtensionProperties* props = &properties[i];
                const char* extname = props->extensionName;
                if (strcmp(extname, VK_KHR_SURFACE_EXTENSION_NAME) == 0) {
                    supports_KHR_surface_ = true;
                    extension_names[extension_count++] = VK_KHR_SURFACE_EXTENSION_NAME;
                } else if (strcmp(extname, VK_KHR_DISPLAY_EXTENSION_NAME) == 0) {
                    supports_KHR_display_ = true;
                    extension_names[extension_count++] = VK_KHR_DISPLAY_EXTENSION_NAME;
                } else {
#ifdef VK_USE_PLATFORM_WAYLAND_KHR
                    if (strcmp(extname, VK_KHR_WAYLAND_SURFACE_EXTENSION_NAME) == 0) {
                        extension_names[extension_count++] = VK_KHR_WAYLAND_SURFACE_EXTENSION_NAME;
                    }
#endif
#ifdef VK_USE_PLATFORM_WIN32_KHR
                    if (strcmp(extname, VK_KHR_WIN32_SURFACE_EXTENSION_NAME) == 0) {
                        extension_names[extension_count++] = VK_KHR_WIN32_SURFACE_EXTENSION_NAME;
                    }
#endif
#ifdef VK_USE_PLATFORM_XCB_KHR
                    if (strcmp(extname, VK_KHR_XCB_SURFACE_EXTENSION_NAME) == 0) {
                        extension_names[extension_count++] = VK_KHR_XCB_SURFACE_EXTENSION_NAME;
                    }
#endif
#ifdef VK_USE_PLATFORM_XLIB_KHR
                    if (strcmp(extname, VK_KHR_XLIB_SURFACE_EXTENSION_NAME) == 0) {
                        extension_names[extension_count++] = VK_KHR_XLIB_SURFACE_EXTENSION_NAME;
                    }
#endif
                }
            }
        }
    }

    // Create instance
    const VkApplicationInfo appinfo = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pNext = NULL,
        .pApplicationName = appname,
        .applicationVersion = ENGINE_VERSION,
        .pEngineName = ENGINE_NAME,
        .engineVersion = ENGINE_VERSION,
        .apiVersion = VK_API_VERSION_1_1,
    };
    const VkInstanceCreateInfo createinfo = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pNext = NULL,
        .flags = 0,
        .pApplicationInfo = &appinfo,
        .enabledLayerCount = layer_count,
        .ppEnabledLayerNames = layer_names,
        .enabledExtensionCount = extension_count,
        .ppEnabledExtensionNames = extension_names,
    };

    VkResult result = vkCreateInstance(&createinfo, NULL, &instance_);
    if (result != VK_SUCCESS) {
        log_fatal("vkCreateInstance failed: %d", (int)result);
        abort();
    }
}

static VkBool32 cgvk_HandleDebugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT           messageSeverity,
    VkDebugUtilsMessageTypeFlagsEXT                  messageTypes,
    const VkDebugUtilsMessengerCallbackDataEXT*      pCallbackData,
    void*                                            pUserData)
{
    const char* message = pCallbackData->pMessage;
    switch (messageSeverity) {
        case VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT:
            log_info("[vk] %s", message);
            break;
        case VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT:
            log_warn("[vk] %s", message);
            break;
        case VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT:
            log_error("[vk] %s", message);
            break;
        default:
            break;
    }
    return VK_FALSE;
}

static void cgvk_init_debug_utils()
{
    const VkDebugUtilsMessengerCreateInfoEXT createinfo = {
        .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
        .pNext = NULL,
        .flags = 0,
        .messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT,
        .messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
        .pfnUserCallback = cgvk_HandleDebugCallback,
        .pUserData = NULL,
    };

    VkResult result = vkCreateDebugUtilsMessengerEXT(instance_, &createinfo, NULL, &debug_messenger_);
    if (result != VK_SUCCESS) {
        log_fatal("vkCreateDebugUtilsMessengerEXT failed: %d", (int)result);
        abort();
    }
}

static void cgvk_init_window(const char* title)
{
    int x = SDL_WINDOWPOS_CENTERED;
    int y = SDL_WINDOWPOS_CENTERED;
    int w = 640;
    int h = 480;
    uint32_t flags = SDL_WINDOW_SHOWN | SDL_WINDOW_ALLOW_HIGHDPI | SDL_WINDOW_VULKAN;
    window_ = SDL_CreateWindow(title, x, y, w, h, flags);
    if (!window_) {
        log_fatal("SDL_CreateWindow failed: %s", SDL_GetError());
        abort();
    }
}

static void cgvk_init_surface()
{
    if(!SDL_Vulkan_CreateSurface(window_, instance_, &surface_)) {
        log_fatal("SDL_Vulkan_CreateSurface failed: %s", SDL_GetError());
        abort();
    }
}

CGVK_API void cgvk_init(const char* appname)
{
    // Init SDL2
    if (SDL_Init(SDL_INIT_VIDEO)) {
        log_fatal("SDL_Init failed: %s", SDL_GetError());
        abort();
    }
    if (SDL_Vulkan_LoadLibrary(NULL)) {
        log_fatal("SDL_Vulkan_LoadLibrary failed: %s", SDL_GetError());
        abort();
    }

    // Init Instance
    PFN_vkGetInstanceProcAddr getprocaddr = (PFN_vkGetInstanceProcAddr)SDL_Vulkan_GetVkGetInstanceProcAddr();
    if (!getprocaddr) {
        log_fatal("SDL_Vulkan_GetVkGetInstanceProcAddr failed: %s", SDL_GetError());
        abort();
    }
    volkInitializeCustom(getprocaddr);
    cgvk_init_instance(appname, true);
    volkLoadInstance(instance_);
    if (supports_EXT_debug_utils_)
        cgvk_init_debug_utils();

    // Init window and surface
    cgvk_init_window(appname);
    cgvk_init_surface();

    // Init device
    cgvk_init_device(&dev_);
}

CGVK_API void cgvk_quit()
{
    // Destroy device
    cgvk_kill_device(&dev_);

    // Destroy surface and window
    if (surface_) {
        vkDestroySurfaceKHR(instance_, surface_, NULL);
        surface_ = NULL;
    }
    if (window_) {
        SDL_DestroyWindow(window_);
        window_ = NULL;
    }

    // Destroy Instance
    if (debug_messenger_) {
        vkDestroyDebugUtilsMessengerEXT(instance_, debug_messenger_, NULL);
        debug_messenger_ = NULL;
    }
    if (instance_) {
        vkDestroyInstance(instance_, NULL);
        instance_ = NULL;
    }

    // Cleanup SDL2
    SDL_Quit();
}

CGVK_API bool cgvk_pump_events()
{
    SDL_Event event;
    while (SDL_PollEvent(&event)) {
        switch (event.type) {
            case SDL_QUIT:
                return false;
            case SDL_WINDOWEVENT:
                break;
        }
    }
    return true;
}

CGVK_API void cgvk_render()
{
}

static VkInstance cgvk_get_instance()
{
    return instance_;
}

static VkSurfaceKHR cgvk_get_surface()
{
    return surface_;
}

static void cgvk_get_drawable_extent(VkExtent2D* extent)
{
    int w, h;
    SDL_Vulkan_GetDrawableSize(window_, &w, &h);
    extent->width = w;
    extent->height = h;
}
