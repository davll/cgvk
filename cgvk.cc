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
#include "uthash.h"
#include <SDL2/SDL.h>
#include <SDL2/SDL_vulkan.h>

#define CGVK_VERSION_MAJOR 0
#define CGVK_VERSION_MINOR 1
#define CGVK_VERSION_PATCH 0

#define CGVK_FRAME_LAG 3
#define CGVK_MAX_IMAGE_POOL_CAPACITY 1024
#define CGVK_MAX_SWAPCHAIN_IMAGE_COUNT 16
#define CGVK_MAX_SEMAPHORE_POOL_CAPACITY 16

// ============================================================================

enum cgvk_Format {
    CGVK_FORMAT_UNDEFINED = 0,
    CGVK_FORMAT_B8G8R8A8_UNORM = 1,
    CGVK_FORMAT_B8G8R8A8_SRGB = 2,
    CGVK_FORMAT_D24_UNORM_S8_UINT = 3,
    CGVK_FORMAT_MAX = 0xFF,
};

enum cgvk_ImageLayout {
    CGVK_IMAGE_LAYOUT_UNDEFINED = 0,
    CGVK_IMAGE_LAYOUT_TRANSFER_SRC = 1,
    CGVK_IMAGE_LAYOUT_TRANSFER_DST = 2,
    CGVK_IMAGE_LAYOUT_SHADER_READ = 3,
    CGVK_IMAGE_LAYOUT_COLOR_ATTACHMENT = 4,
    CGVK_IMAGE_LAYOUT_DEPTH_ATTACHMENT = 5,
    CGVK_IMAGE_LAYOUT_PRESENT_SRC = 6,
    CGVK_IMAGE_LAYOUT_MAX = 0xF,
};

// ============================================================================

struct cgvk_ImagePool;

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

struct cgvk_Image {
    const cgvk_Device* dev;
    cgvk_ImagePool* pool;
    VmaAllocation memory;
    VkImage image;
    VkImageView image_view;
    uint16_t id;
    uint16_t format: 8; // cgvk_Format
    uint16_t levels: 8;
    uint16_t layout: 4; // cgvk_ImageLayout
    uint16_t aspect_color: 1;
    uint16_t aspect_depth: 1;
    uint16_t aspect_stencil: 1;
    uint16_t width;
    uint16_t height;
    uint16_t depth;
    uint16_t layers;
};

struct cgvk_ImagePool {
    const cgvk_Device* dev;
    uint16_t total_count;
    uint16_t free_count;
    uint16_t free_indices[CGVK_MAX_IMAGE_POOL_CAPACITY];
    cgvk_Image images[CGVK_MAX_IMAGE_POOL_CAPACITY];
};

struct cgvk_Swapchain {
    const cgvk_Device* dev;
    VkSwapchainKHR swapchain;
    uint8_t format; // cgvk_Format
    uint8_t image_count;
    uint16_t width;
    uint16_t height;
    cgvk_Image* images[CGVK_MAX_SWAPCHAIN_IMAGE_COUNT];
};

struct cgvk_SemaphorePool {
    uint16_t total_count;
    uint16_t used_count;
    VkDevice device;
    VkSemaphore semaphores[CGVK_MAX_SEMAPHORE_POOL_CAPACITY];
};

struct cgvk_FrameList {
    const cgvk_Device* dev;
    const cgvk_Swapchain* swp;
    uint64_t frame_counter;
    uint8_t present_image_indices[CGVK_FRAME_LAG];
    VkSemaphore acquire_next_image_semaphores[CGVK_FRAME_LAG];
    cgvk_SemaphorePool semaphores[CGVK_FRAME_LAG];
};

// ============================================================================

static cgvk_Device dev_;
static cgvk_Swapchain swp_;
static cgvk_FrameList fl_;
static cgvk_ImagePool imgpool_;

static VkFormat cgvk_decode_format(cgvk_Format fmt);
static VkImageLayout cgvk_decode_image_layout(cgvk_ImageLayout l);

// ============================================================================

static void cgvk_release_image(cgvk_Image* img)
{
    if (img->image_view)
        vkDestroyImageView(img->dev->device, img->image_view, NULL);

    if (img->memory)
        vmaDestroyImage(img->dev->allocator, img->image, img->memory);

    if (img->pool)
        img->pool->free_indices[img->pool->free_count++] = img - img->pool->images;

    memset(img, 0, sizeof(cgvk_Image));
}

static cgvk_Image* cgvk_allocate_image(cgvk_ImagePool* pool)
{
    cgvk_Image* img = NULL;
    size_t id = SIZE_MAX;

    if (pool->free_count > 0) {
        id = pool->free_indices[--pool->free_count];
        img = &pool->images[id];
    } else {
        assert(pool->total_count < CGVK_MAX_IMAGE_POOL_CAPACITY);
        id = pool->total_count++;
        img = &pool->images[id];
    }

    memset(img, 0, sizeof(cgvk_Image));
    img->dev = pool->dev;
    img->pool = pool;
    img->id = id;

    return img;
}

static void cgvk_init_image_pool(const cgvk_Device* dev, cgvk_ImagePool* pool)
{
    memset(pool, 0, sizeof(cgvk_ImagePool));
    pool->dev = dev;
}

static void cgvk_kill_image_pool(cgvk_ImagePool* pool)
{
    for (int i = 0, n = pool->total_count; i < n; ++i) {
        if (!pool->images[i].pool)
            continue;
        pool->images[i].pool = NULL;
        cgvk_release_image(&pool->images[i]);
    }

    memset(pool, 0, sizeof(cgvk_ImagePool));
}

// ============================================================================

static VkSemaphore cgvk_allocate_semaphore(cgvk_SemaphorePool* pool)
{
    if (pool->used_count < pool->total_count) {
        return pool->semaphores[pool->used_count++];
    } else {
        assert(pool->used_count == pool->total_count);
        assert(pool->total_count < CGVK_MAX_SEMAPHORE_POOL_CAPACITY);
        VkSemaphoreCreateInfo create_info = {
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
            .pNext = NULL,
            .flags = 0,
        };
        VkSemaphore semaphore;
        VkResult result = vkCreateSemaphore(pool->device, &create_info, NULL, &semaphore);
        if (result != VK_SUCCESS) {
            log_fatal("vkCreateSemaphore failed: %d", (int)result);
            abort();
        }
        pool->semaphores[pool->total_count++] = semaphore;
        pool->used_count = pool->total_count;
        return semaphore;
    }
}

static void cgvk_reset_semaphore_pool(cgvk_SemaphorePool* pool)
{
    pool->used_count = 0;
}

static void cgvk_init_semaphore_pool(const cgvk_Device* dev, cgvk_SemaphorePool* pool)
{
    memset(pool, 0, sizeof(cgvk_SemaphorePool));
    pool->device = dev->device;
}

static void cgvk_kill_semaphore_pool(cgvk_SemaphorePool* pool)
{
    for (int i = 0, n = pool->total_count; i < n; ++i) {
        vkDestroySemaphore(pool->device, pool->semaphores[i], NULL);
    }

    memset(pool, 0, sizeof(cgvk_SemaphorePool));
}

// ============================================================================

static cgvk_SemaphorePool* cgvk_current_semaphore_pool(cgvk_FrameList* fl)
{
    size_t fidx = fl->frame_counter % CGVK_FRAME_LAG;
    return &fl->semaphores[fidx];
}

static cgvk_Image* cgvk_current_present_image(cgvk_FrameList* fl)
{
    size_t fidx = fl->frame_counter % CGVK_FRAME_LAG;
    return fl->swp->images[fl->present_image_indices[fidx]];
}

static void cgvk_begin_frame(cgvk_FrameList* fl)
{
    VkResult result;
    size_t fidx = fl->frame_counter % CGVK_FRAME_LAG;

    uint32_t image_index = -1;
    VkSemaphore semaphore = cgvk_allocate_semaphore(&fl->semaphores[fidx]);
    result = vkAcquireNextImageKHR(fl->dev->device, fl->swp->swapchain, UINT64_MAX, semaphore, VK_NULL_HANDLE, &image_index);
    if (result != VK_SUCCESS) {
        log_fatal("vkAcquireNextImageKHR failed: %d", (int)result);
        abort();
    }
    fl->present_image_indices[fidx] = image_index;
    fl->acquire_next_image_semaphores[fidx] = semaphore;
}

static void cgvk_end_frame(cgvk_FrameList* fl)
{
    VkResult result;
    size_t fidx = fl->frame_counter % CGVK_FRAME_LAG;

    VkSemaphore semaphore = fl->acquire_next_image_semaphores[fidx];
    uint32_t image_index = fl->present_image_indices[fidx];
    VkPresentInfoKHR present_info = {
        .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
        .pNext = NULL,
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &semaphore,
        .swapchainCount = 1,
        .pSwapchains = &fl->swp->swapchain,
        .pImageIndices = &image_index,
        .pResults = NULL,
    };
    result = vkQueuePresentKHR(fl->dev->present_queue, &present_info);
    if (result != VK_SUCCESS) {
        log_fatal("vkQueuePresentKHR failed: %d", (int)result);
        abort();
    }

    // clear cached data
    fl->acquire_next_image_semaphores[fidx] = VK_NULL_HANDLE;
    fl->present_image_indices[fidx] = -1;

    // ================ NEXT FRAME ==================

    assert(fl->frame_counter < UINT64_MAX);
    fl->frame_counter++;
    fidx = fl->frame_counter % CGVK_FRAME_LAG;

    // TODO: wait for fence
    // TODO: reset command pools
    cgvk_reset_semaphore_pool(&fl->semaphores[fidx]);
}

static void cgvk_init_frame_list(const cgvk_Device* dev, const cgvk_Swapchain* swp, cgvk_FrameList* fl)
{
    memset(fl, 0, sizeof(cgvk_FrameList));
    fl->dev = dev;
    fl->swp = swp;

    for (int i = 0; i < CGVK_FRAME_LAG; ++i)
        cgvk_init_semaphore_pool(dev, &fl->semaphores[i]);
}

static void cgvk_kill_frame_list(cgvk_FrameList* fl)
{
    for (int i = 0; i < CGVK_FRAME_LAG; ++i)
        cgvk_kill_semaphore_pool(&fl->semaphores[i]);
    
    memset(fl, 0, sizeof(cgvk_FrameList));
}

// ============================================================================

static const VkFormat format_mappings_[] = {
    VK_FORMAT_UNDEFINED,
    VK_FORMAT_B8G8R8A8_UNORM,
    VK_FORMAT_B8G8R8A8_SRGB,
    VK_FORMAT_D24_UNORM_S8_UINT,
};

static VkFormat cgvk_decode_format(cgvk_Format fmt)
{
    return format_mappings_[(size_t)fmt];
}

static const VkImageLayout image_layout_mappings_[] = {
    VK_IMAGE_LAYOUT_UNDEFINED,
    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
    VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
    VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
    VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
};

static VkImageLayout cgvk_decode_image_layout(cgvk_ImageLayout l)
{
    return image_layout_mappings_[(size_t)l];
}

// ============================================================================

static VkInstance cgvk_get_instance();
static VkSurfaceKHR cgvk_get_surface();
static void cgvk_get_drawable_extent(VkExtent2D* extent);

// ============================================================================

static VkPresentModeKHR cgvk_choose_present_mode(const cgvk_Device* dev, VkSurfaceKHR surface)
{
    VkResult result;
    VkPresentModeKHR present_mode = VK_PRESENT_MODE_FIFO_KHR;

    uint32_t present_mode_count;
    result = vkGetPhysicalDeviceSurfacePresentModesKHR(dev->gpu, surface, &present_mode_count, NULL);
    if (result != VK_SUCCESS) {
        log_fatal("vkGetPhysicalDeviceSurfacePresentModesKHR failed: %d", (int)result);
        abort();
    }

    VkPresentModeKHR present_modes[present_mode_count];
    result = vkGetPhysicalDeviceSurfacePresentModesKHR(dev->gpu, surface, &present_mode_count, present_modes);
    if (result != VK_SUCCESS) {
        log_fatal("vkGetPhysicalDeviceSurfacePresentModesKHR failed: %d", (int)result);
        abort();
    }

    for (int i = 0, n = present_mode_count; i < n; ++i) {
        if (present_modes[i] == VK_PRESENT_MODE_MAILBOX_KHR) {
            present_mode = VK_PRESENT_MODE_MAILBOX_KHR;
            break;
        }
    }

    return present_mode;
}

static void cgvk_choose_surface_format(
    const cgvk_Device* dev,
    VkSurfaceKHR surface,
    cgvk_Format* found_fmt,
    VkFormat* found_format,
    VkColorSpaceKHR* found_colorspace)
{
    VkResult result;
    cgvk_Format fmt = CGVK_FORMAT_UNDEFINED;
    VkFormat format = VK_FORMAT_UNDEFINED;
    VkColorSpaceKHR colorspace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;

    uint32_t format_count;
    result = vkGetPhysicalDeviceSurfaceFormatsKHR(dev->gpu, surface, &format_count, NULL);
    if (result != VK_SUCCESS) {
        log_fatal("vkGetPhysicalDeviceSurfaceFormatsKHR failed: %d", (int)result);
        abort();
    }
    if (format_count == 0) {
        log_fatal("The surface does not provide formats");
        abort();
    }
    
    VkSurfaceFormatKHR formats[format_count];
    result = vkGetPhysicalDeviceSurfaceFormatsKHR(dev->gpu, surface, &format_count, formats);
    if (result != VK_SUCCESS) {
        log_fatal("vkGetPhysicalDeviceSurfaceFormatsKHR failed: %d", (int)result);
        abort();
    }

    if (format_count == 1 && formats[0].format == VK_FORMAT_UNDEFINED) {
        fmt = CGVK_FORMAT_B8G8R8A8_SRGB;
        format = VK_FORMAT_B8G8R8A8_SRGB;
    } else {
        for (int i = 0, n = format_count; i < n; ++i) {
            const VkSurfaceFormatKHR* f = &formats[i];
            if (f->format == VK_FORMAT_B8G8R8A8_SRGB) {
                fmt = CGVK_FORMAT_B8G8R8A8_SRGB;
                format = f->format;
                colorspace = f->colorSpace;
                break;
            }
        }
    }

    if (format == VK_FORMAT_UNDEFINED) {
        log_fatal("Cannot find valid surface format");
        abort();
    }

    *found_fmt = fmt;
    *found_format = format;
    *found_colorspace = colorspace;
}

static void cgvk_init_swapchain(const cgvk_Device* dev, cgvk_ImagePool* imgpool, cgvk_Swapchain* swp)
{
    VkResult result;

    memset(swp, 0, sizeof(cgvk_Swapchain));

    VkSurfaceKHR surface = cgvk_get_surface();

    // Query surface capabilites
    VkSurfaceCapabilitiesKHR caps;
    result = vkGetPhysicalDeviceSurfaceCapabilitiesKHR(dev->gpu, surface, &caps);
    if (result != VK_SUCCESS) {
        log_fatal("vkGetPhysicalDeviceSurfaceCapabilitiesKHR failed: %d", (int)result);
        abort();
    }

    // Select image count
    uint32_t image_count = 3;
    if (image_count < caps.minImageCount)
        image_count = caps.minImageCount;
    if (caps.maxImageCount != 0 && image_count > caps.maxImageCount)
        image_count = caps.maxImageCount;

    // Select image extent
    VkExtent2D extent = caps.currentExtent;
    if (extent.width == UINT32_MAX || extent.height == UINT32_MAX)
        cgvk_get_drawable_extent(&extent);

    // Select present mode
    VkPresentModeKHR present_mode = cgvk_choose_present_mode(dev, surface);
    
    // Select image format
    cgvk_Format fmt;
    VkFormat format;
    VkColorSpaceKHR colorspace;
    cgvk_choose_surface_format(dev, surface, &fmt, &format, &colorspace);

    // create swapchain
    VkSwapchainCreateInfoKHR create_info = {
        .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
        .pNext = NULL,
        .flags = 0,
        .surface = surface,
        .minImageCount = image_count,
        .imageFormat = format,
        .imageColorSpace = colorspace,
        .imageExtent = extent,
        .imageArrayLayers = 1,
        .imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
        .imageSharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = NULL,
        .preTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR,
        .compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
        .presentMode = present_mode,
        .clipped = VK_TRUE,
        .oldSwapchain = VK_NULL_HANDLE,
    };
    VkSwapchainKHR swapchain;
    result = vkCreateSwapchainKHR(dev->device, &create_info, NULL, &swapchain);
    if (result != VK_SUCCESS) {
        log_fatal("vkCreateSwapchainKHR failed: %d", (int)result);
        abort();
    }

    // Get actual image count
    result = vkGetSwapchainImagesKHR(dev->device, swapchain, &image_count, NULL);
    if (result != VK_SUCCESS) {
        log_fatal("vkGetSwapchainImagesKHR failed: %d", (int)result);
        abort();
    }
    if (image_count > CGVK_MAX_SWAPCHAIN_IMAGE_COUNT) {
        log_fatal("The number of the images of the swapchain exceeds the limit: %u > %u", image_count, CGVK_MAX_SWAPCHAIN_IMAGE_COUNT);
        abort();
    }

    // Get images
    VkImage images[image_count];
    result = vkGetSwapchainImagesKHR(dev->device, swapchain, &image_count, images);
    if (result != VK_SUCCESS) {
        log_fatal("vkGetSwapchainImagesKHR failed: %d", (int)result);
        abort();
    }

    // Create image views
    VkImageView image_views[image_count];
    for (int i = 0, n = image_count; i < n; ++i) {
        VkImageViewCreateInfo create_info = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .pNext = NULL,
            .flags = 0,
            .image = images[i],
            .viewType = VK_IMAGE_VIEW_TYPE_2D,
            .format = format,
            .components = {
                .r = VK_COMPONENT_SWIZZLE_R,
                .g = VK_COMPONENT_SWIZZLE_G,
                .b = VK_COMPONENT_SWIZZLE_B,
                .a = VK_COMPONENT_SWIZZLE_A,
            },
            .subresourceRange = {
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
        };
        result = vkCreateImageView(dev->device, &create_info, NULL, &image_views[i]);
        if (result != VK_SUCCESS) {
            log_fatal("vkCreateImageView failed: %d", (int)result);
            abort();
        }
    }

    // Fill the fields
    swp->dev = dev;
    swp->swapchain = swapchain;
    swp->format = fmt;
    swp->image_count = image_count;
    swp->width = extent.width;
    swp->height = extent.height;

    // Init images
    for (int i = 0, n = image_count; i < n; ++i) {
        cgvk_Image* img = cgvk_allocate_image(imgpool);
        img->image = images[i];
        img->image_view = image_views[i];
        img->format = (uint8_t)fmt;
        img->width = extent.width;
        img->height = extent.height;
        img->depth = 1;
        img->levels = 1;
        img->layers = 1;
        img->layout = CGVK_IMAGE_LAYOUT_UNDEFINED;
        img->aspect_color = 1;
        swp->images[i] = img;
    }
}

static void cgvk_kill_swapchain(cgvk_Swapchain* swp)
{
    for (int i = 0, n = swp->image_count; i < n; ++i) {
        cgvk_release_image(swp->images[i]);
    }

    if (swp->swapchain) {
        vkDestroySwapchainKHR(swp->dev->device, swp->swapchain, NULL);
    }

    memset(swp, 0, sizeof(cgvk_Swapchain));
}

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

static void cgvk_choose_queue_families(
    VkPhysicalDevice gpu,
    VkSurfaceKHR surface,
    uint32_t* found_main_queue_family,
    uint32_t* found_present_queue_family)
{
    VkResult result;
    uint32_t main_queue_family = UINT32_MAX;
    uint32_t present_queue_family = UINT32_MAX;

    uint32_t queue_family_count;
    vkGetPhysicalDeviceQueueFamilyProperties(gpu, &queue_family_count, NULL);
    
    VkQueueFamilyProperties properties[queue_family_count];
    vkGetPhysicalDeviceQueueFamilyProperties(gpu, &queue_family_count, properties);

    VkBool32 supported[queue_family_count];
    for (int i = 0, n = queue_family_count; i < n; ++i) {
        result = vkGetPhysicalDeviceSurfaceSupportKHR(gpu, i, surface, &supported[i]);
        if (result != VK_SUCCESS) {
            log_fatal("vkGetPhysicalDeviceSurfaceSupportKHR failed: %d", (int)result);
            abort();
        }
    }
    for (uint32_t i = 0, n = queue_family_count; i < n; ++i) {
        const VkQueueFamilyProperties* p = &properties[i];
        if (p->queueCount == 0)
            continue;
        VkQueueFlags mask = VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT;
        if (main_queue_family == UINT32_MAX && (p->queueFlags & mask) == mask)
            main_queue_family = i;
        if (present_queue_family == UINT32_MAX && supported[i])
            present_queue_family = i;
    }

    *found_main_queue_family = main_queue_family;
    *found_present_queue_family = present_queue_family;
}

static void cgvk_init_allocator(cgvk_Device* dev)
{
    VmaVulkanFunctions functions = {
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
    VmaAllocatorCreateInfo create_info = {
        .flags = 0,
        .physicalDevice = dev->gpu,
        .device = dev->device,
        .preferredLargeHeapBlockSize = 0,
        .pAllocationCallbacks = NULL,
        .pDeviceMemoryCallbacks = NULL,
        .frameInUseCount = 0,
        .pHeapSizeLimit = NULL,
        .pVulkanFunctions = &functions,
        .pRecordSettings = NULL,
        .instance = cgvk_get_instance(),
        .vulkanApiVersion = VK_API_VERSION_1_1,
    };
    VkResult result = vmaCreateAllocator(&create_info, &dev->allocator);
    if (result != VK_SUCCESS) {
        log_fatal("vmaCreateAllocator failed: %d", (int)result);
        abort();
    }
}

static void cgvk_init_device(cgvk_Device* dev)
{
    VkResult result;

    VkSurfaceKHR surface = cgvk_get_surface();

    uint32_t extension_count = 0;
    const char* extension_names[4];
    VkPhysicalDeviceFeatures2 features, avail_features;

    memset(dev, 0, sizeof(cgvk_Device));


    // Choose GPU
    VkPhysicalDevice gpu = cgvk_choose_gpu();
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
            for (uint32_t i = 0, n = available_extension_count; i < n; ++i) {
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
    uint32_t main_queue_family, present_queue_family;
    cgvk_choose_queue_families(gpu, surface, &main_queue_family, &present_queue_family);
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
    VkDeviceCreateInfo create_info = {
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
    VkDevice device;
    result = vkCreateDevice(gpu, &create_info, NULL, &device);
    if (result != VK_SUCCESS) {
        log_fatal("vkCreateDevice failed: %d", (int)result);
        abort();
    }
    dev->device = device;

    // Get queues
    vkGetDeviceQueue(device, main_queue_family, 0, &dev->main_queue);
    vkGetDeviceQueue(device, present_queue_family, 0, &dev->present_queue);

    // Init allocator
    cgvk_init_allocator(dev);
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
            for (uint32_t i = 0, n = available_layer_count; i < n; ++i) {
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
                for (uint32_t i = 0, n = available_extension_count; i < n; ++i) {
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
            for (uint32_t i = 0, n = available_extension_count; i < n; ++i) {
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

    // Init image pool
    cgvk_init_image_pool(&dev_, &imgpool_);

    // Init swapchain
    cgvk_init_swapchain(&dev_, &imgpool_, &swp_);

    // Init frame list
    cgvk_init_frame_list(&dev_, &swp_, &fl_);
}

CGVK_API void cgvk_quit()
{
    // Destroy frame list
    cgvk_kill_frame_list(&fl_);

    // Destroy swapchain
    cgvk_kill_swapchain(&swp_);

    // Destroy image pool
    cgvk_kill_image_pool(&imgpool_);

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
    cgvk_begin_frame(&fl_);
    cgvk_end_frame(&fl_);
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
