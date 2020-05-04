// CGVK Implementation

// ============================================================================

#include "cgvk.h"
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "volk.h"
#include "vk_mem_alloc.h"
#include "log.h"
#include "uthash.h"
#include "xxhash.h"
#include <SDL2/SDL.h>
#include <SDL2/SDL_vulkan.h>

#define CGVK_VERSION_MAJOR 0
#define CGVK_VERSION_MINOR 1
#define CGVK_VERSION_PATCH 0

#define CGVK_FRAME_LAG 3

// ============================================================================

typedef enum cgvk_Format {
    CGVK_FORMAT_UNDEFINED = 0,
    CGVK_FORMAT_B8G8R8A8_UNORM = 1,
    CGVK_FORMAT_B8G8R8A8_SRGB = 2,
    CGVK_FORMAT_D24_UNORM_S8_UINT = 3,
    CGVK_FORMAT_MAX = 0xFF,
} cgvk_Format;
typedef uint8_t cgvk_PackedFormat;

typedef enum cgvk_ImageLayout {
    CGVK_IMAGE_LAYOUT_UNDEFINED = 0,
    CGVK_IMAGE_LAYOUT_TRANSFER_SRC = 1,
    CGVK_IMAGE_LAYOUT_TRANSFER_DST = 2,
    CGVK_IMAGE_LAYOUT_SHADER_READ = 3,
    CGVK_IMAGE_LAYOUT_COLOR_ATTACHMENT = 4,
    CGVK_IMAGE_LAYOUT_DEPTH_ATTACHMENT = 5,
    CGVK_IMAGE_LAYOUT_PRESENT_SRC = 6,
    CGVK_IMAGE_LAYOUT_MAX = 0xF,
} cgvk_ImageLayout;
typedef uint8_t cgvk_PackedImageLayout;

typedef enum cgvk_SampleCount {
    CGVK_SAMPLE_COUNT_1 = 0,
    CGVK_SAMPLE_COUNT_2 = 1,
    CGVK_SAMPLE_COUNT_4 = 2,
    CGVK_SAMPLE_COUNT_8 = 3,
    CGVK_SAMPLE_COUNT_16 = 4,
    CGVK_SAMPLE_COUNT_32 = 5,
    CGVK_SAMPLE_COUNT_64 = 6,
    CGVK_SAMPLE_COUNT_128 = 7,
    CGVK_SAMPLE_COUNT_MAX = 0x7,
} cgvk_SampleCount;
typedef uint8_t cgvk_PackedSampleCount;

// ============================================================================

typedef struct cgvk_Device {
    VkPhysicalDevice gpu;
    VkDevice device;
    VmaAllocator allocator;
    VkQueue main_queue;
    VkQueue present_queue;
    bool supports_KHR_swapchain;
    uint32_t main_queue_family;
    uint32_t present_queue_family;
} cgvk_Device;

typedef struct cgvk_TransientBlock {
    VkBuffer buffer;
    uint32_t offset;
    uint32_t size;
    void* mapped;
} cgvk_TransientBlock;

typedef struct cgvk_TransientBuffer {
    VmaAllocation memory;
    VkBuffer buffer;
    uint32_t tag;
    uint32_t used;
    VkMemoryPropertyFlags properties;
    void* mapped;
    struct cgvk_TransientBuffer* next;
} cgvk_TransientBuffer;

typedef struct cgvk_TransientAllocator {
    const cgvk_Device* dev;
    uint32_t buffer_size;
    uint16_t capacity;
    uint16_t count;
    VkBufferUsageFlags usage;
    cgvk_TransientBuffer* pending;
    cgvk_TransientBuffer* submitted;
    cgvk_TransientBuffer* free;
    cgvk_TransientBuffer buffers[];
} cgvk_TransientAllocator;

typedef struct cgvk_SwapchainImage {
    VkImage image;
    VkImageView image_view;
} cgvk_SwapchainImage;

typedef struct cgvk_Swapchain {
    const cgvk_Device* dev;
    VkSwapchainKHR swapchain;
    cgvk_Format format;
    uint16_t width;
    uint16_t height;
    uint8_t image_count;
    cgvk_SwapchainImage images[];
} cgvk_Swapchain;

typedef struct cgvk_Shader {
    XXH64_hash_t key;
    VkShaderModule module;
    struct cgvk_Shader* next;
    struct cgvk_Shader* prev;
    UT_hash_handle hh;
} cgvk_Shader;

typedef struct cgvk_ShaderCache {
    const cgvk_Device* dev;
    cgvk_Shader* head;
    cgvk_Shader* oldest;
    cgvk_Shader* newest;
    uint32_t capacity;
    uint32_t count;
    cgvk_Shader objects[];
} cgvk_ShaderCache;

typedef struct cgvk_Frame {
    uint32_t present_image_index;
    VkFence fence;
    VkSemaphore acquire_next_image_semaphore;
    VkSemaphore execute_commands_semaphore;
    VkCommandPool main_command_pool;
} cgvk_Frame;

typedef struct cgvk_Renderer {
    const cgvk_Device* dev;
    const cgvk_Swapchain* swp;
    uint64_t current_frame;
    uint16_t frame_count;
    VkRenderPass render_pass;
    VkPipelineLayout empty_pipeline_layout;
    VkPipeline hello_triangle_pipeline;
    VkFramebuffer* framebuffers;
    cgvk_TransientAllocator* transient_vertex_buffers;
    cgvk_TransientAllocator* transient_uniform_buffers;
    cgvk_Frame* frames;
} cgvk_Renderer;

// ============================================================================

static VkFormat cgvk_decode_format(cgvk_Format fmt);
static VkImageLayout cgvk_decode_image_layout(cgvk_ImageLayout l);

// ============================================================================

static void cgvk_map_transient_buffer(cgvk_TransientAllocator* talloc, cgvk_TransientBuffer* tbuf)
{
    assert(!tbuf->mapped);

    VkResult result = vmaMapMemory(talloc->dev->allocator, tbuf->memory, &tbuf->mapped);
    if (result != VK_SUCCESS) {
        log_fatal("vmaMapMemory failed: %d", (int)result);
        abort();
    }
}

static void cgvk_unmap_transient_buffer(cgvk_TransientAllocator* talloc, cgvk_TransientBuffer* tbuf)
{
    assert(tbuf->mapped);

    VmaAllocator allocator = talloc->dev->allocator;

    if (!(tbuf->properties & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
        vmaFlushAllocation(allocator, tbuf->memory, 0, tbuf->used);
    }

    vmaUnmapMemory(allocator, tbuf->memory);
    tbuf->mapped = NULL;
}

static cgvk_TransientBuffer* cgvk_new_transient_buffer(cgvk_TransientAllocator* talloc)
{
    assert(talloc->count < talloc->capacity);

    VmaAllocator allocator = talloc->dev->allocator;

    cgvk_TransientBuffer* tbuf;

    // intialize structure
    tbuf = &talloc->buffers[talloc->count++];
    memset(tbuf, 0, sizeof(cgvk_TransientBuffer));
    tbuf->mapped = NULL;
    tbuf->next = NULL;

    VkBufferCreateInfo create_info = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .pNext = NULL,
        .flags = 0,
        .size = talloc->buffer_size,
        .usage = talloc->usage,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = NULL,
    };

    VmaAllocationCreateInfo alloc_info = {
        .flags = VMA_ALLOCATION_CREATE_STRATEGY_BEST_FIT_BIT,
        .usage = VMA_MEMORY_USAGE_CPU_TO_GPU,
        .requiredFlags = 0,
        .preferredFlags = 0,
        .memoryTypeBits = 0,
        .pool = VK_NULL_HANDLE,
        .pUserData = NULL,
    };

    VmaAllocationInfo info;
    VkResult result = vmaCreateBuffer(allocator, &create_info, &alloc_info, &tbuf->buffer, &tbuf->memory, &info);
    if (result != VK_SUCCESS) {
        log_fatal("vmaCreateBuffer failed: %d", (int)result);
        abort();
    }

    uint32_t mem_type = info.memoryType;
    vmaGetMemoryTypeProperties(allocator, mem_type, &tbuf->properties);

    return tbuf;
}

static void cgvk_reset_transient_buffers(cgvk_TransientAllocator* talloc, uint32_t tag)
{
    cgvk_TransientBuffer *node = talloc->submitted, *head = NULL, *tmp;
    while (node) {
        tmp = node->next;
        if (node->tag == tag) {
            node->tag = 0;
            node->used = 0;
            // push to free list
            node->next = talloc->free;
            talloc->free = node;
        } else {
            // keep the node
            node->next = head;
            head = node;
        }
        node = tmp;
    }
    talloc->submitted = head;
}

static void cgvk_submit_transient_buffers(cgvk_TransientAllocator* talloc, uint32_t tag)
{
    cgvk_TransientBuffer *node = talloc->pending, *tmp;
    while (node) {
        tmp = node->next;
        {
            // change mode
            cgvk_unmap_transient_buffer(talloc, node);
            node->tag = tag;
            // push to submitted list
            node->next = talloc->submitted;
            talloc->submitted = node;
        }
        node = tmp;
    }

    // the pending list is empty now
    talloc->pending = NULL;
}

static void cgvk_allocate_transient_block(cgvk_TransientAllocator* talloc, size_t size, cgvk_TransientBlock* block)
{
    assert(size <= talloc->buffer_size);

    cgvk_TransientBuffer *node;

    // find available space in pending list
    for (node = talloc->pending; node; node = node->next) {
        if (node->used + size <= talloc->buffer_size) {
            // found
            break;
        }
    }
    // node = NULL if not found

    // create new pending 
    if (!node) {
        if (talloc->free) {
            // extract from free list
            node = talloc->free;
            talloc->free = node->next;
            node->next = NULL;
        } else {
            node = cgvk_new_transient_buffer(talloc);
        }

        cgvk_map_transient_buffer(talloc, node);

        // push to pending list
        node->next = talloc->pending;
        talloc->pending = node;
    }

    // fill the return values
    block->buffer = node->buffer;
    block->offset = node->used;
    block->size = (uint32_t)size;
    block->mapped = (uint8_t*)node->mapped + node->used;

    // update the buffer node
    node->used += (uint32_t)size;
}

static cgvk_TransientAllocator* cgvk_new_transient_allocator(const cgvk_Device* dev, uint32_t buffer_size, uint16_t capacity, VkBufferUsageFlags usage)
{
    size_t size = sizeof(cgvk_TransientAllocator) + sizeof(cgvk_TransientBuffer) * capacity;

    cgvk_TransientAllocator* talloc = (cgvk_TransientAllocator*)malloc(size);

    // initialize
    memset(talloc, 0, size);
    talloc->dev = dev;
    talloc->buffer_size = buffer_size;
    talloc->capacity = capacity;
    talloc->free = NULL;
    talloc->pending = NULL;
    talloc->submitted = NULL;
    talloc->usage = usage;

    return talloc;
}

static void cgvk_free_transient_allocator(cgvk_TransientAllocator* talloc)
{
    VmaAllocator allocator = talloc->dev->allocator;

    // destroy all buffers and allocations
    for (unsigned i = 0, n = talloc->count; i < n; ++i) {
        cgvk_TransientBuffer* tbuf = &talloc->buffers[i];
        vmaDestroyBuffer(allocator, tbuf->buffer, tbuf->memory);
    }

    free(talloc);
}

// ============================================================================

static void cgvk_touch_shader(cgvk_ShaderCache* cache, cgvk_Shader* sh)
{
    if (sh == cache->newest)
        return;
    
    assert(cache->oldest);
    assert(cache->newest);
    assert(cache->oldest != cache->newest);

    // purge the node from the list
    if (sh->prev) {
        sh->prev->next = sh->next;
        sh->next->prev = sh->prev;
    } else {
        assert(sh == cache->oldest);
        cache->oldest = cache->oldest->next;
        cache->oldest->prev = NULL;
    }

    // push the node to the tail of the list
    sh->next = NULL;
    sh->prev = cache->newest;
    cache->newest->next = sh;
    cache->newest = sh;
}

static cgvk_Shader* cgvk_insert_shader(cgvk_ShaderCache* cache, XXH64_hash_t key, size_t size, const void* code)
{
    VkDevice device = cache->dev->device;
    cgvk_Shader* sh;

    if (cache->count < cache->capacity) {
        // allocate new space
        sh = &cache->objects[cache->count++];
    } else {
        // purge and destroy the oldest shader
        assert(cache->oldest);
        assert(cache->oldest != cache->newest);
        assert(cache->oldest->next);
        sh = cache->oldest;
        cache->oldest = sh->next;
        cache->oldest->prev = NULL;
        sh->next = NULL;
        vkDestroyShaderModule(device, sh->module, NULL);
    }
    
    // initialize
    memset(sh, 0, sizeof(cgvk_Shader));
    memcpy(&sh->key, &key, sizeof(XXH64_hash_t));

    // add to the table
    HASH_ADD(hh, cache->head, key, sizeof(XXH64_hash_t), sh);

    // push to the tail of the list
    sh->prev = cache->newest;
    if (cache->newest)
        cache->newest->next = sh;
    else
        cache->oldest = sh;
    cache->newest = sh;

    VkShaderModuleCreateInfo create_info = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .pNext = NULL,
        .flags = 0,
        .codeSize = size,
        .pCode = (const uint32_t*)code,
    };

    VkResult result = vkCreateShaderModule(device, &create_info, NULL, &sh->module);
    if (result != VK_SUCCESS) {
        log_fatal("vkCreateShaderModule failed: %d", (int)result);
        abort();
    }

    return sh;
}

static cgvk_Shader* cgvk_get_shader_by_code(cgvk_ShaderCache* cache, size_t size, const void* code)
{
    XXH64_hash_t key = XXH64(code, size, 0);
    cgvk_Shader* sh = NULL;
    HASH_FIND(hh, cache->head, &key, sizeof(XXH64_hash_t), sh);

    if (sh) {
        cgvk_touch_shader(cache, sh);
    } else {
        sh = cgvk_insert_shader(cache, key, size, code);
    }

    return sh;
}

static cgvk_ShaderCache* cgvk_new_shader_cache(const cgvk_Device* dev, uint32_t capacity)
{
    size_t size = sizeof(cgvk_ShaderCache) + sizeof(cgvk_Shader) * capacity;

    cgvk_ShaderCache* cache = (cgvk_ShaderCache*)malloc(size);

    // initialize
    memset(cache, 0, size);
    cache->dev = dev;
    cache->capacity = capacity;
    cache->head = NULL;
    cache->oldest = NULL;
    cache->newest = NULL;

    return cache;
}

static void cgvk_free_shader_cache(cgvk_ShaderCache* cache)
{
    VkDevice device = cache->dev->device;

    // destroy all shader modules
    for (unsigned i = 0, n = cache->count; i < n; ++i) {
        if (cache->objects[i].module)
            vkDestroyShaderModule(device, cache->objects[i].module, NULL);
    }

    free(cache);
}

// ============================================================================

static VkRenderPass cgvk_new_render_pass(const cgvk_Device* dev, const cgvk_Swapchain* swp)
{
    VkFormat present_format = cgvk_decode_format(swp->format);

    uint32_t attachment_count = 1;
    VkAttachmentDescription attachments[1] = {
        {
            .flags = 0,
            .format = present_format,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
            .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
            .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
        },
    };

    uint32_t color_count = 1;
    VkAttachmentReference color_refs[] = {
        {
            .attachment = 0,
            .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        },
    };

    uint32_t subpass_count = 1;
    VkSubpassDescription subpasses[1] = {
        {
            .flags = 0,
            .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
            .inputAttachmentCount = 0,
            .pInputAttachments = NULL,
            .colorAttachmentCount = color_count,
            .pColorAttachments = color_refs,
            .pResolveAttachments = NULL,
            .pDepthStencilAttachment = NULL,
            .preserveAttachmentCount = 0,
            .pPreserveAttachments = NULL,
        },
    };

    uint32_t dependency_count = 1;
    VkSubpassDependency dependencies[1] = {
        {
            .srcSubpass = VK_SUBPASS_EXTERNAL,
            .dstSubpass = 0,
            .srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            .dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            .srcAccessMask = 0,
            .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
            .dependencyFlags = 0,
        },
    };

    VkRenderPassCreateInfo create_info = {
        .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
        .pNext = NULL,
        .flags = 0,
        .attachmentCount = attachment_count,
        .pAttachments = attachments,
        .subpassCount = subpass_count,
        .pSubpasses = subpasses,
        .dependencyCount = dependency_count,
        .pDependencies = dependencies,
    };

    VkRenderPass render_pass;

    VkResult result = vkCreateRenderPass(dev->device, &create_info, NULL, &render_pass);
    if (result != VK_SUCCESS) {
        log_fatal("vkCreateRenderPass failed");
        abort();
    }

    return render_pass;
}

static VkPipelineLayout cgvk_new_empty_pipeline_layout(const cgvk_Device* dev)
{
    VkPipelineLayoutCreateInfo create_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .pNext = NULL,
        .flags = 0,
        .setLayoutCount = 0,
        .pSetLayouts = NULL,
        .pushConstantRangeCount = 0,
        .pPushConstantRanges = NULL,
    };

    VkPipelineLayout pipeline_layout;

    VkResult result = vkCreatePipelineLayout(dev->device, &create_info, NULL, &pipeline_layout);
    if (result != VK_SUCCESS) {
        log_fatal("vkCreatePipelineLayout failed: %d", (int)result);
        abort();
    }

    return pipeline_layout;
}

static VkPipeline cgvk_new_hello_triangle_pipeline(const cgvk_Device* dev, VkPipelineLayout pipeline_layout, VkRenderPass render_pass)
{
    static const uint32_t vert_code[] = {
        #include "triangle.vert.inc"
    };
    static const uint32_t frag_code[] = {
        #include "triangle.frag.inc"
    };

    cgvk_ShaderCache* cache = cgvk_new_shader_cache(dev, 2);

    cgvk_Shader* vs = cgvk_get_shader_by_code(cache, sizeof(vert_code), vert_code);
    cgvk_Shader* fs = cgvk_get_shader_by_code(cache, sizeof(frag_code), frag_code);

    VkPipelineShaderStageCreateInfo stages[2] = {
        {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .pNext = NULL,
            .flags = 0,
            .stage = VK_SHADER_STAGE_VERTEX_BIT,
            .module = vs->module,
            .pName = "main",
            .pSpecializationInfo = NULL,
        },
        {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .pNext = NULL,
            .flags = 0,
            .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
            .module = fs->module,
            .pName = "main",
            .pSpecializationInfo = NULL,
        },
    };

    VkVertexInputBindingDescription vbs[1] = {
        {
            .binding = 0,
            .stride = 16,
            .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
        },
    };

    VkVertexInputAttributeDescription vas[2] = {
        {
            .location = 0,
            .binding = 0,
            .format = VK_FORMAT_R32G32B32_SFLOAT,
            .offset = 0,
        },
        {
            .location = 1,
            .binding = 0,
            .format = VK_FORMAT_B8G8R8A8_UNORM,
            .offset = 12,
        },
    };

    VkPipelineVertexInputStateCreateInfo vi = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        .pNext = NULL,
        .vertexBindingDescriptionCount = 1,
        .pVertexBindingDescriptions = vbs,
        .vertexAttributeDescriptionCount = 2,
        .pVertexAttributeDescriptions = vas,
    };

    VkPipelineInputAssemblyStateCreateInfo ia = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .pNext = NULL,
        .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
        .primitiveRestartEnable = VK_FALSE,
    };

    VkPipelineViewportStateCreateInfo vps = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .pNext = NULL,
        .viewportCount = 1,
        .pViewports = NULL,
        .scissorCount = 1,
        .pScissors = NULL,
    };

    VkPipelineRasterizationStateCreateInfo rs = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .pNext = NULL,
        .depthClampEnable = VK_FALSE,
        .rasterizerDiscardEnable = VK_FALSE,
        .polygonMode = VK_POLYGON_MODE_FILL,
        .cullMode = VK_CULL_MODE_BACK_BIT,
        .frontFace = VK_FRONT_FACE_CLOCKWISE,
        .depthBiasEnable = VK_FALSE,
        .lineWidth = 1.0f,
    };

    VkPipelineMultisampleStateCreateInfo ms = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .pNext = NULL,
        .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
        .sampleShadingEnable = VK_FALSE,
        .pSampleMask = NULL,
        .alphaToCoverageEnable = VK_FALSE,
        .alphaToOneEnable = VK_FALSE,
    };

    VkPipelineDepthStencilStateCreateInfo dss = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
        .pNext = NULL,
        .depthTestEnable = VK_FALSE,
        .depthWriteEnable = VK_FALSE,
        .depthBoundsTestEnable = VK_FALSE,
        .stencilTestEnable = VK_FALSE,
    };

    VkPipelineColorBlendAttachmentState bas[] = {
        {
            .blendEnable = VK_FALSE,
            .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
        },
    };

    VkPipelineColorBlendStateCreateInfo bs = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .pNext = NULL,
        .logicOpEnable = VK_FALSE,
        .attachmentCount = 1,
        .pAttachments = bas,
    };

    VkDynamicState dyss[] = {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR,
    };

    VkPipelineDynamicStateCreateInfo dys = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        .pNext = NULL,
        .dynamicStateCount = 2,
        .pDynamicStates = dyss,
    };

    VkGraphicsPipelineCreateInfo create_info = {
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .pNext = NULL,
        .flags = 0,
        .stageCount = 2,
        .pStages = stages,
        .pVertexInputState = &vi,
        .pInputAssemblyState = &ia,
        .pTessellationState = NULL,
        .pViewportState = &vps,
        .pRasterizationState = &rs,
        .pMultisampleState = &ms,
        .pDepthStencilState = &dss,
        .pColorBlendState = &bs,
        .pDynamicState = &dys,
        .layout = pipeline_layout,
        .renderPass = render_pass,
        .subpass = 0,
        .basePipelineHandle = VK_NULL_HANDLE,
        .basePipelineIndex = 0,
    };

    VkPipeline pipeline;

    VkResult result = vkCreateGraphicsPipelines(dev->device, VK_NULL_HANDLE, 1, &create_info, NULL, &pipeline);
    if (result != VK_SUCCESS) {
        log_fatal("vkCreateGraphicsPipelines failed: %d", (int)result);
        abort();
    }

    cgvk_free_shader_cache(cache);

    return pipeline;
}

static VkFramebuffer* cgvk_new_framebuffers(const cgvk_Device* dev, const cgvk_Swapchain* swp, VkRenderPass render_pass)
{
    VkFramebuffer* framebuffers = (VkFramebuffer*)malloc(sizeof(VkFramebuffer) * swp->image_count);

    for (int i = 0, n = swp->image_count; i < n; ++i) {
        uint32_t attachment_count = 1;
        VkImageView attachments[1] = {
            swp->images[i].image_view,
        };

        VkFramebufferCreateInfo create_info = {
            .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
            .pNext = NULL,
            .flags = 0,
            .renderPass = render_pass,
            .attachmentCount = attachment_count,
            .pAttachments = attachments,
            .width = swp->width,
            .height = swp->height,
            .layers = 1,
        };

        VkResult result = vkCreateFramebuffer(dev->device, &create_info, NULL, &framebuffers[i]);
        if (result != VK_SUCCESS) {
            log_fatal("vkCreateFramebuffer failed: %d", (int)result);
            abort();
        }
    }

    return framebuffers;
}

static cgvk_Frame* cgvk_new_frames(const cgvk_Device* dev, uint32_t frame_count)
{
    cgvk_Frame* frames = (cgvk_Frame*)malloc(sizeof(cgvk_Frame) * frame_count);
    memset(frames, 0, sizeof(cgvk_Frame) * frame_count);

    for (unsigned i = 0, n = frame_count; i < n; ++i) {
        frames[i].present_image_index = UINT32_MAX;
    }

    // Fences
    {
        VkFenceCreateInfo create_info = {
            .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
            .pNext = NULL,
            .flags = VK_FENCE_CREATE_SIGNALED_BIT,
        };
        
        for (unsigned i = 0, n = frame_count; i < n; ++i) {
            VkResult result = vkCreateFence(dev->device, &create_info, NULL, &frames[i].fence);
            if (result != VK_SUCCESS) {
                log_fatal("vkCreateFence failed: %d", (int)result);
                abort();
            }
        }
    }

    // Semaphores
    {
        VkSemaphoreCreateInfo create_info = {
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
            .pNext = NULL,
            .flags = 0,
        };

        for (unsigned i = 0, n = frame_count; i < n; ++i) {
            VkResult result;
            result = vkCreateSemaphore(dev->device, &create_info, NULL, &frames[i].acquire_next_image_semaphore);
            if (result != VK_SUCCESS) {
                log_fatal("vkCreateSemaphore failed: %d", (int)result);
                abort();
            }
            result = vkCreateSemaphore(dev->device, &create_info, NULL, &frames[i].execute_commands_semaphore);
            if (result != VK_SUCCESS) {
                log_fatal("vkCreateSemaphore failed: %d", (int)result);
                abort();
            }
        }
    }

    // Command Pools
    {
        VkCommandPoolCreateInfo create_info = {
            .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            .pNext = NULL,
            .flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
            .queueFamilyIndex = dev->main_queue_family,
        };
        for (unsigned i = 0, n = frame_count; i < n; ++i) {
            VkResult result = vkCreateCommandPool(dev->device, &create_info, NULL, &frames[i].main_command_pool);
            if (result != VK_SUCCESS) {
                log_fatal("vkCreateCommandPool failed: %d", (int)result);
                abort();
            }
        }
    }

    return frames;
}

static cgvk_Renderer* cgvk_new_renderer(const cgvk_Device* dev, const cgvk_Swapchain* swp)
{
    cgvk_Renderer* rnd = (cgvk_Renderer*)malloc(sizeof(cgvk_Renderer));
    memset(rnd, 0, sizeof(cgvk_Renderer));
    rnd->dev = dev;
    rnd->swp = swp;

    // TODO: customize
    rnd->frame_count = 3;

    rnd->render_pass = cgvk_new_render_pass(dev, swp);
    rnd->empty_pipeline_layout = cgvk_new_empty_pipeline_layout(dev);
    rnd->hello_triangle_pipeline = cgvk_new_hello_triangle_pipeline(dev, rnd->empty_pipeline_layout, rnd->render_pass);
    rnd->framebuffers = cgvk_new_framebuffers(dev, swp, rnd->render_pass);
    rnd->transient_vertex_buffers = cgvk_new_transient_allocator(dev, 1024 * 1024 * 2, 16, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
    rnd->transient_uniform_buffers = cgvk_new_transient_allocator(dev, 1024 * 4, 128, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
    rnd->frames = cgvk_new_frames(dev, rnd->frame_count);

    return rnd;
}

static void cgvk_free_renderer(cgvk_Renderer* rnd)
{
    VkDevice device = rnd->dev->device;

    vkDestroyPipeline(device, rnd->hello_triangle_pipeline, NULL);
    vkDestroyPipelineLayout(device, rnd->empty_pipeline_layout, NULL);

    for (unsigned i = 0, n = rnd->swp->image_count; i < n; ++i) {
        vkDestroyFramebuffer(device, rnd->framebuffers[i], NULL);
    }

    vkDestroyRenderPass(device, rnd->render_pass, NULL);

    for (unsigned i = 0, n = rnd->frame_count; i < n; ++i) {
        vkDestroySemaphore(device, rnd->frames[i].acquire_next_image_semaphore, NULL);
        vkDestroySemaphore(device, rnd->frames[i].execute_commands_semaphore, NULL);
        vkDestroyFence(device, rnd->frames[i].fence, NULL);
        vkDestroyCommandPool(device, rnd->frames[i].main_command_pool, NULL);
    }

    cgvk_free_transient_allocator(rnd->transient_vertex_buffers);
    cgvk_free_transient_allocator(rnd->transient_uniform_buffers);

    free(rnd->framebuffers);
    free(rnd->frames);
    free(rnd);
}

// ============================================================================

static inline uint32_t cgvk_get_current_frame_index(cgvk_Renderer* rnd)
{
    return (uint32_t) (rnd->current_frame % rnd->frame_count);
}

static void cgvk_advance_next_frame(cgvk_Renderer* rnd)
{
    assert(rnd->current_frame < UINT64_MAX);
    rnd->current_frame++;
}

static void cgvk_acquire_swapchain_image(cgvk_Renderer* rnd, uint32_t fidx)
{
    VkResult result = vkAcquireNextImageKHR(rnd->dev->device, rnd->swp->swapchain, UINT64_MAX, rnd->frames[fidx].acquire_next_image_semaphore, VK_NULL_HANDLE, &rnd->frames[fidx].present_image_index);
    if (result != VK_SUCCESS) {
        log_fatal("vkAcquireNextImageKHR failed: %d", (int)result);
        abort();
    }
}

static VkCommandBuffer cgvk_begin_main_command_buffer(cgvk_Renderer* rnd, uint32_t fidx)
{
    VkResult result;
    VkCommandBuffer cmdbuf;

    VkCommandBufferAllocateInfo alloc_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .pNext = NULL,
        .commandPool = rnd->frames[fidx].main_command_pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };

    result = vkAllocateCommandBuffers(rnd->dev->device, &alloc_info, &cmdbuf);
    if (result != VK_SUCCESS) {
        log_fatal("vkAllocateCommandBuffers failed: %d", (int)result);
        abort();
    }

    VkCommandBufferBeginInfo begin_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .pNext = NULL,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        .pInheritanceInfo = NULL,
    };

    result = vkBeginCommandBuffer(cmdbuf, &begin_info);
    if (result != VK_SUCCESS) {
        log_fatal("vkBeginCommandBuffer failed: %d", (int)result);
        abort();
    }

    return cmdbuf;
}

static void cgvk_end_main_command_buffer(cgvk_Renderer* rnd, uint32_t fidx, VkCommandBuffer cmdbuf)
{
    VkResult result;

    result = vkEndCommandBuffer(cmdbuf);
    if (result != VK_SUCCESS) {
        log_fatal("vkEndCommandBuffer failed: %d", (int)result);
        abort();
    }

    result = vkResetFences(rnd->dev->device, 1, &rnd->frames[fidx].fence);
    if (result != VK_SUCCESS) {
        log_fatal("vkResetFences failed: %d", (int)result);
        abort();
    }

    uint32_t wait_semaphore_count = 1;
    VkSemaphore wait_semaphores[1] = {
        rnd->frames[fidx].acquire_next_image_semaphore,
    };
    VkPipelineStageFlags wait_dststages[1] = {
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
    };

    VkSubmitInfo submit_info = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .pNext = NULL,
        .waitSemaphoreCount = wait_semaphore_count,
        .pWaitSemaphores = wait_semaphores,
        .pWaitDstStageMask = wait_dststages,
        .commandBufferCount = 1,
        .pCommandBuffers = &cmdbuf,
        .signalSemaphoreCount = 1,
        .pSignalSemaphores = &rnd->frames[fidx].execute_commands_semaphore,
    };

    result = vkQueueSubmit(rnd->dev->main_queue, 1, &submit_info, rnd->frames[fidx].fence);
    if (result != VK_SUCCESS) {
        log_fatal("vkQueueSubmit failed: %d", (int)result);
        abort();
    }
}

static void cgvk_begin_render_pass(cgvk_Renderer* rnd, uint32_t fidx, VkCommandBuffer cmdbuf)
{
    uint32_t clear_value_count = 1;
    VkClearValue clear_values[1] = {
        {
            .color = {
                .float32 = { 0.0f, 0.0f, 0.0f, 0.0f },
            },
        }
    };

    VkRenderPassBeginInfo main_rp_begin = {
        .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
        .pNext = NULL,
        .renderPass = rnd->render_pass,
        .framebuffer = rnd->framebuffers[rnd->frames[fidx].present_image_index],
        .renderArea = {
            .offset = { 0, 0 },
            .extent = { rnd->swp->width, rnd->swp->height },
        },
        .clearValueCount = clear_value_count,
        .pClearValues = clear_values,
    };

    vkCmdBeginRenderPass(cmdbuf, &main_rp_begin, VK_SUBPASS_CONTENTS_INLINE);
}

static void cgvk_end_render_pass(cgvk_Renderer* rnd, uint32_t fidx, VkCommandBuffer cmdbuf)
{
    (void)rnd;
    (void)fidx;

    vkCmdEndRenderPass(cmdbuf);
}

static void cgvk_set_viewport_and_scissor(cgvk_Renderer* rnd, uint32_t fidx, VkCommandBuffer cmdbuf)
{
    (void)fidx;

    VkViewport viewports[] = {
        {
            .x = 0.0f,
            .y = 0.0f,
            .width = (float)rnd->swp->width,
            .height = (float)rnd->swp->height,
            .minDepth = 0.0f,
            .maxDepth = 1.0f,
        },
    };
    VkRect2D scissors[] = {
        {
            .offset = { 0, 0 },
            .extent = { rnd->swp->width, rnd->swp->height },
        },
    };

    vkCmdSetViewport(cmdbuf, 0, 1, viewports);
    vkCmdSetScissor(cmdbuf, 0, 1, scissors);
}

static void cgvk_draw_frame(cgvk_Renderer* rnd, uint32_t fidx)
{
    VkCommandBuffer cmdbuf = cgvk_begin_main_command_buffer(rnd, fidx);

    cgvk_begin_render_pass(rnd, fidx, cmdbuf);

    vkCmdBindPipeline(cmdbuf, VK_PIPELINE_BIND_POINT_GRAPHICS, rnd->hello_triangle_pipeline);
    cgvk_set_viewport_and_scissor(rnd, fidx, cmdbuf);

    // upload vertex data
    {
        struct vertex {
            float x;
            float y;
            float z;
            uint32_t c;
        };

        cgvk_TransientBlock block;
        cgvk_allocate_transient_block(rnd->transient_vertex_buffers, sizeof(struct vertex) * 3, &block);

        // fill vertex data
        struct vertex* p = (struct vertex*)block.mapped;
        p[0].x = 0.5f;
        p[0].y = -0.5f;
        p[0].z = 0.0f;
        p[0].c = 0xFFFF0000;
        p[1].x = 0.5f;
        p[1].y = 0.5f;
        p[1].z = 0.0f;
        p[1].c = 0xFF00FF00;
        p[2].x = -0.5f;
        p[2].y = 0.5f;
        p[2].z = 0.0f;
        p[2].c = 0xFF0000FF;

        VkDeviceSize offset = 0;
        vkCmdBindVertexBuffers(cmdbuf, 0, 1, &block.buffer, &offset);
    }

    vkCmdDraw(cmdbuf, 3, 1, 0, 0);

    cgvk_end_render_pass(rnd, fidx, cmdbuf);

    cgvk_submit_transient_buffers(rnd->transient_vertex_buffers, fidx);
    cgvk_submit_transient_buffers(rnd->transient_uniform_buffers, fidx);
    cgvk_end_main_command_buffer(rnd, fidx, cmdbuf);
}

static void cgvk_present_frame(cgvk_Renderer* rnd, uint32_t fidx)
{
    uint32_t wait_semaphore_count = 1;
    VkSemaphore wait_semaphores[] = {
        rnd->frames[fidx].execute_commands_semaphore,
    };
    
    uint32_t image_index = rnd->frames[fidx].present_image_index;

    VkPresentInfoKHR present_info = {
        .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
        .pNext = NULL,
        .waitSemaphoreCount = wait_semaphore_count,
        .pWaitSemaphores = wait_semaphores,
        .swapchainCount = 1,
        .pSwapchains = &rnd->swp->swapchain,
        .pImageIndices = &image_index,
        .pResults = NULL,
    };

    VkResult result = vkQueuePresentKHR(rnd->dev->present_queue, &present_info);
    if (result != VK_SUCCESS) {
        log_fatal("vkQueuePresentKHR failed: %d", (int)result);
        abort();
    }
}

static void cgvk_wait_frame(cgvk_Renderer* rnd, uint32_t fidx)
{
    VkResult result = vkWaitForFences(rnd->dev->device, 1, &rnd->frames[fidx].fence, VK_TRUE, UINT64_MAX);
    if (result != VK_SUCCESS) {
        log_fatal("vkWaitForFences failed: %d", (int)result);
        abort();
    }
}

static void cgvk_reset_frame(cgvk_Renderer* rnd, uint32_t fidx)
{
    VkResult result;
    VkDevice device = rnd->dev->device;

    result = vkResetCommandPool(device, rnd->frames[fidx].main_command_pool, 0);
    if (result != VK_SUCCESS) {
        log_fatal("vkResetCommandPool failed: %d", (int)result);
        abort();
    }

    cgvk_reset_transient_buffers(rnd->transient_vertex_buffers, fidx);
    cgvk_reset_transient_buffers(rnd->transient_uniform_buffers, fidx);

    rnd->frames[fidx].present_image_index = UINT32_MAX;
}

static void cgvk_render_frame(cgvk_Renderer* rnd)
{
    uint32_t fidx = cgvk_get_current_frame_index(rnd);

    cgvk_acquire_swapchain_image(rnd, fidx);
    cgvk_draw_frame(rnd, fidx);
    cgvk_present_frame(rnd, fidx);

    cgvk_advance_next_frame(rnd);

    fidx = cgvk_get_current_frame_index(rnd);
    cgvk_wait_frame(rnd, fidx);
    cgvk_reset_frame(rnd, fidx);
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

    for (unsigned i = 0, n = present_mode_count; i < n; ++i) {
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
        for (unsigned i = 0, n = format_count; i < n; ++i) {
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

static cgvk_Swapchain* cgvk_new_swapchain(const cgvk_Device* dev)
{
    VkResult result;

    VkDevice device = dev->device;
    VkSurfaceKHR surface = cgvk_get_surface();

    // Query surface capabilites
    VkSurfaceCapabilitiesKHR caps;
    result = vkGetPhysicalDeviceSurfaceCapabilitiesKHR(dev->gpu, surface, &caps);
    if (result != VK_SUCCESS) {
        log_fatal("vkGetPhysicalDeviceSurfaceCapabilitiesKHR failed: %d", (int)result);
        abort();
    }

    // Select image count
    // TODO: customize image count
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
    result = vkCreateSwapchainKHR(device, &create_info, NULL, &swapchain);
    if (result != VK_SUCCESS) {
        log_fatal("vkCreateSwapchainKHR failed: %d", (int)result);
        abort();
    }

    // Get actual image count
    result = vkGetSwapchainImagesKHR(device, swapchain, &image_count, NULL);
    if (result != VK_SUCCESS) {
        log_fatal("vkGetSwapchainImagesKHR failed: %d", (int)result);
        abort();
    }

    assert(extent.width < UINT16_MAX);
    assert(extent.height < UINT16_MAX);

    size_t size = sizeof(cgvk_Swapchain) + sizeof(cgvk_SwapchainImage) * image_count;

    cgvk_Swapchain* swp = (cgvk_Swapchain*)malloc(size);
    memset(swp, 0, sizeof(cgvk_Swapchain));

    // Fill the fields
    swp->dev = dev;
    swp->swapchain = swapchain;
    swp->format = fmt;
    swp->image_count = (uint8_t)image_count;
    swp->width = (uint16_t)extent.width;
    swp->height = (uint16_t)extent.height;

    // Get images
    VkImage images[image_count];
    result = vkGetSwapchainImagesKHR(device, swapchain, &image_count, images);
    if (result != VK_SUCCESS) {
        log_fatal("vkGetSwapchainImagesKHR failed: %d", (int)result);
        abort();
    }
    for (unsigned i = 0, n = image_count; i < n; ++i) {
        swp->images[i].image = images[i];
    }

    // Create image views
    for (unsigned i = 0, n = image_count; i < n; ++i) {
        VkImageViewCreateInfo create_info = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .pNext = NULL,
            .flags = 0,
            .image = swp->images[i].image,
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
        result = vkCreateImageView(device, &create_info, NULL, &swp->images[i].image_view);
        if (result != VK_SUCCESS) {
            log_fatal("vkCreateImageView failed: %d", (int)result);
            abort();
        }
    }

    return swp;
}

static void cgvk_free_swapchain(cgvk_Swapchain* swp)
{
    VkDevice device = swp->dev->device;

    for (int i = 0, n = swp->image_count; i < n; ++i) {
        vkDestroyImageView(device, swp->images[i].image_view, NULL);
    }

    if (swp->swapchain) {
        vkDestroySwapchainKHR(device, swp->swapchain, NULL);
    }

    free(swp);
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
    uint32_t main_queue_family = UINT32_MAX;
    uint32_t present_queue_family = UINT32_MAX;

    uint32_t queue_family_count;
    vkGetPhysicalDeviceQueueFamilyProperties(gpu, &queue_family_count, NULL);
    
    VkQueueFamilyProperties properties[queue_family_count];
    vkGetPhysicalDeviceQueueFamilyProperties(gpu, &queue_family_count, properties);

    VkBool32 supported[queue_family_count];
    for (unsigned i = 0, n = queue_family_count; i < n; ++i) {
        VkResult result = vkGetPhysicalDeviceSurfaceSupportKHR(gpu, i, surface, &supported[i]);
        if (result != VK_SUCCESS) {
            log_fatal("vkGetPhysicalDeviceSurfaceSupportKHR failed: %d", (int)result);
            abort();
        }
    }
    for (unsigned i = 0, n = queue_family_count; i < n; ++i) {
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
        VkResult result = vkEnumerateDeviceExtensionProperties(gpu, NULL, &available_extension_count, NULL);
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
            for (unsigned i = 0, n = available_extension_count; i < n; ++i) {
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
    VkResult result = vkCreateDevice(gpu, &create_info, NULL, &device);
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
            for (unsigned i = 0, n = available_layer_count; i < n; ++i) {
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
                for (unsigned i = 0, n = available_extension_count; i < n; ++i) {
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
            for (unsigned i = 0, n = available_extension_count; i < n; ++i) {
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

static cgvk_Device dev_;
static cgvk_Swapchain* swp_;
static cgvk_Renderer* rnd_;

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

    //
    cgvk_init_device(&dev_);
    swp_ = cgvk_new_swapchain(&dev_);
    rnd_ = cgvk_new_renderer(&dev_, swp_);
}

CGVK_API void cgvk_quit()
{
    {
        VkResult result = vkDeviceWaitIdle(dev_.device);
        if (result != VK_SUCCESS) {
            log_fatal("vkDeviceWaitIdle failed: %d", (int)result);
            abort();
        }
    }

    //
    cgvk_free_renderer(rnd_);
    cgvk_free_swapchain(swp_);
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
    cgvk_render_frame(rnd_);
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
    assert(w > 0);
    assert(h > 0);
    extent->width = (uint32_t)w;
    extent->height = (uint32_t)h;
}
