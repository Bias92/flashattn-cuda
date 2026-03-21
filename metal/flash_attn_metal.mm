#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

// ============================================================
// FlashAttention Metal Host Code
//   Compiles .metal shader, manages buffers, dispatches compute.
//   Exposes C API for Python ctypes binding.
// ============================================================

struct FlashAttnParams {
    int N;
    int BH;
};

static id<MTLDevice>              g_device   = nil;
static id<MTLCommandQueue>        g_queue    = nil;
static id<MTLComputePipelineState> g_fwd_pso = nil;
static bool g_initialized = false;

// ============================================================
// Initialization: compile shader + create pipeline
// ============================================================
static bool ensure_initialized(void) {
    if (g_initialized) return true;

    g_device = MTLCreateSystemDefaultDevice();
    if (!g_device) {
        fprintf(stderr, "[Metal] No Metal device found\n");
        return false;
    }

    g_queue = [g_device newCommandQueue];
    if (!g_queue) {
        fprintf(stderr, "[Metal] Failed to create command queue\n");
        return false;
    }

    // Load .metal source from same directory as this .mm file
    // At runtime, look for flash_attn.metal next to the dylib
    NSString *path = nil;

    // Try 1: same directory as executable
    NSString *execDir = [[NSBundle mainBundle] bundlePath];
    path = [execDir stringByAppendingPathComponent:@"flash_attn.metal"];
    if (![[NSFileManager defaultManager] fileExistsAtPath:path]) {
        // Try 2: current working directory
        path = @"flash_attn.metal";
        if (![[NSFileManager defaultManager] fileExistsAtPath:path]) {
            // Try 3: metal/ subdirectory
            path = @"metal/flash_attn.metal";
            if (![[NSFileManager defaultManager] fileExistsAtPath:path]) {
                fprintf(stderr, "[Metal] Cannot find flash_attn.metal\n");
                return false;
            }
        }
    }

    NSError *error = nil;
    NSString *source = [NSString stringWithContentsOfFile:path
                                                encoding:NSUTF8StringEncoding
                                                   error:&error];
    if (!source) {
        fprintf(stderr, "[Metal] Failed to read shader: %s\n",
                [[error localizedDescription] UTF8String]);
        return false;
    }

    id<MTLLibrary> library = [g_device newLibraryWithSource:source
                                                    options:nil
                                                      error:&error];
    if (!library) {
        fprintf(stderr, "[Metal] Shader compile error: %s\n",
                [[error localizedDescription] UTF8String]);
        return false;
    }

    id<MTLFunction> fwd_fn = [library newFunctionWithName:@"flash_attn_fwd_kernel"];
    if (!fwd_fn) {
        fprintf(stderr, "[Metal] Function 'flash_attn_fwd_kernel' not found\n");
        return false;
    }

    g_fwd_pso = [g_device newComputePipelineStateWithFunction:fwd_fn error:&error];
    if (!g_fwd_pso) {
        fprintf(stderr, "[Metal] Pipeline error: %s\n",
                [[error localizedDescription] UTF8String]);
        return false;
    }

    printf("[Metal] Initialized: %s\n", [[g_device name] UTF8String]);
    printf("[Metal] Max threadgroup memory: %lu bytes\n",
           (unsigned long)[g_device maxThreadgroupMemoryLength]);

    g_initialized = true;
    return true;
}

// ============================================================
// C API for Python ctypes
// ============================================================
extern "C" {

// Initialize Metal (call once)
int metal_init(void) {
    return ensure_initialized() ? 0 : -1;
}

// Get device name
const char* metal_device_name(void) {
    if (!ensure_initialized()) return "unknown";
    static char name[256];
    strncpy(name, [[g_device name] UTF8String], sizeof(name) - 1);
    return name;
}

// Forward pass
// Q, K, V: [BH * N * D] float arrays (input)
// O:       [BH * N * D] float array (output)
// L:       [BH * N]     float array (output, logsumexp)
// Returns 0 on success, -1 on error
int metal_flash_attn_forward(
    const float* Q_host,
    const float* K_host,
    const float* V_host,
    float* O_host,
    float* L_host,
    int B, int H, int N, int D)
{
    if (!ensure_initialized()) return -1;
    if (D != 64) {
        fprintf(stderr, "[Metal] D must be 64, got %d\n", D);
        return -1;
    }

    int BH = B * H;
    size_t qkv_bytes = (size_t)BH * N * D * sizeof(float);
    size_t l_bytes   = (size_t)BH * N * sizeof(float);

    // Create Metal buffers (shared memory — CPU/GPU unified, no copy needed)
    id<MTLBuffer> buf_Q = [g_device newBufferWithBytes:Q_host
                                                length:qkv_bytes
                                               options:MTLResourceStorageModeShared];
    id<MTLBuffer> buf_K = [g_device newBufferWithBytes:K_host
                                                length:qkv_bytes
                                               options:MTLResourceStorageModeShared];
    id<MTLBuffer> buf_V = [g_device newBufferWithBytes:V_host
                                                length:qkv_bytes
                                               options:MTLResourceStorageModeShared];
    id<MTLBuffer> buf_O = [g_device newBufferWithLength:qkv_bytes
                                               options:MTLResourceStorageModeShared];
    id<MTLBuffer> buf_L = [g_device newBufferWithLength:l_bytes
                                               options:MTLResourceStorageModeShared];

    FlashAttnParams params;
    params.N  = N;
    params.BH = BH;
    id<MTLBuffer> buf_params = [g_device newBufferWithBytes:&params
                                                     length:sizeof(params)
                                                    options:MTLResourceStorageModeShared];

    if (!buf_Q || !buf_K || !buf_V || !buf_O || !buf_L || !buf_params) {
        fprintf(stderr, "[Metal] Buffer allocation failed\n");
        return -1;
    }

    // Dispatch
    int BR = 32;
    int num_q_blocks = (N + BR - 1) / BR;

    id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

    [enc setComputePipelineState:g_fwd_pso];
    [enc setBuffer:buf_Q      offset:0 atIndex:0];
    [enc setBuffer:buf_K      offset:0 atIndex:1];
    [enc setBuffer:buf_V      offset:0 atIndex:2];
    [enc setBuffer:buf_O      offset:0 atIndex:3];
    [enc setBuffer:buf_L      offset:0 atIndex:4];
    [enc setBuffer:buf_params offset:0 atIndex:5];

    MTLSize threadgroupSize = MTLSizeMake(BR, 1, 1);
    MTLSize gridSize        = MTLSizeMake(num_q_blocks, BH, 1);

    [enc dispatchThreadgroups:gridSize
        threadsPerThreadgroup:threadgroupSize];
    [enc endEncoding];

    [cmd commit];
    [cmd waitUntilCompleted];

    if ([cmd error]) {
        fprintf(stderr, "[Metal] Execution error: %s\n",
                [[[cmd error] localizedDescription] UTF8String]);
        return -1;
    }

    // Read back results (shared memory — just memcpy)
    memcpy(O_host, [buf_O contents], qkv_bytes);
    memcpy(L_host, [buf_L contents], l_bytes);

    return 0;
}

// Benchmark: run forward N_runs times, return average ms
double metal_flash_attn_forward_bench(
    const float* Q_host,
    const float* K_host,
    const float* V_host,
    float* O_host,
    float* L_host,
    int B, int H, int N, int D,
    int warmup, int repeats)
{
    if (!ensure_initialized()) return -1.0;
    if (D != 64) return -1.0;

    int BH = B * H;
    size_t qkv_bytes = (size_t)BH * N * D * sizeof(float);
    size_t l_bytes   = (size_t)BH * N * sizeof(float);
    int BR = 32;
    int num_q_blocks = (N + BR - 1) / BR;

    id<MTLBuffer> buf_Q = [g_device newBufferWithBytes:Q_host length:qkv_bytes
                                               options:MTLResourceStorageModeShared];
    id<MTLBuffer> buf_K = [g_device newBufferWithBytes:K_host length:qkv_bytes
                                               options:MTLResourceStorageModeShared];
    id<MTLBuffer> buf_V = [g_device newBufferWithBytes:V_host length:qkv_bytes
                                               options:MTLResourceStorageModeShared];
    id<MTLBuffer> buf_O = [g_device newBufferWithLength:qkv_bytes
                                               options:MTLResourceStorageModeShared];
    id<MTLBuffer> buf_L = [g_device newBufferWithLength:l_bytes
                                               options:MTLResourceStorageModeShared];

    FlashAttnParams params = { N, BH };
    id<MTLBuffer> buf_params = [g_device newBufferWithBytes:&params
                                                     length:sizeof(params)
                                                    options:MTLResourceStorageModeShared];

    MTLSize threadgroupSize = MTLSizeMake(BR, 1, 1);
    MTLSize gridSize        = MTLSizeMake(num_q_blocks, BH, 1);

    // Warmup
    for (int i = 0; i < warmup; i++) {
        id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:g_fwd_pso];
        [enc setBuffer:buf_Q offset:0 atIndex:0];
        [enc setBuffer:buf_K offset:0 atIndex:1];
        [enc setBuffer:buf_V offset:0 atIndex:2];
        [enc setBuffer:buf_O offset:0 atIndex:3];
        [enc setBuffer:buf_L offset:0 atIndex:4];
        [enc setBuffer:buf_params offset:0 atIndex:5];
        [enc dispatchThreadgroups:gridSize threadsPerThreadgroup:threadgroupSize];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }

    // Timed runs
    CFAbsoluteTime t0 = CFAbsoluteTimeGetCurrent();
    for (int i = 0; i < repeats; i++) {
        id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:g_fwd_pso];
        [enc setBuffer:buf_Q offset:0 atIndex:0];
        [enc setBuffer:buf_K offset:0 atIndex:1];
        [enc setBuffer:buf_V offset:0 atIndex:2];
        [enc setBuffer:buf_O offset:0 atIndex:3];
        [enc setBuffer:buf_L offset:0 atIndex:4];
        [enc setBuffer:buf_params offset:0 atIndex:5];
        [enc dispatchThreadgroups:gridSize threadsPerThreadgroup:threadgroupSize];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }
    CFAbsoluteTime t1 = CFAbsoluteTimeGetCurrent();

    double avg_ms = (t1 - t0) / repeats * 1000.0;

    // Copy final output
    memcpy(O_host, [buf_O contents], qkv_bytes);
    memcpy(L_host, [buf_L contents], l_bytes);

    return avg_ms;
}

}  // extern "C"
