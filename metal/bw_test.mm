#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <cstdio>

// Simple copy kernel embedded as string
static const char* copyShader = R"(
#include <metal_stdlib>
using namespace metal;
kernel void copy_kernel(
    device const float* src [[buffer(0)]],
    device float* dst       [[buffer(1)]],
    uint tid [[thread_position_in_grid]])
{
    dst[tid] = src[tid];
}
)";

int main() {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> queue = [dev newCommandQueue];
    printf("Device: %s\n\n", [[dev name] UTF8String]);

    NSError* err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:
        [NSString stringWithUTF8String:copyShader] options:nil error:&err];
    id<MTLFunction> fn = [lib newFunctionWithName:@"copy_kernel"];
    id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:fn error:&err];

    // Test various sizes
    size_t sizes[] = {
        1*1024*1024,    // 1M floats = 4MB
        4*1024*1024,    // 4M = 16MB
        16*1024*1024,   // 16M = 64MB
        64*1024*1024,   // 64M = 256MB
    };

    int warmup = 10, repeats = 50;

    printf("%12s | %10s | %12s | %10s\n", "Size", "GPU Time", "Eff BW", "vs Peak");
    printf("------------------------------------------------------------\n");

    for (int s = 0; s < 4; s++) {
        size_t n = sizes[s];
        size_t bytes = n * sizeof(float);

        id<MTLBuffer> src = [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> dst = [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];

        // Fill src
        float* p = (float*)[src contents];
        for (size_t i = 0; i < n; i++) p[i] = 1.0f;

        NSUInteger maxTpg = [pso maxTotalThreadsPerThreadgroup];
        MTLSize tg = MTLSizeMake(maxTpg, 1, 1);
        MTLSize grid = MTLSizeMake(n, 1, 1);

        // Warmup
        for (int i = 0; i < warmup; i++) {
            id<MTLCommandBuffer> cmd = [queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:pso];
            [enc setBuffer:src offset:0 atIndex:0];
            [enc setBuffer:dst offset:0 atIndex:1];
            [enc dispatchThreads:grid threadsPerThreadgroup:tg];
            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
        }

        // Timed
        double gpu_total = 0;
        for (int i = 0; i < repeats; i++) {
            id<MTLCommandBuffer> cmd = [queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:pso];
            [enc setBuffer:src offset:0 atIndex:0];
            [enc setBuffer:dst offset:0 atIndex:1];
            [enc dispatchThreads:grid threadsPerThreadgroup:tg];
            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
            gpu_total += ([cmd GPUEndTime] - [cmd GPUStartTime]);
        }

        double avg_sec = gpu_total / repeats;
        double total_bytes = 2.0 * bytes; // read + write
        double bw_gbps = (total_bytes / avg_sec) / 1e9;
        double vs_peak = bw_gbps / 273.0 * 100;

        printf("%8.0fMB | %8.3fms | %9.2fGB/s | %7.1f%%\n",
               bytes/1e6, avg_sec*1000, bw_gbps, vs_peak);
    }

    return 0;
}
