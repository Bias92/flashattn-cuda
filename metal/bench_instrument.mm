#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

struct FlashAttnParams { int N; int BH; };

int main() {
    int B=1, H=8, N=1024, D=64;
    int BH = B*H;
    size_t qkv_n = BH*N*D;
    size_t l_n = BH*N;

    // Random data
    float* Q = (float*)malloc(qkv_n*sizeof(float));
    float* K = (float*)malloc(qkv_n*sizeof(float));
    float* V = (float*)malloc(qkv_n*sizeof(float));
    for (size_t i=0; i<qkv_n; i++) {
        Q[i] = (float)drand48()*2-1;
        K[i] = (float)drand48()*2-1;
        V[i] = (float)drand48()*2-1;
    }

    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> queue = [dev newCommandQueue];

    NSError* err = nil;
    NSString* src = [NSString stringWithContentsOfFile:@"flash_attn.metal"
                              encoding:NSUTF8StringEncoding error:&err];
    id<MTLLibrary> lib = [dev newLibraryWithSource:src options:nil error:&err];
    id<MTLFunction> fn = [lib newFunctionWithName:@"flash_attn_fwd_kernel"];
    id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:fn error:&err];

    id<MTLBuffer> bQ = [dev newBufferWithBytes:Q length:qkv_n*4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bK = [dev newBufferWithBytes:K length:qkv_n*4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bV = [dev newBufferWithBytes:V length:qkv_n*4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bO = [dev newBufferWithLength:qkv_n*4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bL = [dev newBufferWithLength:l_n*4 options:MTLResourceStorageModeShared];
    FlashAttnParams p = {N, BH};
    id<MTLBuffer> bP = [dev newBufferWithBytes:&p length:sizeof(p) options:MTLResourceStorageModeShared];

    int BR=32, nq=(N+BR-1)/BR;
    MTLSize tg = MTLSizeMake(BR,1,1);
    MTLSize grid = MTLSizeMake(nq,BH,1);

    // Warmup
    for (int i=0; i<5; i++) {
        id<MTLCommandBuffer> cmd = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:bQ offset:0 atIndex:0];
        [enc setBuffer:bK offset:0 atIndex:1];
        [enc setBuffer:bV offset:0 atIndex:2];
        [enc setBuffer:bO offset:0 atIndex:3];
        [enc setBuffer:bL offset:0 atIndex:4];
        [enc setBuffer:bP offset:0 atIndex:5];
        [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }

    printf("Running 20 iterations for Instruments capture...\n");
    for (int i=0; i<20; i++) {
        id<MTLCommandBuffer> cmd = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:bQ offset:0 atIndex:0];
        [enc setBuffer:bK offset:0 atIndex:1];
        [enc setBuffer:bV offset:0 atIndex:2];
        [enc setBuffer:bO offset:0 atIndex:3];
        [enc setBuffer:bL offset:0 atIndex:4];
        [enc setBuffer:bP offset:0 atIndex:5];
        [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        double gpu_ms = ([cmd GPUEndTime]-[cmd GPUStartTime])*1000.0;
        printf("  iter %2d: %.2f ms\n", i, gpu_ms);
    }
    printf("Done.\n");

    free(Q); free(K); free(V);
    return 0;
}
