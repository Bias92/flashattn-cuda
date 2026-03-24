#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

int main() {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    printf("Device: %s\n\n", [[dev name] UTF8String]);
    
    NSArray<id<MTLCounterSet>> *counterSets = [dev counterSets];
    if (!counterSets || counterSets.count == 0) {
        printf("No counter sets available.\n");
        return 0;
    }
    
    for (id<MTLCounterSet> cs in counterSets) {
        printf("=== Counter Set: %s ===\n", [[cs name] UTF8String]);
        for (id<MTLCounter> c in [cs counters]) {
            printf("  %s\n", [[c name] UTF8String]);
        }
        printf("\n");
    }
    return 0;
}
