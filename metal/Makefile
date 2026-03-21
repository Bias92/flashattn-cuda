# Makefile for FlashAttention Metal
# Build: make
# Clean: make clean

CXX = clang++
CXXFLAGS = -O2 -std=c++17 -fobjc-arc
FRAMEWORKS = -framework Metal -framework Foundation
TARGET = libflash_attn_metal.dylib

.PHONY: all clean

all: $(TARGET)

$(TARGET): flash_attn_metal.mm
	$(CXX) $(CXXFLAGS) -shared -o $@ $< $(FRAMEWORKS)

clean:
	rm -f $(TARGET)
