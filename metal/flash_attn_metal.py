"""
FlashAttention Metal — Python wrapper (ctypes)

Usage:
    from flash_attn_metal import MetalFlashAttention

    fa = MetalFlashAttention()
    O, L = fa.forward(Q, K, V)         # numpy arrays [B,H,N,D]
    ms = fa.bench_forward(Q, K, V)     # returns avg ms
"""

import ctypes
import os
import numpy as np
from pathlib import Path

_LIB = None

def _load_lib():
    global _LIB
    if _LIB is not None:
        return _LIB

    lib_dir = Path(__file__).parent
    lib_path = lib_dir / "libflash_attn_metal.dylib"

    if not lib_path.exists():
        raise RuntimeError(
            f"Metal library not found at {lib_path}.\n"
            f"Build first: cd metal && make"
        )

    _LIB = ctypes.cdll.LoadLibrary(str(lib_path))

    # metal_init() -> int
    _LIB.metal_init.restype = ctypes.c_int
    _LIB.metal_init.argtypes = []

    # metal_device_name() -> const char*
    _LIB.metal_device_name.restype = ctypes.c_char_p
    _LIB.metal_device_name.argtypes = []

    # metal_flash_attn_forward(Q, K, V, O, L, B, H, N, D) -> int
    fp = ctypes.POINTER(ctypes.c_float)
    _LIB.metal_flash_attn_forward.restype = ctypes.c_int
    _LIB.metal_flash_attn_forward.argtypes = [
        fp, fp, fp, fp, fp,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int
    ]

    # metal_flash_attn_forward_bench(..., warmup, repeats) -> double
    _LIB.metal_flash_attn_forward_bench.restype = ctypes.c_double
    _LIB.metal_flash_attn_forward_bench.argtypes = [
        fp, fp, fp, fp, fp,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int
    ]

    ret = _LIB.metal_init()
    if ret != 0:
        raise RuntimeError("Metal initialization failed")

    return _LIB


def _ptr(arr):
    """Get ctypes float pointer from numpy array."""
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


class MetalFlashAttention:
    def __init__(self):
        self.lib = _load_lib()
        self.device_name = self.lib.metal_device_name().decode()
        print(f"[MetalFlashAttention] Device: {self.device_name}")

    def forward(self, Q, K, V):
        """
        Forward pass.
        Args:
            Q, K, V: numpy float32 arrays of shape [B, H, N, D]
        Returns:
            O: [B, H, N, D] float32
            L: [B, H, N]    float32 (logsumexp)
        """
        assert Q.dtype == np.float32 and K.dtype == np.float32 and V.dtype == np.float32
        assert Q.ndim == 4
        B, H, N, D = Q.shape
        assert D == 64, f"D must be 64, got {D}"

        # Ensure contiguous
        Q = np.ascontiguousarray(Q)
        K = np.ascontiguousarray(K)
        V = np.ascontiguousarray(V)

        O = np.zeros_like(Q)
        L = np.zeros((B, H, N), dtype=np.float32)

        ret = self.lib.metal_flash_attn_forward(
            _ptr(Q), _ptr(K), _ptr(V), _ptr(O), _ptr(L),
            B, H, N, D
        )
        if ret != 0:
            raise RuntimeError("Metal forward kernel failed")

        return O, L

    def bench_forward(self, Q, K, V, warmup=10, repeats=100):
        """
        Benchmark forward pass.
        Returns average time in ms.
        """
        assert Q.dtype == np.float32
        B, H, N, D = Q.shape

        Q = np.ascontiguousarray(Q)
        K = np.ascontiguousarray(K)
        V = np.ascontiguousarray(V)

        O = np.zeros_like(Q)
        L = np.zeros((B, H, N), dtype=np.float32)

        ms = self.lib.metal_flash_attn_forward_bench(
            _ptr(Q), _ptr(K), _ptr(V), _ptr(O), _ptr(L),
            B, H, N, D, warmup, repeats
        )
        if ms < 0:
            raise RuntimeError("Metal benchmark failed")

        return ms
