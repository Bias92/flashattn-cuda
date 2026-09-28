# Build, API And Tests

[Project overview](../README.md)

These instructions describe the measured RTX 4060 Ti environment. The build
specifies `sm_89`; there is no check for a particular GPU model name. Operation
and performance on other GPUs have not been verified.

## Build And Use

Tested environment: WSL2 Ubuntu 24.04, Python 3.12, CUDA toolkit 12.8 and
PyTorch 2.10.0+cu128. The build targets `sm_89`.

```sh
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export MAX_JOBS=1
python3 -m pip install -e . --break-system-packages
```

```python
import torch
import attention_forward_cuda as attn

q, k, v = [torch.randn(1, 8, 1024, 64, device="cuda", dtype=torch.float16)
           for _ in range(3)]

o = attn.forward_only(q, k, v, causal=True)
o, lse = attn.forward(q, k, v, causal=True)
```

`forward_only` returns FP16 output shaped like Q. `forward` also returns FP32
row log-sum-exp with shape `[B, H_q, N_q]`. Both accept optional `softmax_scale`
(default `1/sqrt(D)`) and `out` arguments.

| Input property | Support |
|---|---|
| Tensor shapes | Q: `[B, H_q, N_q, D]`; K/V: `[B, H_kv, N_kv, D]` |
| Heads | MHA and GQA; `H_q` must be a multiple of `H_kv` |
| Mask | Dense or causal; causal aligns to the bottom right and requires `N_kv >= N_q` |
| Layout | Strided tensors; incompatible row alignment or last-dimension strides trigger a copy |
| Output buffer | FP16, Q-shaped, aligned, non-overlapping `out` |
| Head dimension | D64 is the performance target; D128 remains for compatibility |

The wrapper converts non-FP16 inputs to FP16. There is no backward, dropout,
arbitrary attention mask, paged KV cache or packed variable-length batch API.

## Tests And Benchmarks

Run from the repository root after setting the build environment above:

```sh
python3 tests/test_attention_forward.py
python3 tests/test_attention_rectangular.py
```

These cover FP32-reference comparisons, causal/dense masks, GQA, boundary
lengths, tensor strides, output buffers and unequal query/KV lengths.

For the D64 backend comparison, use an idle GPU and a new output path:

```sh
TORCH_CUDA_ARCH_LIST=8.9 python3 bench/report_attention_backends.py \
    --run 1 --output /tmp/d64-run1.json
```

Use run IDs 1, 2 and 3 in separate processes, with separate output files.
The runner records source/build hashes, actual backend dispatch, paired samples
and GPU telemetry. Its background-GPU check uses Windows PowerShell through WSL2.
