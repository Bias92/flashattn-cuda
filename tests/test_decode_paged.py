"""Native page reads, ragged lengths and graph replay with changing metadata."""

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "integrations" / "vllm"))
sys.path.insert(0, str(ROOT / "bench"))
from scratch_vllm.loader import SOURCE, load_extension
from bench_attention_decode import exclusive_gpu_check


def make_case(d=64, page_size=16, lengths=(1, 17, 129), hq=32, hkv=4,
              hnd=False, padding=False, chunk=128, scale=None, amplitude=1):
    batch = len(lengths)
    max_len = max(max(lengths), 1)
    capacity = ((max_len + page_size - 1) // page_size + 1) * page_size
    columns = capacity // page_size
    pages = batch * columns + 3
    q = torch.randn(batch + 2, hq, d, device="cuda", dtype=torch.float16) * amplitude
    def cache():
        shape = (pages, hkv, page_size, d) if hnd else (pages, page_size, hkv, d)
        x = torch.randn(shape, device="cuda", dtype=torch.float16) * amplitude
        return x.transpose(1, 2) if hnd else x
    k, v = cache(), cache()
    # Poison unused entries: partial tiles must not dereference them.
    table = torch.full((batch, columns + 3), -987654, device="cuda", dtype=torch.int32)[:, :columns]
    permutation = torch.randperm(pages - 1, device="cuda") + 1
    for b, n in enumerate(lengths):
        needed = (n + page_size - 1) // page_size
        table[b, :needed] = permutation[b * columns:b * columns + needed].int()
    lens = torch.tensor(lengths, device="cuda", dtype=torch.int32)
    qlens = [1] * batch
    if padding:
        qlens[-1] = 0
    starts = torch.tensor([0] + list(torch.tensor(qlens).cumsum(0).tolist()),
                          device="cuda", dtype=torch.int32)
    output_storage = torch.full((batch + 2, hq, d * 2), -7, device="cuda", dtype=torch.float16)
    out = output_storage[..., :d]  # Strided output must not touch neighboring storage.
    splits = (capacity + chunk - 1) // chunk
    workspace = torch.empty((batch, hq, splits, d + 2), device="cuda", dtype=torch.float32)
    return dict(q=q, k=k, v=v, table=table, lengths=lens, starts=starts, out=out,
                storage=output_storage, workspace=workspace, capacity=capacity,
                scale=1 / math.sqrt(d) if scale is None else scale, chunk=chunk)


def invoke(ext, c):
    ext.out(c["q"], c["k"], c["v"], c["table"], c["lengths"], c["starts"],
            c["out"], c["workspace"], c["capacity"], c["scale"], c["chunk"])


def check(c, amplitude=1):
    d = c["q"].shape[-1]
    page_size = c["k"].shape[1]
    groups = c["q"].shape[1] // c["k"].shape[2]
    worst = 0.0
    lengths, starts = c["lengths"].tolist(), c["starts"].tolist()
    for b, n in enumerate(lengths):
        if starts[b + 1] == starts[b]:
            continue
        qi = starts[b]
        if n == 0:
            ref = torch.zeros_like(c["q"][qi], dtype=torch.float32)
        else:
            pages = c["table"][b, :(n + page_size - 1) // page_size].long()
            k = c["k"][pages].reshape(-1, c["k"].shape[2], d)[:n].transpose(0, 1)
            v = c["v"][pages].reshape(-1, c["v"].shape[2], d)[:n].transpose(0, 1)
            k, v = k.repeat_interleave(groups, 0).float(), v.repeat_interleave(groups, 0).float()
            scores = torch.einsum("hd,hnd->hn", c["q"][qi].float(), k) * c["scale"]
            ref = torch.einsum("hn,hnd->hd", scores.softmax(-1), v)
        actual = c["out"][qi].float()
        worst = max(worst, (actual - ref).abs().max().item())
        torch.testing.assert_close(actual, ref, atol=8e-4 * amplitude, rtol=1e-3)
    assert (c["out"][starts[-1]:] == -7).all()
    assert (c["storage"][..., d:] == -7).all()
    return worst


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    exclusive_gpu_check()
    torch.manual_seed(943)
    torch.backends.cuda.matmul.allow_tf32 = False
    ext = load_extension()
    cases = []
    for d in (64, 128):
        for page in (16, 32, 64, 128):
            for hnd in (False, True):
                for hq, hkv in ((32, 4), (17, 1), (4, 4)):
                    c = make_case(d=d, page_size=page, hnd=hnd, hq=hq, hkv=hkv,
                                  lengths=(0, 1, page - 1, page, page + 1, 511), padding=True)
                    invoke(ext, c)
                    diff = check(c)
                    cases.append(dict(d=d, page_size=page, hnd=hnd, hq=hq, hkv=hkv, max_diff=diff))
    for scale in (0, -0.2, 1):
        c = make_case(lengths=(33, 255, 4095), scale=scale)
        invoke(ext, c)
        cases.append(dict(scale=scale, max_diff=check(c)))
    for chunk in (64, 256, 512):
        c = make_case(d=128, lengths=(127, 513, 4097), chunk=chunk, amplitude=8)
        invoke(ext, c)
        cases.append(dict(chunk=chunk, max_diff=check(c, amplitude=8)))

    c = make_case(lengths=(129, 65, 0), padding=True)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        invoke(ext, c)
    torch.cuda.current_stream().wait_stream(stream)
    check(c)
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        invoke(ext, c)
    # Populate previously unused pages and activate the padded request without recapture.
    c["table"].copy_(torch.arange(c["table"].numel(), device="cuda", dtype=torch.int32)
                     .reshape_as(c["table"]))
    c["starts"].copy_(torch.tensor([0, 1, 2, 3], device="cuda", dtype=torch.int32))
    for lens in ([17, 130, 33], [0, 1, 144], [128, 64, 1]):
        c["lengths"].copy_(torch.tensor(lens, device="cuda", dtype=torch.int32))
        graph.replay()
        cases.append(dict(graph_lengths=lens, max_diff=check(c)))

    rejected = 0
    for name, value in (("q", c["q"].float()), ("lengths", c["lengths"].long()),
                        ("workspace", c["workspace"].flatten()[:1]), ("chunk", 17),
                        ("scale", float("nan")), ("capacity", 1 << 29)):
        invalid = dict(c)
        invalid[name] = value
        try:
            invoke(ext, invalid)
        except RuntimeError:
            rejected += 1
        else:
            raise AssertionError(f"Invalid {name} accepted")
    torch.cuda.synchronize()
    result = dict(cases=cases, passed=len(cases), invalid_rejected=rejected,
                  stream_check=True, source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                  so=ext.__file__, torch=torch.__version__, cuda=torch.version.cuda,
                  gpu=torch.cuda.get_device_name())
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "cases"}, indent=2))


if __name__ == "__main__":
    main()
