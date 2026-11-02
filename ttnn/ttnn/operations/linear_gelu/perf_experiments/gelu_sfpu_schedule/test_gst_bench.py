# SPDX-FileCopyrightText: 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Isolated packer-thread GELU bench (gelu_sfpu_schedule). Single core, fp32 DEST (HiFi2, approx off).

precision: reps=1, prints max abs err / bf16 ULP / mean signed error vs torch gelu(approximate="tanh") in fp64.
perf: reps=GST_REPS on GST_TILES tiles, bf16 out; read DEVICE KERNEL DURATION from the profiler CSV
(parse_perf.py) in the printed order.
"""
import os
from pathlib import Path

import pytest
import ttnn

HERE = Path(__file__).parent
TILE = 32
CB_IN, CB_OUT = 0, 16
CHUNK = 4

VARIANTS = {
    "base": 0,
    "base_nr": 1,
    "magic4": 2,
    "magic3": 3,
    "magic4x2": 4,
    "hwexp": 5,
    "magic3x2": 6,
    "magic4x2_nr1": 7,
    "magic3x2_nr1": 8,
    "diag_noclamp": 91,
    "diag_empty": 90,
    "magic4x2w": 10,
    "magic3x2w": 11,
}
SEL = [v for v in os.environ.get("GST_VARIANTS", ",".join(["none"] + list(VARIANTS))).split(",") if v]


def _core():
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])


def _mem(num_tiles):
    return ttnn.create_sharded_memory_config(
        shape=(num_tiles * TILE, TILE),
        core_grid=_core(),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def run(device, x_torch, variant, reps, out_dtype):
    num_tiles = x_torch.shape[0] // TILE
    mem = _mem(num_tiles)
    xin = ttnn.from_torch(x_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem)
    out = ttnn.allocate_tensor_on_device(ttnn.Shape([num_tiles * TILE, TILE]), out_dtype, ttnn.TILE_LAYOUT, device, mem)
    cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True, math_approx_mode=False
    )
    modes = [ttnn.UnpackToDestMode.Default] * 64
    modes[CB_IN] = ttnn.UnpackToDestMode.UnpackToDestFp32
    cfg.unpack_to_dest_mode = modes
    do_gelu = 0 if variant == "none" else 1
    compute = ttnn.KernelDescriptor(
        kernel_source=str(HERE / "bench_compute.cpp"),
        core_ranges=_core(),
        compile_time_args=[num_tiles, CHUNK, do_gelu],
        defines=[("GST_VARIANT", str(VARIANTS.get(variant, 0)))],
        runtime_args=[(ttnn.CoreCoord(0, 0), [reps])],
        config=cfg,
    )
    cbs = [ttnn.cb_descriptor_from_sharded_tensor(CB_IN, xin), ttnn.cb_descriptor_from_sharded_tensor(CB_OUT, out)]
    desc = ttnn.ProgramDescriptor(kernels=[compute], semaphores=[], cbs=cbs)
    res = ttnn.generic_op([xin, out], desc)
    return ttnn.to_torch(res)


def sweep_input(num_tiles=64, seed=0):
    import torch

    n = num_tiles * TILE * TILE
    g = torch.Generator().manual_seed(seed)
    dense = torch.linspace(-12.0, 12.0, n // 2, dtype=torch.float32)
    rnd = torch.randn(n - n // 2, generator=g, dtype=torch.float32) * 3.0
    return torch.cat([dense, rnd]).reshape(num_tiles * TILE, TILE)


def _ordered(bf16):
    import torch

    i = bf16.view(torch.int16).to(torch.int32)
    return torch.where(i < 0, -(i & 0x7FFF), i)


def metrics(x, y):
    import torch

    ref = torch.nn.functional.gelu(x.double(), approximate="tanh")
    err = y.double() - ref
    mask = ref.abs() >= 1e-3
    yb = y.to(torch.bfloat16)
    rb = ref.float().to(torch.bfloat16)
    ulp = (_ordered(yb) - _ordered(rb)).abs()
    rel = err[mask] / ref[mask]
    tail = ~mask
    return dict(
        max_abs=err.abs().max().item(),
        max_ulp_gated=ulp[mask].max().item(),
        frac_ulp1=(ulp[mask] >= 1).double().mean().item(),
        max_ulp_all=ulp.max().item(),
        mean_err=err.mean().item(),
        mean_rel=rel.mean().item(),
        max_rel=rel.abs().max().item(),
        tail_max_abs=err[tail].abs().max().item() if tail.any() else 0.0,
    )


@pytest.mark.parametrize("variant", [v for v in SEL if v != "none"])
def test_precision(device, variant):
    import torch

    x = sweep_input()
    for dt, name in ((ttnn.float32, "fp32out"), (ttnn.bfloat16, "bf16out")):
        y = run(device, x, variant, 1, dt).float()
        m = metrics(x, y)
        print(f"GST_PREC {variant} {name} " + " ".join(f"{k}={v:.4g}" for k, v in m.items()))
        assert torch.isfinite(y).all() or variant == "hwexp"
        if variant != "hwexp":
            assert m["max_ulp_gated"] <= 2


@pytest.mark.parametrize("variant", SEL)
def test_perf(device, variant):
    tiles = int(os.environ.get("GST_TILES", "64"))
    reps = int(os.environ.get("GST_REPS", "16"))
    x = sweep_input(tiles, seed=1)
    run(device, x, variant, reps, ttnn.bfloat16)
    ttnn.synchronize_device(device)
    print(f"GST_PERF_ORDER {variant} tiles={tiles} reps={reps}")


EXTREMES = [
    0.0,
    -0.0,
    1e-30,
    -1e-30,
    8.0,
    -8.0,
    9.5,
    -9.5,
    11.0,
    -11.0,
    20.0,
    -20.0,
    100.0,
    -100.0,
    1e3,
    -1e3,
    4e3,
    -4e3,
    1e5,
    -1e5,
    1e10,
    -1e10,
    1e17,
    -1e17,
    1e20,
    -1e20,
    1e30,
    -1e30,
    3e38,
    -3e38,
]


@pytest.mark.parametrize("variant", [v for v in SEL if v not in ("none",) and not v.startswith("diag")])
def test_extremes(device, variant):
    import torch

    vals = torch.tensor(EXTREMES, dtype=torch.float32)
    x = vals.repeat((TILE * TILE * CHUNK + len(vals) - 1) // len(vals))[: TILE * TILE * CHUNK].reshape(
        CHUNK * TILE, TILE
    )
    y = run(device, x, variant, 1, ttnn.float32).float()
    ref = torch.nn.functional.gelu(x.double(), approximate="tanh")
    bad = []
    for v in EXTREMES:
        m = x == v
        got, want = y[m][0].item(), ref[m][0].item()
        tol = max(1e-3, abs(want) * 2**-8)
        if not (abs(got - want) <= tol):
            bad.append(f"{v:g}->{got:.4g}(want {want:.4g})")
    print(f"GST_EXTREME {variant} bad={len(bad)} " + " ".join(bad))
