# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Isolated Sinkhorn bake-off for mhc_pre (perf tournament round 2, idea E2).

One core, one fp32 coefficient-major tile, REPS x {copy_tile, sinkhorn variant} in one DEST window.
  test_correct: every variant (SKB_VARIANTS) vs v0 (the op's current code; bitwise?) and vs an fp64 torch
                Sinkhorn on the same fp32 logits (max abs err, max rel err, row / col sum deviation).
  test_perf:    DEVICE KERNEL DURATION at REPS = 1 and REPS = SKB_REPS; per-Sinkhorn us = slope.
Run via bench/run.sh (TT_METAL_DEVICE_PROFILER=1 for perf).
"""

import os
import struct
import importlib
from pathlib import Path

import pytest

torch = importlib.import_module("torch")
import ttnn

HERE = Path(__file__).parent
KERNEL = str(HERE / "sinkhorn_bench_compute.cpp")
N = 4
LOGIT0 = 2 * N
EPS = 1e-6
_KEY = "DEVICE KERNEL DURATION [ns]"
VARIANTS = [int(v) for v in os.environ.get("SKB_VARIANTS", "0").split(",")]
ITERS = [int(v) for v in os.environ.get("SKB_ITERS", "20").split(",")]


def _core():
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])


def _memcfg():
    return ttnn.create_sharded_memory_config(
        shape=(32, 32),
        core_grid=_core(),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def slot_pos(k, lane):
    """(row, col) of slot k (= dst_reg[k] of DEST tile 0), SFPU lane `lane`, in the logical 32x32 tile."""
    f, rb, o = k // 8, (k % 8) // 2, k % 2
    rr, cc = lane // 8, lane % 8
    return (f // 2) * 16 + 4 * rb + rr, (f % 2) * 16 + 2 * cc + o


_POS = [[slot_pos(k, l) for l in range(32)] for k in range(32)]


def pack_logits(L, filler):
    """L: [32 lanes, N, N] fp32 -> [32, 32] tile (other slots = filler)."""
    t = filler.clone()
    for i in range(N):
        for j in range(N):
            k = LOGIT0 + i * N + j
            for l in range(32):
                r, c = _POS[k][l]
                t[r, c] = L[l, i, j]
    return t


def unpack_comb(t):
    out = torch.empty((32, N, N), dtype=t.dtype)
    for i in range(N):
        for j in range(N):
            k = LOGIT0 + i * N + j
            for l in range(32):
                r, c = _POS[k][l]
                out[l, i, j] = t[r, c]
    return out


def sinkhorn_ref(L, iters, eps):
    m = torch.softmax(L, dim=-1) + eps
    m = m / (m.sum(dim=-2, keepdim=True) + eps)
    for _ in range(iters - 1):
        m = m / (m.sum(dim=-1, keepdim=True) + eps)
        m = m / (m.sum(dim=-2, keepdim=True) + eps)
    return m


def run_variant(device, tile, variant, reps, iters, defines=(), half=int(os.environ.get("SKB_HALF", 0))):
    tin = ttnn.from_torch(tile, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=_memcfg())
    tout = ttnn.allocate_tensor_on_device(ttnn.Shape([32, 32]), ttnn.float32, ttnn.TILE_LAYOUT, device, _memcfg())
    cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False
    )
    modes = [ttnn.UnpackToDestMode.Default] * 64
    modes[0] = ttnn.UnpackToDestMode.UnpackToDestFp32
    cfg.unpack_to_dest_mode = modes
    eps_bits = struct.unpack("<I", struct.pack("<f", EPS))[0]
    rt = ttnn.RuntimeArgs()
    rt[0][0] = [eps_bits, iters]
    k = ttnn.KernelDescriptor(
        kernel_source=KERNEL,
        core_ranges=_core(),
        compile_time_args=[variant, reps, half],
        runtime_args=rt,
        defines=list(defines),
        config=cfg,
    )
    cbs = [ttnn.cb_descriptor_from_sharded_tensor(0, tin), ttnn.cb_descriptor_from_sharded_tensor(16, tout)]
    desc = ttnn.ProgramDescriptor(kernels=[k], semaphores=[], cbs=cbs)
    out = ttnn.generic_op([tin, tout], desc)
    return ttnn.to_torch(out).float()


def _read_ns(device):
    ttnn.ReadDeviceProfiler(device)
    out = []
    for programs in (ttnn.get_latest_programs_perf_data() or {}).values():
        for p in programs:
            e = (getattr(p, "program_analyses_results", None) or {}).get(_KEY)
            if e is not None:
                out.append(float(e.duration))
    return out


def _cases():
    g = torch.Generator().manual_seed(1234)
    cases = []
    for name, scale in (("s1", 1.0), ("s3", 3.0), ("s8", 8.0), ("s40", 40.0)):
        for seed in range(2):
            L = torch.randn((32, N, N), generator=g) * scale
            if name == "s40" and seed == 1:
                L[:4] = 0.0  # uniform rows
                L[4:8] += 60.0  # large offsets (max subtraction)
            cases.append((f"{name}_{seed}", L.float()))
    return cases


def test_slot_map(device):
    """Variant 98 writes dst_reg[k] = k: checks the (slot, lane) -> tile element map used by the bench."""
    out = run_variant(device, torch.zeros(32, 32), 98, 1, 1)
    for k in range(32):
        for l in range(32):
            r, c = _POS[k][l]
            assert out[r, c].item() == float(k), (k, l, r, c, out[r, c].item())


def test_correct(device):
    filler = torch.full((32, 32), 0.37)
    for iters in ITERS:
        for name, L in _cases():
            tile = pack_logits(L, filler)
            ref = sinkhorn_ref(L.double(), iters, EPS)
            ref32 = sinkhorn_ref(L, iters, EPS)
            base = None
            for v in [0] + [v for v in VARIANTS if v != 0]:
                out = run_variant(device, tile, v, 1, iters)
                comb = unpack_comb(out)
                if v == 0:
                    base = comb
                assert torch.isfinite(comb).all(), (v, name)
                d = (comb.double() - ref).abs()
                rel = (d / ref.abs().clamp_min(1e-30)).max().item()
                bitwise = torch.equal(comb.view(torch.int32), base.view(torch.int32))
                dbase = (comb.double() - base.double()).abs().max().item()
                colerr = (comb.double().sum(-2) - 1).abs().max().item()
                rowerr = (comb.double().sum(-1) - 1).abs().max().item()
                t32 = (ref32.double() - ref).abs().max().item()
                # the rest of the tile (except the scratch slots 24..31) must be untouched
                keep = torch.ones(32, 32, dtype=torch.bool)
                for k in range(LOGIT0, 32):
                    for l in range(32):
                        keep[_POS[k][l]] = False
                untouched = torch.equal(out[keep], tile[keep])
                print(
                    f"CORRECT v{v} it{iters} {name}: bitwise_vs_v0={bitwise} maxabs_vs_v0={dbase:.3e} "
                    f"maxabs_vs_fp64={d.max().item():.3e} maxrel_vs_fp64={rel:.3e} (torch_fp32 {t32:.3e}) "
                    f"col|sum-1|={colerr:.2e} row|sum-1|={rowerr:.2e} untouched={untouched}"
                )
                assert untouched, (v, name)
                assert d.max().item() < 1e-4, (v, name, d.max().item())


@pytest.mark.skipif(os.environ.get("TT_METAL_DEVICE_PROFILER") != "1", reason="needs TT_METAL_DEVICE_PROFILER=1")
def test_perf(device):
    reps_hi = int(os.environ.get("SKB_REPS", 11))
    g = torch.Generator().manual_seed(7)
    L = (torch.randn((32, N, N), generator=g) * 2.0).float()
    tile = pack_logits(L, torch.full((32, 32), 0.37))
    _read_ns(device)
    rows = []
    for iters in ITERS:
        for v in VARIANTS:
            ns = {}
            for reps in (1, reps_hi):
                run_variant(device, tile, v, reps, iters)
                ttnn.synchronize_device(device)
                ns[reps] = _read_ns(device)[-1]
            per = (ns[reps_hi] - ns[1]) / (reps_hi - 1)
            rows.append((v, iters, ns[1], ns[reps_hi], per))
    for v, iters, a, b, per in rows:
        print(f"PERF v{v} it{iters}: reps1 {a:.0f} ns, reps{reps_hi} {b:.0f} ns, per-sinkhorn {per / 1000:.3f} us")
