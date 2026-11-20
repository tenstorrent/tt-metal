# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""fill_noc_split bench: the conv2d_3x3 ROW_MAJOR interleaved slab fill of the FLUX.2 focus cases, alone, on the
op's 110 cores (Blackhole p150b, 11 x 10). Every core walks its M-blocks exactly as the op's K-blocked tap-direct walk
does (core_m 596 tiles, bm tiles per block, three disjoint kernel-row bands of bm*32 + 2 pixels, one 64/128/256 B DRAM
page per pixel) and waits for each block's slab before the next one. Variants change only how the pages are read (see
kernels/fill_bench.cpp). Correctness: every core's last slab is written to a dump tensor and compared bit-exactly with
the input pixels the op's slab_bands places there. Timing: DEVICE KERNEL DURATION (kernel span) and the "fill_loop"
zone, via scripts/run_safe_pytest.sh --profile (see run.sh / report.py)."""

import math
import sys
from pathlib import Path

import pytest
import ttnn

KDIR = Path(__file__).resolve().parent / "kernels"
GRID_X, GRID_Y = 11, 10
NCORES = GRID_X * GRID_Y
CORE_M = 596

# name: (H incl. halo rows, W, C_in, bm, C_out tiles)
PATTERNS = {
    "p1": (1026, 2048, 32, 50, 4),  # idx41 1026x2048 32->128
    "p5": (1026, 2048, 64, 38, 4),  # idx44 1026x2048 64->128
    "p3": (1026, 2048, 128, 34, 1),  # idx35 1026x2048 128->3
}


def plan(H, W, bm, core_m=CORE_M, ncores=NCORES):
    """Per core, per M-block, the three band pieces (page0, count, slab_px) the op's slab_bands reads
    (stride 1, dilation 1, padding top 0 / left 1, band_last_tap 2)."""
    H_out, W_out = H - 2, W
    num_out_px, num_in_px = H_out * W_out, H * W
    tensor_m_tiles = math.ceil(num_out_px / 32)
    blocks = {}
    max_band = 0
    for i in range(ncores):
        m_tile_start = i * core_m
        actual = min(core_m, tensor_m_tiles - m_tile_start)
        if actual <= 0:
            blocks[i] = []
            continue
        out = []
        for mb in range(math.ceil(actual / bm)):
            done = mb * bm
            valid = min(actual - done, bm)
            o_first = (m_tile_start + done) * 32
            o_end = min(o_first + valid * 32, num_out_px)
            ho_f, wo_f = divmod(o_first, W_out)
            ho_l, wo_l = divmod(o_end - 1, W_out)
            f = ho_f * W + wo_f - 1
            band_len = (ho_l - ho_f) * W + (wo_l - wo_f) + 3
            max_band = max(max_band, band_len)
            out.append((f, band_len))
        blocks[i] = out
    band_stride = min(W, max_band)
    pieces = {}
    for i, out in blocks.items():
        pcs = []
        for f, band_len in out:
            row = []
            for kh in range(3):
                v_minus_x = f + kh * W - kh * band_stride
                seg_lo = kh * band_stride
                seg_hi = kh * band_stride + band_len
                x_lo = max(seg_lo, -v_minus_x)
                x_hi = min(seg_hi, num_in_px - v_minus_x)
                row.append((x_lo + v_minus_x, max(0, x_hi - x_lo), x_lo))
            pcs.append(row)
        pieces[i] = pcs
    return pieces, 3 * band_stride


def split_pieces(row, frac):
    """One block's pieces into (reader, writer): each band's first (1 - frac) to the reader, the rest to the writer."""
    rd, wr = [], []
    for page0, count, px in row:
        w = int(round(count * frac))
        r = count - w
        rd.append((page0, r, px))
        wr.append((page0 + r, w, px + r))
    return rd, wr


# name: dict(mode, window, split fraction (0 = reader only), store, store_mode (0 first, 1 late, 2 mixed), cores ("all" / "one" / "q"),
#            wmode (writer address mode, default = mode))
VARIANTS = {
    "base": dict(mode=0, window=0, frac=0.0),
    "split": dict(mode=0, window=0, frac=0.5),
    "inc": dict(mode=1, window=0, frac=0.0),
    "inc_split": dict(mode=1, window=0, frac=0.5),
    "bank": dict(mode=2, window=0, frac=0.0),
    "bank_split": dict(mode=2, window=0, frac=0.5),
    "w8": dict(mode=0, window=8, frac=0.0),
    "w32": dict(mode=0, window=32, frac=0.0),
    "local": dict(mode=3, window=0, frac=0.0),
    "base_1core": dict(mode=0, window=0, frac=0.0, cores="one"),
    "inc_1core": dict(mode=1, window=0, frac=0.0, cores="one"),
    "split_1core": dict(mode=0, window=0, frac=0.5, cores="one"),
    "base_q": dict(mode=0, window=0, frac=0.0, cores="q"),
    "base_store": dict(mode=0, window=0, frac=0.0, store=True),
    "split_store": dict(mode=0, window=0, frac=0.5, store=True, store_mode=0),
    "split_store_late": dict(mode=0, window=0, frac=0.5, store=True, store_mode=1),
    "split33_store": dict(mode=0, window=0, frac=0.33, store=True, store_mode=0),
    "inc_split_store": dict(mode=1, window=0, frac=0.5, store=True, store_mode=0),
    "bank_split_store": dict(mode=2, window=0, frac=0.5, store=True, store_mode=0),
    # lock-step (every block released when every core finished the previous one, as the op's weight rounds do)
    "base_ls": dict(mode=0, window=0, frac=0.0, ls=True),
    "split_ls": dict(mode=0, window=0, frac=0.5, ls=True),
    "inc_ls": dict(mode=1, window=0, frac=0.0, ls=True),
    "inc_split_ls": dict(mode=1, window=0, frac=0.5, ls=True),
    "w32_ls": dict(mode=0, window=32, frac=0.0, ls=True),
    "w32_split_ls": dict(mode=0, window=32, frac=0.5, ls=True),
    "base_store_ls": dict(mode=0, window=0, frac=0.0, store=True, ls=True),
    "split_store_ls": dict(mode=0, window=0, frac=0.5, store=True, store_mode=0, ls=True),
    "split_store_late_ls": dict(mode=0, window=0, frac=0.5, store=True, store_mode=1, ls=True),
    "split40_store_ls": dict(mode=0, window=0, frac=0.4, store=True, store_mode=0, ls=True),
    "split60_ls": dict(mode=0, window=0, frac=0.6, ls=True),
    "dyn_ls": dict(mode=0, window=0, frac=0.5, dyn=64, ls=True),
    "dyn_store_mix_ls": dict(mode=0, window=0, frac=0.5, dyn=64, store=True, store_mode=2, ls=True),
    "dyn_store_late_ls": dict(mode=0, window=0, frac=0.5, dyn=64, store=True, store_mode=1, ls=True),
    "dyn_store_first_ls": dict(mode=0, window=0, frac=0.5, dyn=64, store=True, store_mode=0, ls=True),
    "dyn256_store_mix_ls": dict(mode=0, window=0, frac=0.5, dyn=256, store=True, store_mode=2, ls=True),
    "split_store_mix_ls": dict(mode=0, window=0, frac=0.5, store=True, store_mode=2, ls=True),
    "split55_store_late_ls": dict(mode=0, window=0, frac=0.55, store=True, store_mode=1, ls=True),
    "split55_store_mix_ls": dict(mode=0, window=0, frac=0.55, store=True, store_mode=2, ls=True),
    "split40_ls": dict(mode=0, window=0, frac=0.4, ls=True),
}

_inputs = {}


def _pass():
    # run_safe_pytest's precompile pass runs every body under NO_DISPATCH with shape-only tensors: never reuse one.
    m = sys.modules.get("tests.plugins.up_front_collect")
    return getattr(m, "_PASS", "real") if m else "real"


def _input(device, pattern):
    import torch

    name = (pattern, _pass())
    if name not in _inputs:
        _inputs.clear()
        H, W, C, _, _ = PATTERNS[pattern]
        torch.manual_seed(0)
        x = torch.randn(1, 1, H * W, C).to(torch.bfloat16)
        t = ttnn.from_torch(
            x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        _inputs[name] = (x, t)
    return _inputs[name]


def run_variant(device, pattern, variant):
    import torch

    H, W, C, bm, out_n = PATTERNS[pattern]
    v = VARIANTS[variant]
    x, tin = _input(device, pattern)
    page_bytes = C * 2
    pieces, slab_px = plan(H, W, bm)
    cores_sel = v.get("cores", "all")
    active = {
        "all": set(range(NCORES)),
        "one": {0},
        "q": set(range(0, NCORES, 4)),
    }[cores_sel]
    frac = v["frac"]
    split = frac > 0
    store = v.get("store", False)
    store_tiles = bm * out_n if store else 0
    use_writer = split or store

    dump = ttnn.from_torch(
        torch.zeros(1, 1, NCORES * slab_px, C, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    store_t = ttnn.from_torch(
        torch.zeros(1, 1, max(1, NCORES * store_tiles), 1024, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    cores = [ttnn.CoreCoord(i % GRID_X, i // GRID_X) for i in range(NCORES)]
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(GRID_X - 1, GRID_Y - 1))])
    cb = ttnn.CBDescriptor(
        total_size=slab_px * page_bytes,
        core_ranges=crs,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.bfloat16, page_size=page_bytes)],
    )
    sems = [ttnn.SemaphoreDescriptor(id=s, core_ranges=crs, initial_value=0) for s in (0, 1, 2, 3, 4, 5)]
    lockstep = v.get("ls", False)
    noc0_xy = {}
    for i, c in enumerate(cores):
        p = device.worker_core_from_logical_core(c)
        noc0_xy[i] = (p.x << 16) | p.y
    act_sorted = sorted(active)
    acc = (
        list(ttnn.TensorAccessorArgs(tin).get_compile_time_args())
        + list(ttnn.TensorAccessorArgs(dump).get_compile_time_args())
        + list(ttnn.TensorAccessorArgs(store_t).get_compile_time_args())
    )
    kernels = []
    expect = {}
    for role, cfg in ((0, ttnn.ReaderConfigDescriptor()), (1, ttnn.WriterConfigDescriptor())):
        if role == 1 and not use_writer:
            continue
        mode = v["mode"] if role == 0 else v.get("wmode", v["mode"])
        ct = [0, page_bytes, mode, v["window"], role, int(split), 0, 1, store_tiles, 2048, int(v.get("store_mode", 0))]
        ct += [int(lockstep), 2, 3, list(VARIANTS).index(variant), v.get("dyn", 0), 4, 5]
        rt = ttnn.RuntimeArgs()
        max_blocks = max(len(pieces[i]) for i in active)
        for i, c in enumerate(cores):
            blocks = pieces[i] if i in active else []
            if lockstep and blocks:
                # every core joins every lock-step barrier: a core with fewer blocks runs empty ones
                blocks = blocks + [[(0, 0, 0)] * 3] * (max_blocks - len(blocks))
            args = [
                tin.buffer_address(),
                dump.buffer_address(),
                store_t.buffer_address(),
                i * slab_px,
                slab_px,
                len(blocks),
                i * store_tiles,
                i % 8,
                noc0_xy[act_sorted[0]],
                len(active) if (lockstep and role == 0 and blocks and i == act_sorted[0]) else 0,
                0,
            ]
            for row in blocks:
                if v.get("dyn"):
                    rd, wr = row, row
                else:
                    rd, wr = split_pieces(row, frac) if split else (row, [(0, 0, 0)] * 3)
                for page0, count, px in rd if role == 0 else wr:
                    args += [page0, count, px]
            if args[9]:
                args[10] = len(args)
                args += [noc0_xy[j] for j in act_sorted]
            rt[c.x][c.y] = args
            if blocks:
                expect[i] = pieces[i][-1]
        kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=str(KDIR / "fill_bench.cpp"),
                core_ranges=crs,
                compile_time_args=ct + acc,
                runtime_args=rt,
                config=cfg,
            )
        )
    desc = ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=[cb])
    ttnn.generic_op([tin, dump, store_t], desc)
    ttnn.ReadDeviceProfiler(device)
    got = ttnn.to_torch(dump).reshape(NCORES * slab_px, C)
    if v["mode"] == 3:
        return  # diagnostic: the source is local L1, nothing to check
    xf = x.reshape(-1, C)
    bad = 0
    for i, row in expect.items():
        for page0, count, px in row:
            if count and not torch.equal(got[i * slab_px + px : i * slab_px + px + count], xf[page0 : page0 + count]):
                bad += 1
    assert bad == 0, f"{pattern}/{variant}: {bad} slab pieces differ"


MAIN = ("base_ls", "split_ls", "dyn_ls", "base_store_ls", "split_store_late_ls", "dyn_store_mix_ls")
SWEEP = [("p1", v) for v in VARIANTS] + [(p, v) for p in ("p5", "p3") for v in MAIN]


@pytest.mark.parametrize("pattern,variant", SWEEP, ids=[f"{p}-{v}" for p, v in SWEEP])
def test_fill(pattern, variant, device):
    run_variant(device, pattern, variant)
