# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Isolated bench (Perf 11 `tap_shift_rate`): conv2d_3x3's tap-shift prelude (one M-block's kw = 0 / 2 taps from the
input ring) on one core, HEAD's run() vs the knobbed candidate (bench/tsr.hpp). Compute only, the ring resident.

    TSR_BENCH=1 TSR_VARIANTS=base,b16 TSR_D=1 TSR_CT=1 TSR_W=64 TSR_H=32 TSR_ROWS=8 TSR_GH=4 TSR_T0=8 TSR_REPS=32 \
        scripts/run_safe_pytest.sh --run-all ttnn/ttnn/operations/conv2d_3x3/perf_experiments/tap_shift_rate/test_bench.py

bf8_b ring -> bf8_b taps, fp32 DEST, LoFi (the op's formats). Every non-diag variant is gated on bit-exactness against
the torch tap gather of the bf8_b input. Perf (DEVICE KERNEL DURATION, in-process profiler) is printed, never asserted.
Appends to results_bench.jsonl.
"""

import os

if os.environ.get("TSR_BENCH") == "1" and os.environ.get("TSR_PROF", "1") == "1":
    os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
    os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
    os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")
    os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "error")

import json
import socket
from pathlib import Path

import pytest

import ttnn

HERE = Path(__file__).parent
KERNEL = HERE / "bench" / "bench_tsr.cpp"
READER = HERE / "bench" / "bench_reader.cpp"
RESULTS = HERE / "results_bench.jsonl"
TILE = 32
CB_IN, CB_OUT = 0, 16
_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"

VARIANTS = {
    "base": 0,  # HEAD tap_shift::run
    "f32_imm": 1,  # tsr fp32, one-row immediates, 4 / section (HEAD with immediates instead of replays)
    "f32_packonly": 2,  # diag: pack only, fp32, 4 / section
    "f32_nomove": 3,  # diag: unpack + 1 move + release, fp32
    "f32_m4": 4,  # fp32 with 4-row moves at any alignment (expected wrong unless d % 8 == 0)
    "b16": 10,  # 16-bit DEST, one-row moves, 4 / section
    "b16_m4": 11,  # 16-bit DEST, 4-row moves, 4 / section
    "b16_m4_s8": 12,  # 16-bit DEST, 4-row moves, 8 / section
    "b16_packonly8": 13,  # diag
    "b16_nomove8": 14,  # diag
    "b16_packonly4": 15,  # diag
    "b16_s8": 16,  # 16-bit DEST, one-row moves, 8 / section
    "rowbuf": 20,  # 16-bit DEST row buffer (G = 2 units per slot), the packer reads the shifted windows
    "rowbuf_g1": 21,
    "rowbuf_g4": 22,
    "rb_packonly": 23,  # diag: rowbuf without loads / moves
    "rb_onemove": 24,  # diag: rowbuf loads, one move per load
    "rb_nopack": 25,  # diag: rowbuf loads + moves, no packs
    "rb_stdpack": 26,  # diag: rowbuf b16 with the standard pack MOP at the window (wrong faces 2 / 3)
    "rb_f32": 27,  # rowbuf in fp32 DEST (1 slot / section at G = 2)
    "rb_f32_stdpack": 28,  # diag
    "rb_f32_packonly": 29,  # diag
    "rb_uoff": 30,  # rowbuf b16, SrcB written at row offset sigma = d % 4 (even d): windows on multiples of 4
    "rb_uoff_g1": 31,
    "rb2": 40,  # fp32 row buffer, layout A (d % 4 == 0) / B (1-row moves, 2-row PACR for odd d), G = 2
    "rb2_packonly": 41,
    "rb2_nopack": 42,
    "rb2_g1": 43,
    "rb2_onemove": 44,  # diag: loads + one move per load, packs
    "rb2_loadsonly": 45,  # diag: loads + one move per load, no packs
    "empty": 99,
}
DIAG = {
    "rb2_onemove",
    "rb2_loadsonly",
    "rb2_packonly",
    "rb2_nopack",
    "rb_stdpack",
    "rb_f32_stdpack",
    "rb_f32_packonly",
    "rb_packonly",
    "rb_onemove",
    "rb_nopack",
    "f32_packonly",
    "f32_nomove",
    "b16_packonly8",
    "b16_nomove8",
    "b16_packonly4",
    "empty",
}


def _core():
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])


def _shard_cfg(h, w):
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(_core(), [h, w], ttnn.ShardOrientation.ROW_MAJOR),
    )


def _read_kernel_ns(device):
    ttnn.ReadDeviceProfiler(device)
    total, found = 0.0, False
    for programs in (ttnn.get_latest_programs_perf_data() or {}).values():
        for program in programs:
            entry = (getattr(program, "program_analyses_results", None) or {}).get(_DURATION_KEY)
            if entry is not None:
                total += float(entry.duration)
                found = True
    return total if found else None


def _plan(o_first, o_end, W, d, num_in_px):
    """padded_slab::plan's segments (stride 1, p == d): [(first input tile-row, count)]."""
    ho_first, wo_first = divmod(o_first, W)
    ho_last, wo_last = divmod(o_end - 1, W)
    f = (ho_first - d) * W + wo_first - d
    band_len = (ho_last - ho_first) * W + (wo_last - wo_first) + 2 * d + 1
    segs, have_prev, prev_t_hi = [], False, 0
    for kh in range(3):
        v_lo = f + kh * d * W
        lo, hi = max(v_lo, 0), min(v_lo + band_len, num_in_px)
        if lo >= hi:
            continue
        t_lo, t_hi = lo // TILE, (hi + TILE - 1) // TILE
        shares = have_prev and t_lo < prev_t_hi
        first_new = prev_t_hi if shares else t_lo
        if t_hi > first_new:
            segs.append((first_new, t_hi - first_new))
        prev_t_hi, have_prev = t_hi, True
    return segs


def _groups(rows, gh, o_first, W, H, d):
    out = []
    for g in range((rows + gh - 1) // gh):
        live = set()
        for t in range(g * gh, min((g + 1) * gh, rows)):
            ho = (o_first + 32 * t) // W
            live |= {kh for kh in range(3) if 0 <= ho + kh * d - d < H}
        out.append((min(live), max(live) + 1) if live else (0, 1))
    return out


def _reference(X, ct, rows, W, H, d, gh, o_first):
    """[rows * 32, 6 ct * 32] taps (unit (t, kh) at columns kh * 2 ct + {0, ct} + ch) and the written-tile mask."""
    import torch

    in0_k = 6 * ct
    ref = torch.zeros(rows * TILE, in0_k * TILE)
    mask = torch.zeros(rows, in0_k, dtype=torch.bool)
    groups = _groups(rows, gh, o_first, W, H, d)
    for t in range(rows):
        lo, hi = groups[t // gh]
        for kh in range(lo, hi):
            for kw in (0, 2):
                for ch in range(ct):
                    col = kh * 2 * ct + (kw // 2) * ct + ch
                    mask[t, col] = True
                    for r in range(TILE):
                        o = o_first + 32 * t + r
                        ho, wo = divmod(o, W)
                        h, w = ho + kh * d - d, wo + (kw - 1) * d
                        if 0 <= h < H and 0 <= w < W and o < H * W:
                            ref[t * TILE + r, col * TILE : (col + 1) * TILE] = X[h * W + w, ch * TILE : (ch + 1) * TILE]
    return ref, mask


def _descriptor(tt_in, tt_out, code, reps, ct, rows, W, H, d, gh, o_first, pages):
    compute = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL),
        core_ranges=_core(),
        compile_time_args=[code, reps, ct, rows, W, H, d, gh, o_first],
        defines=[],
        config=ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.LoFi,
            fp32_dest_acc_en=True,
            math_approx_mode=False,
            dst_full_sync_en=False,
        ),
    )
    reader = ttnn.KernelDescriptor(
        kernel_source=str(READER),
        core_ranges=_core(),
        compile_time_args=[CB_IN, pages],
        defines=[],
        config=ttnn.ReaderConfigDescriptor(),
    )
    cbs = [
        ttnn.cb_descriptor_from_sharded_tensor(CB_IN, tt_in),
        ttnn.cb_descriptor_from_sharded_tensor(CB_OUT, tt_out),
    ]
    return ttnn.ProgramDescriptor(kernels=[reader, compute], semaphores=[], cbs=cbs)


_DS = [int(v) for v in os.environ.get("TSR_D", "1").split(",")]


@pytest.mark.skipif(os.environ.get("TSR_BENCH") != "1", reason="bench: run with TSR_BENCH=1")
@pytest.mark.parametrize("d", _DS, ids=[f"d{v}" for v in _DS])
def test_tsr_bench(device, d):
    import torch

    box, arch = socket.gethostname(), str(device.arch())
    e = os.environ
    ct, W, H = int(e.get("TSR_CT", "1")), int(e.get("TSR_W", "64")), int(e.get("TSR_H", "32"))
    rows, gh = int(e.get("TSR_ROWS", "8")), int(e.get("TSR_GH", "4"))
    o_first = int(e.get("TSR_T0", "8")) * TILE
    reps = int(e.get("TSR_REPS", "32"))
    variants = e.get("TSR_VARIANTS", "base,b16").split(",")
    tag = e.get("TSR_TAG", "")
    o_end = min(o_first + rows * TILE, H * W)
    segs = _plan(o_first, o_end, W, d, H * W)
    slab_tiles = [i for f, c in segs for i in range(f, f + c)]
    S = len(slab_tiles)
    torch.manual_seed(7)
    img = (torch.randn(H * W, ct * TILE) * 3).to(torch.bfloat16)
    full = ttnn.from_torch(img, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
    X = ttnn.to_torch(full).float()  # the bf8_b values
    slab = torch.cat([X[i * TILE : (i + 1) * TILE] for i in slab_tiles]).to(torch.bfloat16)
    tt_in = ttnn.from_torch(
        slab,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=_shard_cfg(S * TILE, ct * TILE),
    )
    ref, mask = _reference(X, ct, rows, W, H, d, gh, o_first)
    n_taps = int(mask.sum())
    lines = [
        f"tap shift bench box={box} arch={arch} 1 core fp32 DEST LoFi: d={d} ct={ct} W={W} H={H} rows={rows} gh={gh} "
        f"o_first={o_first} slab={S} rows reps={reps} taps/rep={n_taps}"
    ]
    failures = []
    base_ns = None
    for name in variants:
        code = VARIANTS[name]
        tt_out = ttnn.from_torch(
            torch.full((rows * TILE, 6 * ct * TILE), 77.0),
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=_shard_cfg(rows * TILE, 6 * ct * TILE),
        )
        desc = _descriptor(tt_in, tt_out, code, reps, ct, rows, W, H, d, gh, o_first, S * ct)
        tensors = [tt_in, tt_out]
        ttnn.generic_op(tensors, desc)  # compile + correctness run
        got = ttnn.to_torch(tt_out).float()
        prof = e.get("TSR_PROF", "1") == "1"
        if prof:
            _read_kernel_ns(device)  # flush
        ttnn.generic_op(tensors, desc)
        ns = (_read_kernel_ns(device) or 0.0) if prof else 0.0
        exact = None
        nbad = None
        if name not in DIAG:
            m = mask.repeat_interleave(TILE, 0).repeat_interleave(TILE, 1)
            bad = (got != ref) & m
            nbad = int(bad.reshape(rows, TILE, 6 * ct, TILE).any(-1).sum())  # bad 32-wide rows
            exact = nbad == 0
            if not exact and e.get("TSR_DEBUG") == "1":
                # per tap tile (t, col): each 16-datum row of the output -> the (pixel, channel-half) of X it equals
                keys = {}
                for q in range(X.shape[0]):
                    for hf in range(2 * ct):
                        keys.setdefault(tuple(X[q, hf * 16 : hf * 16 + 16].tolist()), f"{q}.{hf}")
                for t, col in [(0, 2), (0, 3), (1, 2), (1, 3)]:
                    if not mask[t, col]:
                        continue
                    line = []
                    for hf in range(2):
                        for r in range(TILE):
                            v = got[t * TILE + r, col * TILE + hf * 16 : col * TILE + hf * 16 + 16]
                            w = ref[t * TILE + r, col * TILE + hf * 16 : col * TILE + hf * 16 + 16]
                            k = "Z" if bool((v == 0).all()) else keys.get(tuple(v.tolist()), "?")
                            line.append(("" if torch.equal(v, w) else "*") + k)
                    print(f"DBG {name} d={d} t={t} col={col}: " + " ".join(line))
            if not exact:
                rows_bad = bad.reshape(rows, TILE, 6 * ct, TILE).any(-1).nonzero()[:8].tolist()
                failures.append(f"{name}: {nbad} bad tile rows, first (t, r, col) {rows_bad}")
        if name == "base":
            base_ns = ns
        speed = f"{base_ns / ns:5.3f}x" if (base_ns and ns) else "  -  "
        per_tile = ns / (n_taps * reps) if ns and n_taps else 0
        lines.append(
            f"  {name:<14} {ns:>10.0f} ns  {speed}  {per_tile:6.1f} ns/tap-tile  "
            f"{'-' if exact is None else ('bit-exact vs torch' if exact else f'DIFFERS ({nbad} rows)')}"
        )
        with RESULTS.open("a") as f:
            f.write(
                json.dumps(
                    dict(
                        variant=name,
                        ns=ns,
                        d=d,
                        ct=ct,
                        W=W,
                        H=H,
                        rows=rows,
                        gh=gh,
                        o_first=o_first,
                        reps=reps,
                        taps=n_taps,
                        ns_per_tap=per_tile,
                        exact=exact,
                        box=box,
                        arch=arch,
                        tag=tag,
                    )
                )
                + "\n"
            )
        tt_out.deallocate()
    print("\n" + "\n".join(lines))
    assert not failures, failures
