# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Isolated 1-tile-wide tilize bake-off (conv2d_3x3 padded_tilize tw == 1; compute only, one core, no data movement).

    NTB_BENCH=1 NTB_VARIANTS=base,b4,b4nz NTB_N=42 NTB_REPS=24 \
        scripts/run_safe_pytest.sh --run-all \
        ttnn/ttnn/operations/conv2d_3x3/perf_experiments/res4_narrow_tilize_batch/test_bench.py

bf16 row-major slab (32 channels, 64 B per pixel) -> bf8_b tiles, fp32 DEST, LoFi (the res4.x prelude's formats).
Every non-diag variant is gated on bit-exactness against `base` and PCC vs torch; diag_* are wrong by construction and
only timed. Perf (DEVICE KERNEL DURATION, in-process profiler) is printed, never asserted. Appends to
results_bench.jsonl.
"""

import os

# `import ttnn` walks every module under ttnn/operations; only touch the environment when asked.
if os.environ.get("NTB_BENCH") == "1":
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
KERNEL = HERE / "bench" / "bench_tilize.cpp"
RESULTS = HERE / "results_bench.jsonl"
TILE = 32
CB_SLAB, CB_OUT = 0, 16
SLAB_PX = 352
SPAN_PX = SLAB_PX - TILE - 1  # tile k reads pixels pix(k) .. +31, a window pair up to +33
STRIDE_PX = 7
_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"

# name -> (variant code, batch)
VARIANTS = {
    "base": (0, 1),
    "b1": (1, 1),
    "b2": (1, 2),
    "b3": (1, 3),
    "b4": (1, 4),
    "b4nz": (2, 4),
    "b2nz": (2, 2),
    "b4r": (3, 4),
    "b4lean": (4, 4),
    "b2lean": (4, 2),
    "win3": (30, 1),
    "fast1": (31, 1),
    "diag_nopack_b1": (10, 1),
    "diag_nopack_b4": (10, 4),
    "diag_packonly_b1": (11, 1),
    "diag_packonly_b4": (11, 4),
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


def _descriptor(tt_slab, tt_out, code, n, reps, batch):
    compute = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL),
        core_ranges=_core(),
        compile_time_args=[code, n, reps, batch, SPAN_PX, STRIDE_PX],
        defines=[],
        config=ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.LoFi,
            fp32_dest_acc_en=True,
            math_approx_mode=False,
            dst_full_sync_en=False,
        ),
    )
    cb_slab = ttnn.cb_descriptor_from_sharded_tensor(CB_SLAB, tt_slab)
    # One 2 KB bf16 page per 32 pixels, as conv2d_3x3's cb_im2col_rm / cb_halo_slab.
    cb_slab.format_descriptors = [
        ttnn.CBFormatDescriptor(buffer_index=CB_SLAB, data_format=ttnn.bfloat16, page_size=2 * TILE * TILE)
    ]
    cb_out = ttnn.cb_descriptor_from_sharded_tensor(CB_OUT, tt_out)
    return ttnn.ProgramDescriptor(kernels=[compute], semaphores=[], cbs=[cb_slab, cb_out])


@pytest.mark.skipif(os.environ.get("NTB_BENCH") != "1", reason="bench: run with NTB_BENCH=1")
def test_tilize_bench(device):
    import torch

    box, arch = socket.gethostname(), str(device.arch())
    n = int(os.environ.get("NTB_N", "42"))
    reps = int(os.environ.get("NTB_REPS", "24"))
    variants = os.environ.get("NTB_VARIANTS", "base,b4").split(",")
    tag = os.environ.get("NTB_TAG", "")
    lines = [f"narrow tilize bench box={box} arch={arch} 1 core, fp32 DEST half-sync, n={n} reps={reps}"]
    failures = []
    torch.manual_seed(7)
    slab = (torch.randn(SLAB_PX, TILE) * 3).to(torch.bfloat16)
    tt_slab = ttnn.from_torch(
        slab, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=_shard_cfg(SLAB_PX, TILE)
    )
    # tile k: taps kw = 0 / 2 (pixel + 2) of kernel-row pair k // 2, as padded_tilize's kw loop at d = 1
    pix = [((k // 2) * STRIDE_PX) % SPAN_PX + 2 * (k % 2) for k in range(n)]
    ref = torch.cat([slab[p : p + TILE] for p in pix]).float()
    base_out, base_ns = None, None
    for name in variants:
        code, batch = VARIANTS[name]
        tt_out = ttnn.allocate_tensor_on_device(
            ttnn.Shape([n * TILE, TILE]), ttnn.bfloat8_b, ttnn.TILE_LAYOUT, device, _shard_cfg(n * TILE, TILE)
        )
        desc = _descriptor(tt_slab, tt_out, code, n, reps, batch)
        ttnn.generic_op([tt_slab, tt_out], desc)  # compile + correctness run
        got = ttnn.to_torch(tt_out).float()
        _read_kernel_ns(device)  # flush
        ttnn.generic_op([tt_slab, tt_out], desc)
        ns = _read_kernel_ns(device)
        pcc = float(torch.corrcoef(torch.stack([got.flatten(), ref.flatten()]))[0, 1])
        if name == "base":
            base_out, base_ns = got, ns
        exact = None if base_out is None else bool(torch.equal(got, base_out))
        if not name.startswith("diag"):
            if pcc < 0.999:
                failures.append(f"{name}: pcc {pcc}")
            if exact is False:
                bad = (got != base_out).reshape(-1, TILE, TILE).any(dim=2).nonzero()
                failures.append(f"{name}: differs from base ({bad.shape[0]} bad rows, first {bad[:4].tolist()})")
        speed = f"{base_ns / ns:5.3f}x" if (base_ns and ns) else "  -  "
        per_tile = ns / (n * reps) if ns else 0
        lines.append(
            f"  {name:<18} {ns:>10.0f} ns  {speed}  {per_tile:6.1f} ns/tile  pcc {pcc:.6f}  "
            f"{'-' if exact is None else ('bit-exact vs base' if exact else 'DIFFERS')}"
        )
        with RESULTS.open("a") as f:
            f.write(
                json.dumps(
                    dict(
                        variant=name,
                        ns=ns,
                        n=n,
                        reps=reps,
                        ns_per_tile=per_tile,
                        pcc=pcc,
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
