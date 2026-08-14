# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Bake-off harness for the RETILE *direct* permutation (isolated, op untouched).

IDEA
----
Perf 1 measured a 4.19x "direct" retile — the reader lands the face permutation
straight in the OUTPUT TILE, so the row-major intermediate and the compute tilize
both disappear — and declined it because it needed `out_dtype == in_dtype` PLUS a
DRAM-alignment predicate, i.e. a four-way host dispatch. This bench looks for the
WIDEST single-path form of that arm: how few carve-outs can it ship with?

ISOLATION MODEL
---------------
The idea lives entirely inside the reader's `R_RETILE` branch plus the compute
kernel's shape, so the bench swaps ONLY those: `tilize_program_descriptor.KERNEL_DIR`
is pointed at a generated per-arm shim directory whose `tilize_reader.cpp` and
`tilize_compute.cpp` are one-line `#define RETILE_ARM n` + include of this dir's
`experiment_kernels/`, and whose `tilize_writer.cpp` is a one-line include of the
op's REAL writer. `experiment_kernels/retile_reader.cpp` is a copy of TODAY's op
reader with only the retile branch replaced by the arm switch, so arm 0 is the
op's current approach bit-for-bit.

Everything else — blocking, work split, CB sizes, grid, dtypes, precision config —
is the op's own host code, unmodified.

The program-descriptor hash includes `kernel_source`, so each arm is a distinct
program-cache entry: no arm can silently re-measure another.

Nothing in this directory is imported by the op.

Box/arch: bgd-lab-16 (`ARCH_NAME=wormhole_b0`), DRAM NoC alignment 32 B.
"""

import os

# In-process on-device profiler — all three, before the device opens.
os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")

from pathlib import Path  # noqa: E402


# `ttnn/` may not import torch at module scope (scripts/validate_no_global_torch_imports.py).
def _load_torch():
    global torch
    import torch


_load_torch()
import ttnn  # noqa: E402
from loguru import logger  # noqa: E402

from ttnn.operations.tilize import tilize  # noqa: E402
from ttnn.operations.tilize import tilize_program_descriptor as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[5]
KERNELS = HERE / "experiment_kernels"
GEN = HERE / "generated"

_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"

# arm id -> (slug, one-line description, predicate it needs to be CORRECT)
VARIANTS = {
    0: ("baseline", "op today: staged pages -> RM sticks (local NoC), real tilize compute", "none"),
    1: ("direct_dram", "DRAM -> OUTPUT TILE face runs, no staging, compute no-op", "same dtype + run%DRAM_ALIGN"),
    2: ("direct_noc", "staged page -> OUTPUT TILE face runs (local NoC), compute no-op", "same dtype"),
    3: ("direct_dram_cast", "DRAM -> output-shaped tile in INPUT dtype; compute datacopies/casts", "run%DRAM_ALIGN"),
    4: ("direct_noc_cast", "staged page -> output-shaped tile in INPUT dtype; compute datacopies", "none"),
    5: ("direct_dram_merge", "(1) + FULL-WIDTH runs (both column halves fuse)", "same dtype, src_face==out_face"),
    6: ("direct_noc_merge", "(2) + FULL-WIDTH runs", "same dtype, src_face==out_face"),
}

FACE_W = 16
DRAM_ALIGN = 32  # wormhole_b0


def _elem_bytes(dtype):
    return {
        ttnn.bfloat16: 2,
        ttnn.float32: 4,
        ttnn.uint8: 1,
        ttnn.uint16: 2,
        ttnn.uint32: 4,
        ttnn.int32: 4,
    }[dtype]


def geometry(in_tile_h, tile_h, dtype=ttnn.bfloat16):
    """The compile-time run geometry the kernel derives, mirrored on the host —
    this is exactly what a program descriptor would have to compute to pick an arm."""
    eb = _elem_bytes(dtype)
    src_face_h = min(in_tile_h, FACE_W)
    out_face_h = min(tile_h, FACE_W)
    rows_per_run = min(src_face_h, out_face_h)
    run_bytes = rows_per_run * FACE_W * eb
    merge_ok = src_face_h == out_face_h
    merge_run_bytes = min(in_tile_h, tile_h) * 32 * eb
    return {
        "run_bytes": run_bytes,
        "dram_aligned": run_bytes % DRAM_ALIGN == 0,
        "merge_ok": merge_ok,
        "merge_run_bytes": merge_run_bytes if merge_ok else None,
    }


def arms_for(in_tile_h, tile_h, dtype=ttnn.bfloat16):
    """Arms worth RUNNING on this geometry. The merge arms compile to arms 1/2
    when src_face_h != out_face_h, so running them there would re-measure 1/2."""
    g = geometry(in_tile_h, tile_h, dtype)
    arms = [0, 1, 2, 3, 4]
    if g["merge_ok"]:
        arms += [5, 6]
    return arms


def _shim_dir(variant):
    """Generate (once) the per-arm kernel shim directory and return it."""
    d = GEN / f"v{variant}"
    d.mkdir(parents=True, exist_ok=True)
    rel_reader = (KERNELS / "retile_reader.cpp").relative_to(REPO).as_posix()
    rel_compute = (KERNELS / "retile_compute.cpp").relative_to(REPO).as_posix()
    (d / "tilize_reader.cpp").write_text(f'#define RETILE_ARM {variant}\n#include "{rel_reader}"\n')
    (d / "tilize_compute.cpp").write_text(f'#define RETILE_ARM {variant}\n#include "{rel_compute}"\n')
    (d / "tilize_writer.cpp").write_text('#include "ttnn/ttnn/operations/tilize/kernels/tilize_writer.cpp"\n')
    return d


def _read_kernel_ns(device):
    ttnn.ReadDeviceProfiler(device)
    per_chip = ttnn.get_latest_programs_perf_data()
    total, found = 0.0, False
    for programs in (per_chip or {}).values():
        for program in programs:
            results = getattr(program, "program_analyses_results", None) or {}
            entry = results.get(_DURATION_KEY)
            if entry is None:
                continue
            total += float(entry.duration)
            found = True
    return total if found else None


def height_shard(shape, cores):
    """Height-sharded L1 config over `cores` cores — the W_REGION work mode."""
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(cores - 1, 0))])
    rows = 1
    for d in shape[:-1]:
        rows *= d
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, [rows // cores, shape[-1]], ttnn.ShardOrientation.ROW_MAJOR),
    )


def _torch_in(shape, dtype):
    if dtype in (ttnn.uint8, ttnn.uint16, ttnn.uint32, ttnn.int32):
        tdt = {ttnn.uint8: torch.uint8}.get(dtype, torch.int32)
        return torch.randint(0, 100, shape, dtype=torch.int32).to(tdt)
    return torch.randn(shape).to(torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32)


# Oracle cache for the `oracle="baseline"` mode: arm 0's own device output.
# Needed for a NARROWING cast (fp32 -> bf16), where the packer's rounding is not
# torch's `.to(bfloat16)` — the op itself is not bit-equal to torch there, so the
# only meaningful correctness bar for the other arms is "same bytes as the op".
_BASELINE_OUT = {}


def run(
    device,
    variant,
    shape,
    in_tile_h,
    tile_h,
    dtype=ttnn.bfloat16,
    out_dtype=None,
    measure=True,
    out_mem_config=None,
    in_mem_config=None,
    oracle="torch",
):
    """One arm on one retile case. -> (ns_or_None, bit_exact_bool).

    ONE fresh-cache measured launch per arm (device kernel duration has no
    warm-up transient); the preceding launch only fills the program/kernel cache.

    `out_dtype` != `dtype` is the CASTING retile — the case the plain direct arms
    cannot express, and the reason arms 3/4 exist.
    """
    saved_dir = pd.KERNEL_DIR
    pd.KERNEL_DIR = _shim_dir(variant)
    ns, exact, err = None, False, None
    key = (tuple(shape), in_tile_h, tile_h, str(dtype), str(out_dtype), str(out_mem_config), str(in_mem_config))
    try:
        if oracle == "baseline":
            torch.manual_seed(0)  # every arm gets the SAME input on this cell
        torch_in = _torch_in(shape, dtype)
        tt_in = ttnn.from_torch(
            torch_in,
            dtype=dtype,
            device=device,
            memory_config=in_mem_config if in_mem_config is not None else ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
            tile=ttnn.Tile([in_tile_h, 32]),
        )
        call = dict(tile=ttnn.Tile([tile_h, 32]))
        if out_mem_config is not None:
            call["memory_config"] = out_mem_config
        if out_dtype is not None:
            call["dtype"] = out_dtype
        ref = (
            torch_in
            if out_dtype is None
            else torch_in.to(torch.float32 if out_dtype == ttnn.float32 else torch.bfloat16)
        )
        out = tilize(tt_in, **call)
        ttnn.synchronize_device(device)
        # tilize is a PERMUTATION (plus at most a value-preserving cast): the bar
        # is bit-exact, not a PCC.
        got = ttnn.to_torch(out)
        if oracle == "baseline":
            if variant == 0:
                _BASELINE_OUT[key] = got.clone()
                exact = True
            else:
                exact = bool(torch.equal(got, _BASELINE_OUT[key]))
        else:
            exact = bool(torch.equal(got, ref))
        if measure:
            _read_kernel_ns(device)  # flush the warm-up window
            out = tilize(tt_in, **call)
            ttnn.synchronize_device(device)
            ns = _read_kernel_ns(device)
    except Exception as exc:  # a REFUSED case is data, not a crash
        err = f"{type(exc).__name__}: {str(exc)[:160]}"
    finally:
        pd.KERNEL_DIR = saved_dir
    slug = VARIANTS[variant][0]
    cast = "" if out_dtype is None else f" ->{out_dtype}"
    logger.info(
        f"RETILE-DIRECT {in_tile_h}->{tile_h} {list(shape)} {dtype}{cast} "
        f"arm={variant}:{slug} ns={ns} bit_exact={exact}" + (f" ERR={err}" if err else "")
    )
    return ns, exact


def table(rows, header):
    base = next((ns for v, _s, ns, _e in rows if v == 0), None)
    out = [f"=== {header}  (baseline {base} ns) ==="]
    for variant, slug, ns, exact in sorted(rows, key=lambda r: (r[2] is None, r[2] or 0)):
        speed = f"x{base / ns:5.2f}" if (base and ns) else "   -  "
        shown = f"{ns:10.0f}" if ns else "       n/a"
        out.append(f"  {variant} {slug:18s} {shown} ns   {speed}   bit_exact={exact}")
    logger.info("\n".join(out))
    return out
