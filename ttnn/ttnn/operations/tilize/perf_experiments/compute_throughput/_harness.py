# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Bake-off harness for TRISC compute throughput (isolated; the op is untouched).

ISOLATION MODEL
---------------
The idea under test lives entirely inside the compute kernel, so the bench swaps
ONLY that file: `tilize_program_descriptor.KERNEL_DIR` is pointed at a generated
per-arm shim directory whose `tilize_compute.cpp` is

    #define CT_VARIANT <n>
    #include ".../compute_throughput/experiment_kernels/ct_compute.cpp"

while `tilize_reader.cpp` / `tilize_writer.cpp` are one-line includes of the op's
REAL reader and writer.  Host blocking, CB geometry, grid, dtypes and the entire
ComputeConfig (fp32_dest_acc_en, dst sync mode, fidelity, unpack_to_dest_mode)
are the op's own, unmodified — so arm 0 is the op's current approach bit-for-bit
and no arm can move the user's precision contract.

CONCEPT ISOLATION.  The focus case is the same-spec HEIGHT-sharded L1 plan, where
BOTH CBs are aliased on the resident shard: the reader issues no NoC read and the
writer no NoC write, so the reader/writer stages are held trivial and constant and
the measured wall is the TRISC pipeline.  Every regime below is run in that same
zero-NoC configuration for the same reason.

The program-descriptor hash includes `kernel_source`, so each arm is a distinct
program-cache entry: no arm can silently re-measure another.

Nothing in this directory is imported by the op.
"""

import os

# In-process on-device profiler — all three, before the device opens.
os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")

from pathlib import Path  # noqa: E402


# `ttnn/` may not import torch at module scope; these benches need it for the
# bit-exact oracle, so publish it from a function scope.
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

VARIANTS = {
    0: ("baseline", "op's current compute kernel: kernel_lib tilize helper, WaitBlock"),
    1: ("wait_upfront", "helper, WaitMode::WaitUpfront"),
    2: ("nowait", "helper, WaitMode::NoWait — handshake floor (zero-copy input only)"),
    3: ("payload_floor", "raw LLK, NO CB handshake — pure payload; output WRONG by construction"),
    4: ("wide_dest", "CANDIDATE: regular tilize with a full DEST section per acquire"),
    5: ("raw_regular_ctl", "control for arm 4: same open-coded loop, stock tilize_block"),
    6: ("wide_dest_w2", "arm 4 with DEST window = 2"),
    7: ("wide_dest_half", "arm 4 with DEST window / 2"),
    8: ("wide_dest_x2", "arm 4 with 2x the DEST window (probe: is it legal?)"),
    9: ("wide_dest_x4", "arm 4 with 4x the DEST window (probe: is it legal?)"),
}

# Arms whose output is not the op's output by construction — measured for the
# floor, never graduated.
NON_GRADUABLE = {3}


def _source_tag():
    """Hash of the arm sources, stamped into the shim.

    The JIT kernel cache keys on the file the KernelDescriptor names — the shim —
    NOT on what it #includes.  Without this, editing `ct_compute.cpp` silently
    re-measures the STALE cached binary (observed once, on the uint32 arm).
    """
    import hashlib

    h = hashlib.sha1()
    for name in ("ct_compute.cpp", "tilize_wide.hpp"):
        h.update((KERNELS / name).read_bytes())
    return h.hexdigest()[:12]


def _shim_dir(variant):
    d = GEN / f"v{variant}"
    d.mkdir(parents=True, exist_ok=True)
    rel = (KERNELS / "ct_compute.cpp").relative_to(REPO).as_posix()
    (d / "tilize_compute.cpp").write_text(f'// src {_source_tag()}\n#define CT_VARIANT {variant}\n#include "{rel}"\n')
    (d / "tilize_reader.cpp").write_text('#include "ttnn/ttnn/operations/tilize/kernels/tilize_reader.cpp"\n')
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


def height_shard(shape, cores, dtype_bytes=None):
    """Same-spec HEIGHT-sharded L1 config over `cores` cores (zero-copy both sides)."""
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(cores - 1, 0))])
    rows = 1
    for d in shape[:-1]:
        rows *= d
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, [rows // cores, shape[-1]], ttnn.ShardOrientation.ROW_MAJOR),
    )


_TORCH_DT = {
    ttnn.bfloat16: "bfloat16",
    ttnn.float32: "float32",
    ttnn.uint8: "uint8",
    ttnn.uint16: "int32",
    ttnn.uint32: "int32",
    ttnn.int32: "int32",
}


def _make_input(shape, dtype):
    if dtype == ttnn.uint8:
        return torch.randint(0, 200, shape, dtype=torch.uint8)
    if dtype in (ttnn.uint16, ttnn.uint32, ttnn.int32):
        return torch.randint(0, 10000, shape, dtype=torch.int32)
    if dtype == ttnn.float32:
        return torch.randn(shape).to(torch.float32)
    return torch.randn(shape).to(torch.bfloat16)


def run(
    device,
    variant,
    shape,
    *,
    cores=None,
    dtype=ttnn.bfloat16,
    out_dtype=None,
    tile_h=32,
    sharded=True,
    measure=True,
    check=True,
    label="",
):
    """One arm on one case -> (ns_or_None, bit_exact_or_None).

    ONE fresh-cache measured launch per arm (device kernel duration has no
    warm-up transient); the preceding launch only fills the program/kernel cache.
    """
    saved_dir = pd.KERNEL_DIR
    pd.KERNEL_DIR = _shim_dir(variant)
    try:
        torch_in = _make_input(shape, dtype)
        mem_in = height_shard(shape, cores) if sharded else ttnn.DRAM_MEMORY_CONFIG
        tt_in = ttnn.from_torch(
            torch_in,
            dtype=dtype,
            device=device,
            memory_config=mem_in,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        call = dict(tile=ttnn.Tile([tile_h, 32]))
        if sharded:
            call["memory_config"] = height_shard(shape, cores)
        if out_dtype is not None:
            call["dtype"] = out_dtype

        out = tilize(tt_in, **call)
        ttnn.synchronize_device(device)
        exact = None
        if check:
            got = ttnn.to_torch(out)
            ref = torch_in
            if out_dtype is not None and out_dtype != dtype:
                # value-preserving cast oracle
                ref = ttnn.to_torch(ttnn.from_torch(torch_in, dtype=out_dtype))
            exact = bool(torch.equal(got.to(ref.dtype), ref))
        ns = None
        if measure:
            _read_kernel_ns(device)  # flush the warm-up window
            out = tilize(tt_in, **call)
            ttnn.synchronize_device(device)
            ns = _read_kernel_ns(device)
    finally:
        pd.KERNEL_DIR = saved_dir
    slug = VARIANTS[variant][0]
    logger.info(f"CT-BAKEOFF {label} {list(shape)} arm={variant}:{slug} ns={ns} bit_exact={exact}")
    return ns, exact
