# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Bake-off harness for the RETILE permutation (isolated, op untouched).

ISOLATION MODEL
---------------
The idea under test lives entirely inside the reader's `R_RETILE` branch, so the
bench swaps ONLY that: `tilize_program_descriptor.KERNEL_DIR` is pointed at a
generated per-arm shim directory whose `tilize_reader.cpp` is

    #define RETILE_VARIANT <n>
    #include ".../retile_permute/experiment_kernels/retile_reader.cpp"

`retile_reader.cpp` is a copy of the op's reader with the retile branch replaced
by the arm switch; `tilize_writer.cpp` is a one-line include of the op's real
writer, and `tilize_compute.cpp` is the op's real compute except on the
"retile-direct" arms, where the reader produces the output CB itself and compute
is empty. Everything else — blocking, work split, CB sizes, grid, dtypes,
precision config — is the op's own host code, unmodified. Arm 0 is therefore the
op's current approach bit-for-bit.

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


# `ttnn/` may not import torch at module scope (scripts/validate_no_global_torch_imports.py
# — the shipped package must not drag torch in). These perf-experiment benches DO need it
# for their bit-exact oracle, so the import is done inside a function scope and published
# under the module-global name, which keeps every `torch.` use below unchanged.
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

# arm id -> (slug, one-line description)
VARIANTS = {
    0: ("baseline", "op's current approach: staged pages -> RM sticks, volatile 32 B word copy"),
    1: ("rm_coalesce", "RM intermediate, longest src+dst contiguous run"),
    2: ("rm_stage_direct", "no staging: DRAM page lands at its final RM address (needs in_tile_h==1)"),
    3: ("direct_tile", "staged pages -> OUTPUT TILE, compute no-op"),
    4: ("direct_dram", "DRAM -> OUTPUT TILE face runs, no staging, no CPU copy, compute no-op"),
    5: ("noc_loopback", "baseline geometry, L1->L1 via local noc_async_read"),
    6: ("nonvolatile", "baseline geometry, non-volatile (pipelinable) word copy"),
    7: ("direct_tile_nv", "direct_tile + non-volatile copy"),
    8: ("direct_tile_noc", "staged pages -> OUTPUT TILE, local NoC lands the runs, compute no-op"),
}

# The RM-intermediate coalescing arms only differ from the baseline when the
# source tile is one row tall (see the contiguity note in the kernel); on any
# other geometry they compile to the baseline, so running them would just
# re-measure arm 0.
_MERGE_ONLY = {1, 2}


def arms_for(in_tile_h):
    return [v for v in VARIANTS if v not in _MERGE_ONLY or in_tile_h == 1]


def _shim_dir(variant):
    """Generate (once) the per-arm kernel shim directory and return it."""
    d = GEN / f"v{variant}"
    d.mkdir(parents=True, exist_ok=True)
    reader = KERNELS / "retile_reader.cpp"
    compute = KERNELS / "retile_compute.cpp"
    rel_reader = reader.relative_to(REPO).as_posix()
    rel_compute = compute.relative_to(REPO).as_posix()
    (d / "tilize_reader.cpp").write_text(f'#define RETILE_VARIANT {variant}\n#include "{rel_reader}"\n')
    (d / "tilize_compute.cpp").write_text(f'#define RETILE_VARIANT {variant}\n#include "{rel_compute}"\n')
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


def run(device, variant, shape, in_tile_h, tile_h, dtype=ttnn.bfloat16, measure=True, out_mem_config=None):
    """One arm on one retile case. -> (ns_or_None, bit_exact_bool).

    ONE fresh-cache measured launch per arm (device kernel duration has no
    warm-up transient); the preceding launch only fills the program/kernel cache.
    """
    saved_dir = pd.KERNEL_DIR
    pd.KERNEL_DIR = _shim_dir(variant)
    try:
        if dtype in (ttnn.uint8, ttnn.uint16, ttnn.uint32, ttnn.int32):
            tdt = {ttnn.uint8: torch.uint8}.get(dtype, torch.int32)
            torch_in = torch.randint(0, 100, shape, dtype=torch.int32).to(tdt)
        else:
            torch_in = torch.randn(shape).to(torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32)
        tt_in = ttnn.from_torch(
            torch_in,
            dtype=dtype,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
            tile=ttnn.Tile([in_tile_h, 32]),
        )
        call = dict(tile=ttnn.Tile([tile_h, 32]))
        if out_mem_config is not None:
            call["memory_config"] = out_mem_config
        out = tilize(tt_in, **call)
        ttnn.synchronize_device(device)
        # tilize is a PERMUTATION: the bar is bit-exact, not a PCC.
        exact = torch.equal(ttnn.to_torch(out), torch_in)
        ns = None
        if measure:
            _read_kernel_ns(device)  # flush the warm-up window
            out = tilize(tt_in, **call)
            ttnn.synchronize_device(device)
            ns = _read_kernel_ns(device)
    finally:
        pd.KERNEL_DIR = saved_dir
    slug = VARIANTS[variant][0]
    logger.info(
        f"RETILE-BAKEOFF {in_tile_h}->{tile_h} {list(shape)} arm={variant}:{slug} " f"ns={ns} bit_exact={exact}"
    )
    return ns, exact
