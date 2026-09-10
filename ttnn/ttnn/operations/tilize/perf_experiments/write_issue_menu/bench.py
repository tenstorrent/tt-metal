# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Isolated bake-off for idea `write_issue_menu` — PERF EXPERIMENT, not the
real op. Mechanism menu is documented at the head of
`kernels/tilize_writer_issue_menu.cpp`.

Builds the REAL op's program descriptor via
`tilize_program_descriptor.create_program_descriptor` (unmodified — reader and
compute kernels are the op's own, byte-for-byte, as are every CB, runtime arg
and compute-kernel config) and swaps ONLY the writer `KernelDescriptor`'s
`kernel_source` (a `def_rw` field, mutable post-construction) for this dir's
copy, plus one `WRITE_ISSUE_MODE` define. Never touches the real op's files.

The ablation switch is read from `os.environ["TILIZE_ABLATE"]` at
descriptor-BUILD time (`tilize_program_descriptor._ablation_defines`), so this
module can flip it per call and get both the isolated write stage and the whole
op out of ONE pytest invocation — different defines hash to different compiled
kernels, so there is no cross-contamination.
"""

import contextlib
import importlib
import os
from pathlib import Path

import ttnn

import ttnn.operations.tilize.tilize_program_descriptor as pd

# NOT `import ...tilize.tilize as tilize_mod`: the package `__init__` rebinds
# the attribute `tilize` to the FUNCTION, so the dotted form hands back the
# function. `importlib` goes straight through `sys.modules`.
tilize_mod = importlib.import_module("ttnn.operations.tilize.tilize")

KERNEL_DIR = Path(__file__).parent / "kernels"
WRITER_KERNEL = KERNEL_DIR / "tilize_writer_issue_menu.cpp"

# Mode name -> WRITE_ISSUE_MODE value. Must match the #defines in the kernel.
MODES = {
    "baseline": 0,  # today's op verbatim
    "rot_none": 1,  # diagnostic control: no issue-order rotation at all
    "rot_bankunif": 2,  # (a) maximally uniform starting-bank coverage
    "flush_half": 3,  # (b) mid-batch noc_async_writes_flushed
    "barrier_half": 4,  # (b) two full barriers per batch
    "cmdbuf2": 5,  # (c) round-robin over NoC1 cmd bufs {0,2}
    "cmdbuf4": 6,  # (c) round-robin over NoC1 cmd bufs {0,2,3,1}
    "posted": 7,  # (e) posted writes + posted-writes-flushed  [HAZARD: see kernel]
    "flush_only": 8,  # (b/e) per-batch flush (SENT) + ONE real barrier at kernel end — hazard-free
}


@contextlib.contextmanager
def ablate(stages):
    """`TILIZE_ABLATE` for the duration of a descriptor build. `stages` is a
    comma string like "reads,compute", or "" for the full op."""
    prev = os.environ.get("TILIZE_ABLATE")
    if stages:
        os.environ["TILIZE_ABLATE"] = stages
    else:
        os.environ.pop("TILIZE_ABLATE", None)
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop("TILIZE_ABLATE", None)
        else:
            os.environ["TILIZE_ABLATE"] = prev


def build_program_descriptor(input_tensor, output_tensor, mode_name, *, low_l1=False, compute_kernel_config=None):
    """The real op's descriptor, writer kernel source swapped for our variant."""
    assert mode_name in MODES, f"unknown mode {mode_name!r}"
    program_descriptor = pd.create_program_descriptor(
        input_tensor,
        output_tensor,
        low_l1=low_l1,
        pad_value=None,
        compute_kernel_config=compute_kernel_config,
    )
    swapped = False
    for kd in program_descriptor.kernels:
        if kd.kernel_source.endswith("tilize_writer.cpp"):
            kd.kernel_source = str(WRITER_KERNEL)
            kd.defines = list(kd.defines) + [("WRITE_ISSUE_MODE", str(MODES[mode_name]))]
            swapped = True
    assert swapped, "no writer kernel in this plan's descriptor (native sharded output?)"
    return program_descriptor


def run_variant_from_tt(tt_input, mode_name, *, ablate_stages="", dtype=None, low_l1=False):
    """Mirrors `tilize()`'s own body for the no-pad / no-custom-tile / no-shard
    case this bench covers, so the ONLY difference from calling the real op is
    the writer kernel source + its one define (+ the ablation defines)."""
    device = tt_input.device()
    out_dtype = dtype if dtype is not None else tt_input.dtype
    out_tile = ttnn.Tile([32, tilize_mod.TILE_WIDTH])
    grid_size = device.compute_with_storage_grid_size()
    core_grid = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_size.x - 1, grid_size.y - 1))}
    )
    output_tensor = ttnn.allocate_tensor_on_device(
        tilize_mod._output_tensor_spec(tt_input.shape, out_dtype, tt_input.memory_config(), out_tile, None, core_grid),
        device,
    )
    with ablate(ablate_stages):
        program_descriptor = build_program_descriptor(tt_input, output_tensor, mode_name, low_l1=low_l1)
    ttnn.generic_op([tt_input, output_tensor], program_descriptor)
    plan = pd.derive_plan(tt_input, output_tensor, low_l1=low_l1, grid=grid_size)
    return output_tensor, plan


def current_mode_from_env(default="baseline"):
    """`WRITE_ISSUE_MODE_NAME` env var — set by the shell loop that runs one
    `scripts/run_safe_pytest.sh --profile` per variant."""
    return os.environ.get("WRITE_ISSUE_MODE_NAME", default)
