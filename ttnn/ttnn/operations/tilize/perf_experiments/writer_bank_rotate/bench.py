# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Isolated bake-off for idea `writer_bank_rotate` — PERF EXPERIMENT, not the
real op. See the kernel-head comment in
`kernels/tilize_writer_bankrotate.cpp` for the mechanism.

This module builds the REAL op's program descriptor via
`tilize_program_descriptor.create_program_descriptor` (unmodified — reader and
compute kernels are the op's own, byte-for-byte) and then swaps ONLY the
writer `KernelDescriptor.kernel_source` (a `def_rw` field, mutable post-
construction) to point at this dir's `tilize_writer_bankrotate.cpp`, with a
`WRITER_ROTATE_MODE` define selecting the rotation. Everything else — CBs,
reader, compute, runtime args, compute-kernel config — is exactly what the
real `tilize()` call would build. Never touches
`ttnn/ttnn/operations/tilize/tilize_program_descriptor.py` or its kernels.
"""

import importlib
import os
from pathlib import Path

import ttnn

import ttnn.operations.tilize.tilize_program_descriptor as pd

# NOT `import ttnn.operations.tilize.tilize as tilize_mod`: the package's own
# `ttnn/ttnn/operations/tilize/__init__.py` does `from .tilize import tilize`,
# which rebinds the package attribute `tilize` (that a submodule import would
# normally chain through) to the FUNCTION of the same name. The dotted
# `import ... as` form resolves via that (now-shadowed) attribute, so it would
# hand back the function, not the module — `importlib.import_module` goes
# straight through `sys.modules` and sidesteps the shadow.
tilize_mod = importlib.import_module("ttnn.operations.tilize.tilize")

KERNEL_DIR = Path(__file__).parent / "kernels"
WRITER_KERNEL = KERNEL_DIR / "tilize_writer_bankrotate.cpp"

# Mode name -> WRITER_ROTATE_MODE value (see the kernel-head comment).
MODES = {
    "baseline": 0,
    "rotate_chunk": 1,
    "rotate_blockid": 2,
    "rotate_bankexact": 3,
    "rotate_rows_cols": 4,
}


def build_program_descriptor(input_tensor, output_tensor, mode_name, *, low_l1=False, compute_kernel_config=None):
    """The real op's descriptor, writer kernel swapped for our variant."""
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
            kd.defines = list(kd.defines) + [("WRITER_ROTATE_MODE", str(MODES[mode_name]))]
            swapped = True
    assert swapped, "no writer kernel found in the real op's program descriptor for this plan (native output?)"
    return program_descriptor


def run_variant_from_tt(tt_input, mode_name, *, dtype=None, low_l1=False):
    """Runs the op (writer swapped for `mode_name`) on an already-on-device
    `tt_input` (ROW_MAJOR, as every test in this bench builds it). Mirrors
    `tilize()`'s own body (tilize.py) exactly for the no-pad, no-custom-tile,
    no-shard case this bench covers, so the ONLY difference from calling the
    real op is the writer kernel source + its one define."""
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
    program_descriptor = build_program_descriptor(tt_input, output_tensor, mode_name, low_l1=low_l1)
    ttnn.generic_op([tt_input, output_tensor], program_descriptor)
    plan = pd.derive_plan(tt_input, output_tensor, low_l1=low_l1, grid=grid_size)
    return output_tensor, plan


def current_mode_from_env(default="baseline"):
    """`WRITER_ROTATE_MODE_NAME` env var, mirroring the op's own
    `TILIZE_ABLATE` env-var-to-defines convention — set by the shell loop that
    runs one `scripts/run_safe_pytest.sh --profile` per variant."""
    return os.environ.get("WRITER_ROTATE_MODE_NAME", default)
