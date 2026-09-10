# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Isolated bake-off for idea `reader_bank_rotate`.

QUESTION: on the focus shape `[1,1,32,16384]` (R=1, C=512 -> 64 blocks / 64
cores, ONE tile-row per block), every core shares `start_page = row_start *
tile_h == 0` (R==1, one row group). The reader's stick loop
(`read_sticks_for_tilize`'s TILE mode) walks `row = 0 .. tile_h-1` in strict
ascending order, so all 64 cores issue their read of page 0 first, page 1
second, ... in LOCKSTEP: at read-step r the whole grid hammers DRAM bank
`r % NUM_BANKS` (interleaved round-robin, `page % num_banks`) while the other
11 banks idle. This file tests whether staggering the ISSUE ORDER of those
tile_h reads per core (same bytes, same transaction size, same count; only
which stick is read 1st/2nd/... changes) recovers bandwidth by spreading the
grid across all 12 banks every step instead of one bank at a time.

Mechanism note: every stick still lands at its baseline-identical L1 offset
(`l1_base + row * block_row_bytes`), addressed by ROW and not by issue slot, so
reordering behind the ONE shared `noc_async_read_barrier()` is free — the CB
push happens only after every read in the block has landed, regardless of
which one happened first. This is why the candidate is bit-identical to the
baseline on every shape (a correctness assert runs on every case, not just the
happy path).

Candidates (`TILIZE_ROTATE_VARIANT` env var, read by the monkeypatched
`_ablation_defines` below and turned into a compile `#define` for
`kernels/tilize_reader.cpp`, this dir's own copy — see that file's
`reader_read_block` branch for the implementation):
  0 (unset)        baseline: unmodified `read_sticks_for_tilize` helper call, ascending.
  1                raw loop, ascending (r0=0) -- the raw-vs-helper CONTROL.
  2 (+ STRIDE)     rotate: this block iteration's start row = (w_chunk * STRIDE) % tile_h.
                   STRIDE=1 is "rotate by the core's own linear index" (w_chunk IS the
                   core's raster position whenever num_blocks_this_core==1, which is every
                   shape below); swept over {1,5,7,11,13} per the task's coprime family.
  3                rotate so this iteration's FIRST read lands on bank
                   `w_chunk % NUM_DRAM_BANKS` exactly (solved, not guessed, from the
                   page->bank round-robin rule; NUM_DRAM_BANKS=12 is hardcoded in the
                   kernel as a box constant for this bake-off -- see that file's comment).

Run once per `TILIZE_ROTATE_VARIANT` (+ `TILIZE_ROTATE_STRIDE`) setting, matching
the existing `TILIZE_ABLATE` harness's own discipline (env var read at process
start, one `--profile` run per setting):

    for V in "" 1 "2:1" "2:5" "2:7" "2:11" "2:13" 3; do
        IFS=: read -r variant stride <<< "$V"
        TILIZE_ROTATE_VARIANT=$variant TILIZE_ROTATE_STRIDE=$stride \
            scripts/run_safe_pytest.sh --profile \
            ttnn/ttnn/operations/tilize/perf_experiments/reader_bank_rotate/test_reader_bank_rotate.py
    done

Every run is bit-identity gated (`torch.equal`) — tilize is a pure re-lay, so
any deviation here is a bug in the rotation, not a precision tradeoff (the op's
precision contract, `fp32_dest_acc_en`/`math_fidelity`/dtypes, is untouched;
this idea only reorders NoC issue order on the reader, nothing upstream of
DEST or the packer).
"""

import os
from pathlib import Path

import pytest

import ttnn

import ttnn.operations.tilize.tilize_program_descriptor as pd
from ttnn.operations.tilize import tilize

KERNEL_DIR = Path(__file__).parent / "kernels"

FOCUS_SHAPE = (1, 1, 32, 16384)  # R=1 C=512 -- attention: LOOSE_CASES[0], total lockstep

# Domain sweep named in the task: a wide-short geometry, a square one, a tall
# narrow one (disjoint page ranges per core -- no lockstep to begin with) and a
# small one.
DOMAIN_SHAPES = [
    ((1, 1, 32, 32768), "short_wide_wide"),  # R=1 C=1024 -- also total lockstep
    ((1, 1, 1024, 1024), "square_mid"),  # R=32 -- PARTIAL lockstep (num_row_groups < R)
    ((1, 1, 16384, 32), "tall_narrow"),  # R=512 C=1 -- disjoint per-core ranges, no lockstep
    ((1, 1, 32, 2048), "width_chunked"),  # R=1 C=64 -- small, total lockstep
]

_orig_ablation_defines = pd._ablation_defines


def _rotate_aware_defines():
    """Wraps the real op's own `_ablation_defines` (TILIZE_ABLATE support kept
    so this file can still cross-check against the reads-only ablation if
    needed) and layers this experiment's own env-driven defines on top."""
    defines = list(_orig_ablation_defines())
    variant = os.environ.get("TILIZE_ROTATE_VARIANT", "").strip()
    if variant:
        defines.append(("TILIZE_ROTATE_VARIANT", variant))
        stride = os.environ.get("TILIZE_ROTATE_STRIDE", "").strip()
        if stride:
            defines.append(("TILIZE_ROTATE_STRIDE", stride))
    return defines


def _build(device, shape, monkeypatch):
    # Point the op's own KERNEL_DIR at THIS experiment's kernels/ (reader is
    # modified only in its `reader_read_block` else-branch; writer/compute are
    # verbatim copies), and layer the rotate defines on top of the real
    # ablation ones. Neither the real op's files nor its module globals persist
    # any change past this test (monkeypatch auto-restores).
    # `import torch` is function-local, not module-level: `scripts/validate_no_global_torch_imports.py`
    # forbids a global torch import anywhere under `ttnn/ttnn/`, so that `import ttnn` never drags
    # torch in. Same convention the perf examples under `operations/examples/` follow.
    import torch

    monkeypatch.setattr(pd, "KERNEL_DIR", KERNEL_DIR)
    monkeypatch.setattr(pd, "_ablation_defines", _rotate_aware_defines)
    monkeypatch.setattr(pd, "_PLAN_CACHE", {})

    grid = device.compute_with_storage_grid_size()
    torch.manual_seed(11)
    torch_input = torch.randn(shape, dtype=torch.float32).bfloat16()
    tt_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_output = tilize(tt_input)
    plan = pd.derive_plan(tt_input, tt_output, low_l1=False, grid=grid)
    variant = os.environ.get("TILIZE_ROTATE_VARIANT", "<baseline>")
    stride = os.environ.get("TILIZE_ROTATE_STRIDE", "-")
    cores = len(plan.assignment)
    print(
        f"\n[reader_bank_rotate variant={variant} stride={stride}] {shape}: "
        f"R={plan.tensor_row_blocks} C={plan.tensor_col_tiles} bw={plan.block_width_tiles} "
        f"row_groups={plan.num_row_groups} rows/blk={plan.tensor_row_blocks // plan.num_row_groups} "
        f"blocks={plan.num_blocks_total} cores={cores}/{grid.x * grid.y}"
    )
    got = ttnn.to_torch(tt_output)
    assert torch.equal(got, torch_input), (
        f"reader_bank_rotate variant={variant} stride={stride} on {shape}: not bit-identical "
        "-- rotation broke a stick's destination offset"
    )
    return plan


ALL_SHAPES = [(FOCUS_SHAPE, "attention")] + DOMAIN_SHAPES


@pytest.mark.parametrize("shape,label", ALL_SHAPES, ids=[s[1] for s in ALL_SHAPES])
def test_reader_bank_rotate(device, shape, label, monkeypatch):
    _build(device, shape, monkeypatch)
