# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Isolated bake-off for idea `wave_ladder_v2`.

QUESTION. `PIPELINE_WAVES_PER_CORE` (cap 4) and `MIN_BLOCK_ROW_BYTES` (512) were
calibrated in Refinement 3, when the READ stage was 33-46% of the wall. It is
now 26% (Perf 1's DRAM bank-spread rotation) and the WRITER is starved 61% of
its span. A wave is bought by HALVING `block_width_tiles`, i.e. by halving the
read transaction; the focus shape `[1,1,32,16384]` sits at exactly 512 B / ONE
wave, which is the configuration `MIN_BLOCK_ROW_BYTES` refuses to trade away
from. Does 256 B x 2 waves (or 128 B x 4, 64 B x 8) now pay, when it did not?

WHAT THIS FILE MEASURES. Whole-op device kernel ns at waves in {1,2,4,8} with
`MIN_BLOCK_ROW_BYTES` disabled (so the rung requested is the rung taken), on the
focus shape and on the six domain-sweep geometries, 3 reps per rung.

TWO INSTRUMENT SUBTLETIES, both deliberate:

  1. ZONE-FREE KERNELS BY DEFAULT. The op's kernels carry permanent
     `MaybeDeviceZoneScope` markers that only exist under `--profile` -- which is
     the only mode that also reports `DEVICE KERNEL DURATION`. Their cost scales
     with BLOCKS PER CORE, which is exactly the quantity this ladder varies (w8
     runs 8x the zone executions of w1 on an R==1 shape), so measuring the ladder
     with zones on biases it against the deep rungs by an amount of the same
     order as the ~5% effect being chased. `kernels_nozone/` is a byte-copy of
     the op's kernels with the zone macro `#define`d to `((void)0)`; `KERNEL_DIR`
     is monkeypatched onto it. Set `TILIZE_ZONES=1` to use the op's real kernels
     instead (needed for the `zone_report.py` mechanism check, and to reconcile
     against the coordinator's with-zones baseline).

  2. THE SPLIT READER RIDES ALONG. `SPLIT_READER_MAX_ROW_BYTES = 256` means the
     w2 rung on the focus shape (256 B) does not only buy a wave, it ALSO flips
     the split reader on. Those are two separate mechanisms and a menu that
     conflates them is useless, so `TILIZE_SPLIT=off` forces the split reader off
     at every rung, giving the wave axis on its own.

Correctness: `torch.equal` on every rep of every rung. tilize does no arithmetic,
so anything short of bit identity is a bug, not a precision trade.

MEASURED (Wormhole B0 n150, 8x8 = 64/64 cores at EVERY rung of EVERY shape,
1 GHz, 12 DRAM banks; `DEVICE KERNEL DURATION [ns]`, median; zone-free kernels).

FOCUS `[1,1,32,16384]` -- 12 samples per rung, pooled over FOUR independent
sessions (split=auto, split=off, WRITE_BATCH_MIN_TILES=8, and zones-on):
    w1  bw=8  512 B   12515 ns   <-- what the shipped rule already picks
    w2  bw=4  256 B   13048 ns   (1.043x SLOWER)
    w4  bw=2  128 B   15539 ns   (1.241x SLOWER)
    w8  bw=1   64 B   27040 ns   (2.160x SLOWER)
w2 was slower than w1 in all four sessions independently, so the 4% is a real
sign even though it sits in the +-3% band per session.

DOMAIN, 7 reps per rung, median ns (w1 / w2 / w4 / w8), 64/64 cores throughout:
    [1,1,32,32768]   23657 / 23445 / 24836 / 29686   (1024/512/256/128 B)
    [1,1,1024,1024]  23679 / 23524 / 25828 / 28109   (1024/512/256/128 B)
    [1,1,2048,2048]  93486 / 88787 / 85644 / 84998   (4096/2048/1024/512 B)
    [1,1,16384,32]   17143 / 17564 / 17420 / 17123   (C==1: knob INERT, flat)
    [1,1,2048,64]     4947 /  6227 /  6429 /  6556   (128/64/64/64 B)
    [1,1,32,2048]     3520 /  3549 /  3480 /  3550   (w1 already 64 B: INERT)

WHY THE LADDER DOES NOT PAY -- zone shift on the focus shape, w1 vs w2
(`TILIZE_ZONES=1`, `zone_report.py`, per-core mean over 64 cores):
                       w1 (512 B)      w2 (256 B)
    NCRISC span            5453            7439   (+36%)
      reader_read_block    5324            7259   (+1935)
    BRISC span             9489           10723   (+13%)
      writer_wait_out      6117            4356   (-1761)   <-- overlap DID work
      writer_issue         1870            2497   (+627)
      writer_barrier       1221            3401   (+2180)
The starvation relief the idea predicted is REAL and measurable (-1761 ns of
`writer_wait_out`, 64.5% -> 40.6% of the writer's span). It is simply smaller
than what it costs: halving the read transaction doubles the NoC command count
for the same bytes and adds +1935 ns to the reader, and the narrower block also
lengthens `writer_barrier`. Holding the writer's in-flight transaction count at
8 across the ladder (`TILIZE_WB=8`, so the wave is not silently charged for a
writer regression) does not change the answer: w2 = 12884 ns there vs w1 =
12449 ns in the same session.

CONCLUSION: `MIN_BLOCK_ROW_BYTES` stays 512 and `PIPELINE_WAVES_PER_CORE` stays
4. 512 is pinned from BOTH sides by this sweep: it must be <= 512 or
`[1,1,1024,1024]` and `[1,1,32,32768]` lose their 512 B rung, and > 256 or
`[1,1,32,16384]` takes a 4%-slower 256 B rung and `[1,1,2048,64]` a 26%-slower
64 B one. (Read sizes are quantized to 64 B x 2^k on bf16, so every value in
(256, 512] decides identically; 512 is the natural representative.) Raising the
cap to 8 would only move `[1,1,2048,2048]` (w4 85644 vs w8 84998, sign flips
between sessions = flat), so it buys nothing.

RUN (foreground only):
    scripts/run_safe_pytest.sh --run-all --profile \
        ttnn/ttnn/operations/tilize/perf_experiments/wave_ladder_v2/test_wave_ladder_v2.py -k focus
    python3 ttnn/ttnn/operations/tilize/perf_experiments/wave_ladder_v2/collect.py
"""

import os
from pathlib import Path

import pytest

import ttnn

import ttnn.operations.tilize.tilize_program_descriptor as pd
from ttnn.operations.tilize import tilize

NOZONE_KERNELS = Path(__file__).parent / "kernels_nozone"

FOCUS_SHAPE = (1, 1, 32, 16384)  # R=1 C=512 -- attention: LOOSE_CASES[0]

DOMAIN_SHAPES = [
    ((1, 1, 32, 32768), "short_wide_wide"),
    ((1, 1, 1024, 1024), "square_mid"),
    ((1, 1, 2048, 2048), "square_large"),
    ((1, 1, 16384, 32), "tall_narrow"),
    ((1, 1, 2048, 64), "full_width_small"),
    ((1, 1, 32, 2048), "short_wide_small"),
]

WAVES = [1, 2, 4, 8]
# Reps per rung. 3 by default; raised for the two geometries that pin
# MIN_BLOCK_ROW_BYTES from opposite sides (square_mid wants <=512, short_wide_wide
# marginally wants >512), where the gap is inside the run-to-run band.
REPS = int(os.environ.get("TILIZE_REPS", 3))


def _prep(device, shape, waves, monkeypatch):
    # `import torch` is function-local, not module-level: `scripts/validate_no_global_torch_imports.py`
    # forbids a global torch import anywhere under `ttnn/ttnn/`.
    import torch

    # The rung requested is the rung taken: floor 0 makes the plan's
    # "deepest pipe clearing MIN_BLOCK_ROW_BYTES" loop accept `waves` outright.
    monkeypatch.setattr(pd, "PIPELINE_WAVES_PER_CORE", waves)
    monkeypatch.setattr(pd, "MIN_BLOCK_ROW_BYTES", 0)
    if not os.environ.get("TILIZE_ZONES"):
        monkeypatch.setattr(pd, "KERNEL_DIR", NOZONE_KERNELS)
    split = os.environ.get("TILIZE_SPLIT", "auto")
    if split == "off":
        monkeypatch.setattr(pd, "SPLIT_READER_MAX_ROW_BYTES", 0)
    elif split == "on":
        monkeypatch.setattr(pd, "SPLIT_READER_MAX_ROW_BYTES", 1 << 30)
    # THE WRITER TWIN. `write_rows_per_barrier = ceil(WRITE_BATCH_MIN_TILES/bw)`,
    # so the writer's in-flight transaction count is
    # `wrpb * bw = max(bw, WRITE_BATCH_MIN_TILES)` -- i.e. buying a wave by
    # halving bw from 8 to 4 ALSO halves the writes in flight from 8 to 4 at the
    # shipped WRITE_BATCH_MIN_TILES=4. Charging the wave ladder for a writer
    # regression it did not ask for would be a strawman, so `TILIZE_WB=8` holds
    # the in-flight count at 8 across every rung (w1's bw=8 is unaffected).
    wb = os.environ.get("TILIZE_WB")
    if wb:
        monkeypatch.setattr(pd, "WRITE_BATCH_MIN_TILES", int(wb))
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
    return torch, torch_input, tt_input, grid, split


def _ladder(device, shape, label, waves, monkeypatch):
    """Run one rung REPS times and print one marker line the collector aligns on."""
    torch, torch_input, tt_input, grid, split = _prep(device, shape, waves, monkeypatch)

    plan = None
    for rep in range(REPS):
        tt_output = tilize(tt_input)
        if plan is None:
            plan = pd.derive_plan(tt_input, tt_output, low_l1=False, grid=grid)
        assert torch.equal(ttnn.to_torch(tt_output), torch_input), f"{label} w{waves} rep{rep}: not bit-identical"

    cores = plan.num_cores_used
    print(
        f"\nWLV2_MARK label={label} shape={tuple(shape)} waves={waves} split={split} reps={REPS} "
        f"bw={plan.block_width_tiles} chunks={plan.num_w_chunks} rows={plan.tensor_row_blocks} "
        f"blocks={plan.num_blocks_total} tail={plan.tail_group is not None} cores={cores}/{grid.x * grid.y} "
        f"read={plan.block_row_bytes}B split_reader={plan.split_reader} "
        f"wrpb={plan.write_rows_per_barrier} inflight={plan.write_rows_per_barrier * plan.block_width_tiles} "
        f"waves_per_core={plan.tensor_row_blocks * plan.num_w_chunks / max(cores, 1):.2f} "
        f"L1={plan.l1_per_core_bytes // 1024}KiB"
    )


@pytest.mark.parametrize("waves", WAVES, ids=[f"w{w}" for w in WAVES])
def test_focus_ladder(device, waves, monkeypatch):
    _ladder(device, FOCUS_SHAPE, "focus", waves, monkeypatch)


@pytest.mark.parametrize("shape,label", DOMAIN_SHAPES, ids=[d[1] for d in DOMAIN_SHAPES])
@pytest.mark.parametrize("waves", WAVES, ids=[f"w{w}" for w in WAVES])
def test_domain_ladder(device, shape, label, waves, monkeypatch):
    _ladder(device, shape, label, waves, monkeypatch)
