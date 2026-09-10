# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""writer_starvation — isolated bake-off harness.

Correctness (bit identity — tilize does no arithmetic):
    scripts/run_safe_pytest.sh --run-all \
        ttnn/ttnn/operations/tilize/perf_experiments/writer_starvation/test_writer_starvation.py \
        -k correctness

Wall-clock (device kernel ns), 3 reps per variant, zones OFF (marker-safe):
    scripts/run_safe_pytest.sh --profile \
        ttnn/ttnn/operations/tilize/perf_experiments/writer_starvation/test_writer_starvation.py \
        -k perf_focus

Zone capture (focus shape only — the fine path puts a zone inside a per-group
loop, which is only marker-safe at this geometry):
    scripts/run_safe_pytest.sh --profile ... -k perf_zones

Every dispatch prints a `DISPATCH n | <label>` line in execution order; row n of
`generated/profiler/reports/*/ops_perf_results*.csv` is that dispatch's
`DEVICE KERNEL DURATION [ns]`. `parse.py` in this directory does the join.
"""

import os

import pytest
import ttnn

from ttnn.operations.tilize.perf_experiments.writer_starvation import bench

FOCUS_SHAPE = (1, 1, 32, 16384)

DOMAIN_SHAPES = [
    ((1, 1, 32, 16384), "focus"),
    ((1, 1, 32, 32768), "wide2x"),
    ((1, 1, 1024, 1024), "square1k"),
    ((1, 1, 2048, 2048), "square2k"),
    ((1, 1, 16384, 32), "tall_bw1"),
    ((1, 1, 2048, 64), "tall_bw2"),
]

# Reps per variant. Device kernel time has no warm-up transient, so reps exist
# only to median out run-to-run DRAM/dispatch noise (measured band on this box:
# ~+-5% on a 12 us kernel). Raise for a call that sits inside that band.
REPS = int(os.environ.get("WS_REPS", "3"))

_DISPATCH = [0]


def _make_tensors(device, shape):
    # Function-local torch import: `scripts/validate_no_global_torch_imports.py`
    # forbids a global torch import anywhere under `ttnn/ttnn/`.
    import torch

    torch.manual_seed(11)
    torch_input = torch.randn(shape, dtype=torch.float32).bfloat16()
    tt_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_output = ttnn.allocate_tensor_on_device(
        ttnn.Shape(shape),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        device,
        ttnn.DRAM_MEMORY_CONFIG,
    )
    return torch_input, tt_input, tt_output


def _run(device, shape, variant, label, check=True, zones=False):
    import torch

    torch_input, tt_input, tt_output = _make_tensors(device, shape)
    try:
        plan = bench.derive_geometry(tt_input, tt_output, device)
        descriptor = bench.build_descriptor(tt_input, tt_output, plan, variant, zones=zones)
    except bench.Inexpressible as exc:
        print(f"\nSKIP {label} {variant} {tuple(shape)}: INEXPRESSIBLE — {exc}")
        return None
    ttnn.generic_op([tt_input, tt_output], descriptor)
    _DISPATCH[0] += 1
    spec = bench.variant_spec(plan, variant)
    print(
        f"\nDISPATCH {_DISPATCH[0]} | {label} | {variant} | {tuple(shape)} | "
        f"bw={plan.block_width_tiles} chunks={plan.num_w_chunks} R={plan.tensor_row_blocks} "
        f"wrpb={plan.write_rows_per_barrier} cores={len(plan.assignment)} | "
        f"read_barriers={spec[0]} push={spec[2]} wait={spec[3]} row_rot={spec[4]}"
    )
    if check:
        got = ttnn.to_torch(tt_output)
        ok = torch.equal(got, torch_input)
        print(f"    bit_identical={ok}")
        assert ok, f"{label}/{variant}: tilize is not bit-identical"
    return plan


# ---------------------------------------------------------------------------
# correctness — the only pass/fail
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape,label", DOMAIN_SHAPES, ids=[s[1] for s in DOMAIN_SHAPES])
def test_correctness(device, shape, label):
    for variant in bench.VARIANTS:
        _run(device, shape, variant, label, check=True)


# ---------------------------------------------------------------------------
# perf — measured, never asserted
# ---------------------------------------------------------------------------


def test_perf_focus(device):
    """3 reps x every variant on the focus shape, rep-major so any drift is
    spread evenly over the variants rather than pinned to one."""
    for rep in range(REPS):
        for variant in bench.VARIANTS:
            _run(device, FOCUS_SHAPE, variant, f"focus_rep{rep}", check=False)


def test_perf_domain(device):
    """The domain sweep, 3 reps, rep-major within each shape."""
    for shape, label in DOMAIN_SHAPES:
        for rep in range(REPS):
            for variant in bench.VARIANTS:
                _run(device, shape, variant, f"{label}_rep{rep}", check=False)


def test_perf_zones(device):
    """Focus shape only, zones ON: the writer_wait_out / BRISC-span shift.
    Read with `perf_experiments/zone_report.py --run <n>`."""
    for variant in ("baseline", "fine_half", "fine_min", "split_read", "combo"):
        _run(device, FOCUS_SHAPE, variant, "zones", check=False, zones=True)
