# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Isolated bake-off harness — idea `positional_work_skew`. PERF EXPERIMENT ONLY.

Does not touch the real op; `bench.py` wraps `derive_plan` at runtime and gives
the grid's rows DIFFERENT `block_width_tiles` using the machinery the op already
has for the ragged column tail (two-plus core ranges, each with its own width /
`col_tile_offset` / CB sizes).

    TILIZE_SKEW_MODES=baseline,half97,half97_rev TILIZE_SKEW_CASES=focus \
      scripts/run_safe_pytest.sh --profile \
      ttnn/ttnn/operations/tilize/perf_experiments/positional_work_skew/test_positional_work_skew.py

Every mode is measured IN THE SAME PROCESS, INTERLEAVED rep by rep. That is
sound here (and was not for the sibling's pure core permutation) because
`compute_program_descriptor_hash` hashes `kernel.core_ranges` and
`kernel.compile_time_args`, both of which a width skew changes — so the program
cache cannot alias two modes. Interleaving puts every mode under the same drift
and drops the comparison into the ~5% intra-process band.

GATES, on EVERY call of EVERY mode:
  * `torch.equal` — tilize is a pure byte re-lay, so bit identity is the oracle;
  * the widths sum to `C` exactly (in `build_skew_plan`) — a variant that moved
    fewer bytes would not be a comparison;
  * the core count equals the baseline plan's — a variant that shrank the split
    would be reporting the split, not the kernel.
"""

import json
import os
from pathlib import Path

import pytest

import ttnn

from ttnn.operations.tilize.perf_experiments.positional_work_skew import bench

REPS = int(os.environ.get("TILIZE_SKEW_REPS", "3"))

FOCUS = ((1, 1, 32, 16384), "attention_focus")
CASES = {
    "focus": [FOCUS],
    "wide2": [((1, 1, 32, 32768), "short_wide_wide")],
    "sweep": [
        FOCUS,
        ((1, 1, 32, 32768), "short_wide_wide"),
        ((1, 1, 1024, 1024), "square_mid"),
        ((1, 1, 2048, 2048), "square_large"),
        ((1, 1, 16384, 32), "tall_narrow"),
        ((1, 1, 2048, 64), "tall_narrow_small"),
    ],
}


def _cases():
    which = os.environ.get("TILIZE_SKEW_CASES", "focus").strip()
    assert which in CASES, f"unknown TILIZE_SKEW_CASES {which!r}; known: {sorted(CASES)}"
    return CASES[which]


def test_positional_work_skew(device):
    import torch

    modes = bench.parse_modes()
    bench.install()
    dispatches = []  # one entry per generic_op dispatch, in dispatch order
    plans = {}

    for shape, label in _cases():
        torch.manual_seed(11)
        torch_input = torch.randn(shape, dtype=torch.float32).bfloat16()
        tt_input = ttnn.from_torch(
            torch_input,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        baseline_cores = None
        # --- phase 1: one compile/allocate pass per mode, and the plan record --
        for mode in modes:
            bench.set_mode(mode)
            out = ttnn.tilize(tt_input)
            got = ttnn.to_torch(out)
            assert torch.equal(got, torch_input), f"{mode} {label} warm: NOT bit-identical"
            ttnn.deallocate(out)
            plan = bench.LAST_PLANS[-1]
            rec = {
                "mode": mode,
                "label": label,
                "shape": list(shape),
                "cores_used": bench.cores_used(plan),
                "num_groups": len(plan.groups),
                "reason_not_applied": bench.LAST_REASON.get(mode),
                "groups": bench.plan_summary(plan),
            }
            plans[(label, mode)] = rec
            if mode == "baseline" or baseline_cores is None:
                baseline_cores = rec["cores_used"]
            dispatches.append({"phase": "warm", "mode": mode, "label": label})
            print(
                f"[{label}] {mode}: cores={rec['cores_used']} groups={rec['num_groups']} "
                f"widths={[(g['width'], g['cores'], g['row_bytes']) for g in rec['groups']]}"
                + (f"  NOT APPLIED: {rec['reason_not_applied']}" if rec["reason_not_applied"] else "")
            )

        for mode in modes:
            rec = plans[(label, mode)]
            assert rec["cores_used"] == baseline_cores, (
                f"{mode} {label}: uses {rec['cores_used']} cores vs baseline's {baseline_cores} — "
                "a variant that changes the split is reporting the split, not the kernel"
            )

        # --- phase 2: interleaved measured reps ---------------------------
        # The mode order is ROTATED by one every rep. Measured reason, not
        # cosmetics: with a FIXED order the sequence position of a dispatch is
        # worth ~3% on its own (`uniform8`, which is byte-identical to
        # `baseline`, read 0.970x of it purely by sitting one slot later), which
        # is the same size as the effect under test. Rotating gives every mode
        # every position, so the position term averages out instead of aliasing
        # onto whichever modes happened to be listed at the fast slots.
        for rep in range(REPS):
            k = rep % len(modes)
            for mode in modes[k:] + modes[:k]:
                bench.set_mode(mode)
                out = ttnn.tilize(tt_input)
                got = ttnn.to_torch(out)
                assert torch.equal(got, torch_input), f"{mode} {label} rep{rep}: NOT bit-identical"
                ttnn.deallocate(out)
                dispatches.append({"phase": "measure", "mode": mode, "label": label, "rep": rep})

        ttnn.deallocate(tt_input)

    out_dir = Path(__file__).parent / "logs"
    out_dir.mkdir(exist_ok=True)
    tag = os.environ.get("TILIZE_SKEW_TAG", os.environ.get("TILIZE_SKEW_CASES", "focus"))
    (out_dir / f"dispatches_{tag}.json").write_text(
        json.dumps({"dispatches": dispatches, "plans": list(plans.values())}, indent=1)
    )
