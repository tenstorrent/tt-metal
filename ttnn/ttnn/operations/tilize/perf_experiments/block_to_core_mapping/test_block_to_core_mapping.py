# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Isolated bake-off harness — idea `block_to_core_mapping`.

PERF EXPERIMENT ONLY. Does not touch the real op; `bench.py` wraps
`derive_plan` at runtime and permutes only the CORE column of the plan's block
assignment (see its module docstring).

    TILIZE_PERM_MODE=baseline TILIZE_PERM_CASES=focus \
      scripts/run_safe_pytest.sh --profile \
      ttnn/ttnn/operations/tilize/perf_experiments/block_to_core_mapping/test_block_to_core_mapping.py

`TILIZE_PERM_MODE` in {baseline, colmajor, snake, reverse, stride5, stride9,
bitrev, diag}; `TILIZE_PERM_CASES` in {focus, sweep}. Correctness (bit identity,
tilize is a pure re-lay) is gated on EVERY call in every mode, and the number of
cores the plan uses is asserted equal to the baseline plan's, so a mode can
never "win" by quietly shrinking the split.
"""

import json
import os
from pathlib import Path

import pytest

import ttnn

from ttnn.operations.tilize.perf_experiments.block_to_core_mapping import bench

REPS = int(os.environ.get("TILIZE_PERM_REPS", "3"))

FOCUS = ((1, 1, 32, 16384), "attention_focus")
SWEEP = [
    FOCUS,
    ((1, 1, 32, 32768), "short_wide_wide"),
    ((1, 1, 1024, 1024), "square_mid"),
    ((1, 1, 2048, 2048), "square_large"),
    ((1, 1, 16384, 32), "tall_narrow"),
    ((1, 1, 2048, 64), "tall_narrow_small"),
]


def _cases():
    which = os.environ.get("TILIZE_PERM_CASES", "focus").strip()
    if which == "focus":
        return [FOCUS]
    if which == "sweep":
        return SWEEP
    raise AssertionError(f"unknown TILIZE_PERM_CASES {which!r}")


def test_block_to_core_mapping(device):
    import torch

    mode = bench.current_mode_from_env()
    bench.install()
    records = []
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
        for rep in range(REPS + 1):  # rep 0 is the compile/allocate pass
            out = ttnn.tilize(tt_input)
            got = ttnn.to_torch(out)
            assert torch.equal(got, torch_input), f"{mode} {label} rep{rep}: NOT bit-identical"
            ttnn.deallocate(out)
        plan = bench.LAST_PLANS[-1]
        records.append(
            {
                "mode": mode,
                "label": label,
                "shape": list(shape),
                "num_cores_used": plan.num_cores_used,
                "block_width_tiles": plan.block_width_tiles,
                "num_blocks_total": plan.num_blocks_total
                + (0 if plan.tail_group is None else plan.tail_group.num_blocks),
                "assignment": bench.assignment_map(plan),
            }
        )
        ttnn.deallocate(tt_input)
        print(
            f"[{mode}] {label} {list(shape)}: cores={records[-1]['num_cores_used']} "
            f"bw={plan.block_width_tiles} blocks={records[-1]['num_blocks_total']}"
        )

    out_dir = Path(__file__).parent / "logs"
    out_dir.mkdir(exist_ok=True)
    (out_dir / f"plan_{mode}_{os.environ.get('TILIZE_PERM_CASES','focus')}.json").write_text(
        json.dumps(records, indent=1)
    )
