# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""pytest plugin: run the op's golden suite on a per_tile_pass_chains kernel variant.

Swaps the program descriptor's KERNEL_DIR to k_<PTC_GOLDEN_VARIANT> (default n1u) and adds the variant's extra CB
(CB 25, the constant -1 bf16 tile) to every program and to the L1 footprint the planner sees.

  PYTHONPATH=<this dir>:$PYTHONPATH scripts/run_safe_pytest.sh --run-all \
      eval/golden_tests/groupnorm_sc_N_1_HW_C_b/ -p ptc_golden_plugin
"""
import os
from pathlib import Path

import ttnn
import ttnn.operations.groupnorm_sc_N_1_HW_C_b.groupnorm_sc_N_1_HW_C_b_program_descriptor as pd

HERE = Path(__file__).parent
VARIANT = os.environ.get("PTC_GOLDEN_VARIANT", "n1u")
pd.KERNEL_DIR = HERE / f"k_{VARIANT}"
EXTRA = [(25, ttnn.bfloat16, 2048)] if VARIANT.startswith("n1") else []
_orig_fp, _orig_cpd = pd.footprint_bytes, pd.create_program_descriptor


def _fp(*a, **k):
    return _orig_fp(*a, **k) + sum(page for _, _, page in EXTRA)


def _cpd(plan, *a, **k):
    desc = _orig_cpd(plan, *a, **k)
    if EXTRA:
        cores = plan.shard_grid if plan.sharded else pd._used_core_range_set(len(plan.work), plan.grid_x)
        cbs = list(desc.cbs)
        for cb_id, dt, page in EXTRA:
            cbs.append(pd._cb(page, cores, [(cb_id, dt, page)]))
        desc.cbs = cbs
    return desc


pd.footprint_bytes, pd.create_program_descriptor = _fp, _cpd
print(f"PTC_GOLDEN_PLUGIN: kernel dir {pd.KERNEL_DIR}, extra CBs {EXTRA}")
