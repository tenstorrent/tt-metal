# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""x_bcast_active_rect bake-off. Run under scripts/run_safe_pytest.sh --profile.

XR_CASES selects named cases (comma separated, run in that order; "<case>~<W_NOC1_K_BLOCKS>~<T|F>"
sets the NoC1 W share and W_NOC1_AFTER_X); XR_KNOBS adds knob overrides to every case
(k=v pairs separated by ","; round sizes use ":"). Each case runs the experiment copy of the op twice; with
ABLATION=none the output is checked against torch (PCC) and for bit-identity across the two calls.
"""

import os

import pytest
import ttnn

from eval.golden_tests.wq_b_linear.helpers import WEIGHT_SCALE

if os.environ.get("XR_DESC") == "graduate":
    from ttnn.operations.wq_b_linear.perf_experiments.x_bcast_active_rect.graduate import (
        wq_b_linear_program_descriptor as pd,
    )
else:
    from ttnn.operations.wq_b_linear.perf_experiments.x_bcast_active_rect import xr_descriptor as pd

PCC = 0.9998

CASES = {
    "base": {},
    "rect_brisc": {"X_RECT": "active", "X_INJECTOR_PLACEMENT": "outside_diagonal"},
    "rect_ncrisc": {"X_RECT": "active", "X_INJECTOR_PLACEMENT": "outside_diagonal", "X_PREFETCH_RISC": "ncrisc"},
    "rect_ncrisc_row": {"X_RECT": "active", "X_INJECTOR_PLACEMENT": "outside_row", "X_PREFETCH_RISC": "ncrisc"},
    "rect_ncrisc_conc": {
        "X_RECT": "active",
        "X_INJECTOR_PLACEMENT": "outside_diagonal",
        "X_PREFETCH_RISC": "ncrisc",
        "X_ROUNDS_CONCURRENT": True,
    },
    "rect_ncrisc_small0": {
        "X_RECT": "active",
        "X_INJECTOR_PLACEMENT": "outside_diagonal",
        "X_PREFETCH_RISC": "ncrisc",
        "X_ROUND_SIZES": "2:4:5:5",
    },
}
_RECT = {"X_RECT": "active", "X_INJECTOR_PLACEMENT": "outside_diagonal"}
CASES.update(
    {
        "base_5r": {"X_ROUND_SIZES": "1:3:4:4:4"},
        "rect_brisc_conc": {**_RECT, "X_ROUNDS_CONCURRENT": True},
        "rect_brisc_small0": {**_RECT, "X_ROUND_SIZES": "2:4:5:5"},
        "rect_brisc_row": {**_RECT, "X_INJECTOR_PLACEMENT": "outside_row"},
        "rect_brisc_5r": {**_RECT, "X_ROUND_SIZES": "1:3:4:4:4"},
        "rect_ncrisc_5r": {**_RECT, "X_ROUND_SIZES": "1:3:4:4:4", "X_PREFETCH_RISC": "ncrisc"},
    }
)

# Graduate descriptor (XR_DESC=graduate) cases.
CASES.update(
    {
        "grad": {},
        "grad_off": {"X_ACTIVE_RECT": False, "X_FIRST_ROUND_K_BLOCKS": 0},
        "grad_fill_grid": {"CORE_N_TILES": 0},
        "grad_spread": {"CORE_PLACEMENT": "spread"},
        "grad_n3": {"CORE_N_TILES": 3},
        "grad_k_remainder": {"K_REMAINDER_SPLIT": True},
        "grad_rotating": {"X_MCAST_CHANNELS": "rotating", "K_BLOCK_TILES": 6, "X_ROUND_K_BLOCKS": 1},
        "grad_8_rounds": {"K_BLOCK_TILES": 6, "X_ROUND_K_BLOCKS": 1},
        "grad_concurrent": {"X_ROUNDS_CONCURRENT": True},
        "grad_w_noc1": {"W_NOC1_K_BLOCKS": 2},
    }
)


def _parse(v, ref):
    if isinstance(ref, bool):
        return v == "True"
    return type(ref)(v)


def _case_names():
    return [c for c in os.environ.get("XR_CASES", "base").split(",") if c]


@pytest.mark.parametrize("case", _case_names())
def test_xr(device, monkeypatch, case):
    import torch

    name, *w_share = case.split("~")
    knobs = dict(CASES[name])
    if w_share:
        knobs["W_NOC1_K_BLOCKS"] = int(w_share[0])
        knobs["W_NOC1_AFTER_X"] = w_share[1] == "T"
    for kv in filter(None, os.environ.get("XR_KNOBS", "").split(",")):
        k, v = kv.split("=")
        knobs[k] = _parse(v, getattr(pd, k))
    for k, v in knobs.items():
        assert hasattr(pd, k), k
        monkeypatch.setattr(pd, k, v)
    print(f"XR_CASE {case} {knobs}")

    torch.manual_seed(7)
    x = torch.randn(1, 1, 160, 1536).to(torch.bfloat16)
    w = (torch.randn(1536, 4096) * WEIGHT_SCALE).to(torch.bfloat16)
    tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tw = ttnn.from_torch(w, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=False, math_approx_mode=False
    )

    def run():
        y = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, 160, 4096]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.L1_MEMORY_CONFIG
        )
        return ttnn.generic_op([tx, tw, y], pd.create_program_descriptor(tx, tw, y, cfg))

    y0 = ttnn.to_torch(run())
    y1 = ttnn.to_torch(run())
    ttnn.synchronize_device(device)
    if getattr(pd, "ABLATION", "none") != "none":
        return
    expected = torch.matmul(x.to(torch.float64), ttnn.to_torch(tw).to(torch.float64))
    pcc = torch.corrcoef(torch.stack([y1.flatten().double(), expected.flatten()]))[0, 1].item()
    print(f"XR_PCC {case} {pcc:.6f} bit_identical={torch.equal(y0, y1)}")
    assert pcc >= PCC, pcc
    assert torch.equal(y0, y1)
