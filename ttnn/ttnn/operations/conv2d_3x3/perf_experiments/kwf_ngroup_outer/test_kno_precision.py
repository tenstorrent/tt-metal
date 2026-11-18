# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Precision of the n-group-outer kw-fold copy vs the base op (ttnn.operations.conv2d_3x3 at 3ca4eb28f46) on the
[perf] focus cases it touches (sem res2 fc0 / fc1, now kw-fold) and the existing kw-fold cases (same arithmetic: N <=
sbw): PCC, mean / max abs error of each against a fp32 torch conv on the bf8_b-quantized operands, the share of
bit-identical outputs and the largest base-vs-copy difference. No profiler (both ops run in one process).

    scripts/run_safe_pytest.sh --run-all --import-mode=prepend \
        ttnn/ttnn/operations/conv2d_3x3/perf_experiments/kwf_ngroup_outer/test_kno_precision.py
"""

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from kno_copy import copy_modules  # noqa: E402

OUT = Path(__file__).resolve().parent / "precision.jsonl"
FOCUS_IDS = [
    "semantic3",
    "semantic2",
    "stem_conv2",
    "stem_conv3",
    "res2_{0,1,2}_conv2",
    "res3_{1,2,3}_conv2",
    "instance0",
    "instance1",
    "instance2",
    "center_head_1",
]


def _cases():
    from eval.golden_tests.conv2d_3x3.feature_spec import LOOSE_CASES

    cases = [c for c in LOOSE_CASES if "[perf]" in c["extras"]["layer"]]
    ids = [c["extras"]["layer"].split(" ")[0].replace(".", "_") for c in cases]
    # The perf harness's ids repeat for the decoder layers (semantic / instance): number them in order.
    seen, out_ids = {}, []
    for i in ids:
        if i in ("semantic", "instance"):
            seen[i] = seen.get(i, -1) + 1
            i = f"{i}{seen[i]}"
        out_ids.append(i)
    keep = [(c, i) for c, i in zip(cases, out_ids) if i in FOCUS_IDS]
    return [c for c, _ in keep], [i for _, i in keep]


_CASES, _IDS = _cases()


@pytest.mark.parametrize("case", _CASES, ids=_IDS)
def test_kno_precision(case, device, request, monkeypatch):
    import torch
    import ttnn
    from eval.golden_tests.conv2d_3x3 import helpers
    from ttnn.operations.conv2d_3x3 import conv2d_3x3 as base_op, prepare_conv2d_bias, prepare_conv2d_weights

    op_mod, pd_mod = copy_modules()
    taken = []
    real_mode = pd_mod.kw_fold_mode
    monkeypatch.setattr(pd_mod, "kw_fold_mode", lambda *a: taken.append(real_mode(*a)) or taken[-1])
    torch.manual_seed(0)
    (H, W, c_in), c_out, padding, stride, dilation = case["inputs"]
    extras = case["extras"]
    dtype, weight_dtype, layout = case["dtype"], case["weight_dtype"], case["layout"]
    mc = helpers.build_input_memory_config(
        (1, 1, H * W, c_in), case["memory_layout"], layout=layout, dtype=dtype, device=device, extras=extras
    )
    kwargs = dict(
        stride=stride,
        padding=padding,
        dilation=dilation,
        input_dtype=dtype,
        input_layout=layout,
        weight_dtype=weight_dtype,
        input_memory_config=mc,
        device=device,
    )
    tw = helpers.make_weight(c_out, c_in, dtype=torch.bfloat16)
    tb = helpers.make_bias(c_out, dtype=torch.bfloat16)
    tx = helpers.make_activation(H, W, c_in, dtype=torch.bfloat16)
    if os.environ.get("KNO_FLIP"):
        # Stale-output guard: an input no earlier run used (a kernel that never wrote its output fails the PCC).
        tx = -tx.flip(-2)
    act = None if case["activation"] == "none" else case["activation"]
    xq = ttnn.to_torch(ttnn.from_torch(tx, dtype=dtype, layout=ttnn.TILE_LAYOUT)).float()
    wq = ttnn.to_torch(ttnn.from_torch(tw, dtype=weight_dtype, layout=ttnn.TILE_LAYOUT)).float()
    bq = ttnn.to_torch(ttnn.from_torch(tb.reshape(1, c_out), dtype=weight_dtype, layout=ttnn.TILE_LAYOUT)).float()[0]
    xn = xq.reshape(1, H, W, c_in).permute(0, 3, 1, 2)
    ref = torch.nn.functional.conv2d(xn, wq, bias=bq, stride=stride, padding=padding, dilation=dilation)
    if act == "relu":
        ref = torch.relu(ref)
    ref = ref.permute(0, 2, 3, 1).reshape(1, 1, -1, c_out)
    outs = {}
    cfg = helpers.make_compute_kernel_config(case["fp32_dest_acc_en"], extras["math_fidelity"])
    # The copy first: an output that a kernel never wrote cannot hold the other op's result (same L1 address).
    for name, fn in (("copy", op_mod.conv2d_3x3), ("base", base_op)):
        x = ttnn.from_torch(tx, dtype=dtype, layout=layout, device=device, memory_config=mc)
        # Two dispatches: the second is the program-cache hit (fresh buffers, same shapes).
        for _ in range(2):
            y = fn(
                x,
                prepare_conv2d_weights(tw, **kwargs),
                input_height=H,
                input_width=W,
                bias=prepare_conv2d_bias(tb, **kwargs),
                padding=padding,
                stride=stride,
                dilation=dilation,
                activation=act,
                compute_kernel_config=cfg,
            )
            o = ttnn.to_torch(y).float().reshape(ref.shape)
            if name in outs:
                assert torch.equal(outs[name], o), f"{name}: cache-hit dispatch differs"
            outs[name] = o
    rec = {"case": request.node.callspec.id, "layer": extras["layer"], "relu": act == "relu", "copy_kwf": taken[:1]}
    for name, o in outs.items():
        a, b = o.flatten(), ref.flatten()
        pcc = torch.corrcoef(torch.stack([a, b]))[0, 1].item()
        rec[name] = {
            "pcc": round(pcc, 6),
            "mae": round((a - b).abs().mean().item(), 5),
            "maxe": round((a - b).abs().max().item(), 4),
        }
    d = (outs["copy"] - outs["base"]).reshape(-1, c_out)
    rec["identical_frac"] = round((d == 0).float().mean().item(), 6)
    rec["max_base_copy_diff"] = round(d.abs().max().item(), 5)
    print("KNO_PREC", json.dumps(rec))
    with open(OUT, "a") as f:
        f.write(json.dumps(rec) + "\n")
    pcc_min, _ = extras.get("tolerance", (0.99, None))
    assert rec["copy"]["pcc"] >= pcc_min
    # Not worse than the base beyond noise: mean abs error within 2% of the base's.
    assert rec["copy"]["mae"] <= rec["base"]["mae"] * 1.02 + 1e-6
