# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Which [perf] focus cases take kw-fold in the base op vs this experiment's copy (the copy's kw-fold CBs are smaller,
so its L1 check can only admit more cases). Records kw_fold_mode's verdict on every dispatch; no PCC, no profiler.

    scripts/run_safe_pytest.sh --run-all --import-mode=prepend \
        ttnn/ttnn/operations/conv2d_3x3/perf_experiments/kw_fold_seedless/test_kwfs_selection.py
"""

import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from kwfs_copy import copy_modules  # noqa: E402


def _cases():
    from eval.golden_tests.conv2d_3x3.feature_spec import LOOSE_CASES

    cases = [c for c in LOOSE_CASES if "[perf]" in c["extras"]["layer"]]
    return cases, [c["extras"]["layer"].split(" @")[0] for c in cases]


_CASES, _IDS = _cases()


@pytest.mark.parametrize("case", _CASES, ids=[f"c{i:02d}" for i in range(len(_CASES))])
def test_kwfs_selection(case, device):
    import torch
    import ttnn
    from eval.golden_tests.conv2d_3x3 import helpers

    base_pd = importlib.import_module("ttnn.operations.conv2d_3x3.conv2d_3x3_program_descriptor")
    base_op = importlib.import_module("ttnn.operations.conv2d_3x3.conv2d_3x3")
    copy_op, copy_pd = copy_modules()
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
    has_bias = case["bias_mode"] == "with_bias"
    tb = helpers.make_bias(c_out, dtype=torch.bfloat16) if has_bias else None
    tx = helpers.make_activation(H, W, c_in, dtype=torch.bfloat16)
    verdicts = {}
    for name, op_mod, pd in (("base", base_op, base_pd), ("copy", copy_op, copy_pd)):
        orig = pd.kw_fold_mode
        seen = []
        pd.kw_fold_mode = lambda *a, _o=orig, _s=seen: _s.append(_o(*a)) or _s[-1]
        try:
            x = ttnn.from_torch(tx, dtype=dtype, layout=layout, device=device, memory_config=mc)
            op_mod.conv2d_3x3(
                x,
                op_mod.prepare_conv2d_weights(tw, **kwargs),
                input_height=H,
                input_width=W,
                bias=op_mod.prepare_conv2d_bias(tb, **kwargs) if has_bias else None,
                padding=padding,
                stride=stride,
                dilation=dilation,
                activation=None if case["activation"] == "none" else case["activation"],
                compute_kernel_config=helpers.make_compute_kernel_config(
                    case["fp32_dest_acc_en"], extras.get("math_fidelity", ttnn.MathFidelity.HiFi4)
                ),
            )
        finally:
            pd.kw_fold_mode = orig
        verdicts[name] = seen
    print("KWFS_SEL", extras["layer"].split(" @")[0], verdicts)
    assert verdicts["base"] == verdicts["copy"], f"kw-fold selection changed: {verdicts}"
