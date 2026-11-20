# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Guard: every non-FLUX LOOSE case (the Panoptic-DeepLab layers) through op_base and op_final (the integration
candidate: test_wsd2.VARIANTS["final"]). Outputs must be bit-identical; each call is signposted `cmp|<layer>|<variant>|0`
for cmp_windows.py. WSD_PAN_VARIANTS (default base,final). Run: WSD_TEST=<this file> run.sh <label> 0 -"""

import json
import os

import pytest
import ttnn
from tracy import signpost

from ttnn.operations.conv2d_3x3.perf_experiments.weight_stream_decouple_v2.test_wsd2 import (
    HERE,
    MODS,
    VARIANTS,
    _in_collect_pass,
    _set_knobs,
)


def _cases():
    from eval.golden_tests.conv2d_3x3 import feature_spec

    return [c for c in feature_spec.LOOSE_CASES if "FLUX" not in c["extras"]["layer"]]


CASES = _cases()


@pytest.mark.parametrize("idx", range(len(CASES)), ids=[c["extras"]["layer"] for c in CASES])
def test_panoptic_guard(device, idx):
    import torch
    from eval.golden_tests.conv2d_3x3.helpers import (
        build_input_memory_config,
        create_ttnn_input_tensor,
        make_activation,
        make_bias,
        make_compute_kernel_config,
        make_weight,
    )

    case = CASES[idx]
    extras = case["extras"]
    (H, W, c_in), c_out, padding, stride, dilation = case["inputs"]
    torch.manual_seed(0)
    c_in_real = extras.get("c_in_real")
    w = make_weight(c_out, c_in, dtype=torch.bfloat16, c_in_real=c_in_real)
    b = make_bias(c_out, dtype=torch.bfloat16) if case["bias_mode"] == "with_bias" else None
    x = make_activation(H, W, c_in, dtype=torch.bfloat16, c_in_real=c_in_real)
    outs = {}
    for vname in os.environ.get("WSD_PAN_VARIANTS", "base,final").split(","):
        pkg, knobs = VARIANTS[vname]
        op, _ = MODS[pkg]
        _set_knobs(knobs if pkg == "op_copy" else {})
        device.clear_program_cache()
        mc = build_input_memory_config(
            (1, 1, H * W, c_in),
            case["memory_layout"],
            layout=case["layout"],
            dtype=case["dtype"],
            device=device,
            extras=extras,
        )
        prep = dict(
            stride=stride,
            padding=padding,
            dilation=dilation,
            input_dtype=case["dtype"],
            input_layout=case["layout"],
            weight_dtype=case["weight_dtype"],
            input_memory_config=mc,
            device=device,
        )
        tw = op.prepare_conv2d_weights(w, **prep)
        tb = op.prepare_conv2d_bias(b, **prep) if b is not None else None
        tx = create_ttnn_input_tensor(x, device, dtype=case["dtype"], layout=case["layout"], memory_config=mc)

        def run():
            return op.conv2d_3x3(
                tx,
                tw,
                input_height=H,
                input_width=W,
                bias=tb,
                padding=padding,
                stride=stride,
                dilation=dilation,
                activation=None if case["activation"] == "none" else case["activation"],
                compute_kernel_config=make_compute_kernel_config(
                    case["fp32_dest_acc_en"], extras.get("math_fidelity", ttnn.MathFidelity.HiFi4)
                ),
            )

        out = run()
        ttnn.synchronize_device(device)
        ttnn.ReadDeviceProfiler(device)
        ttnn.deallocate(out)
        signpost(header=f"cmp|{extras['layer']}|{vname}|0")
        out = run()
        ttnn.synchronize_device(device)
        ttnn.ReadDeviceProfiler(device)
        signpost(header=f"cmpend|{extras['layer']}|{vname}")
        if not _in_collect_pass():
            outs[vname] = ttnn.to_torch(out).contiguous().view(torch.int16)
        for t in (out, tx, tw) + ((tb,) if tb is not None else ()):
            ttnn.deallocate(t)
    _set_knobs({})
    if _in_collect_pass():
        return
    names = list(outs)
    same = all(torch.equal(outs[names[0]], outs[n]) for n in names[1:])
    rec = dict(layer=extras["layer"], memory_layout=str(case["memory_layout"]), bit_identical=same)
    print("WSD_PAN:", json.dumps(rec))
    with open(HERE / "runs" / f"{os.environ.get('WSD_LOG', 'pan')}.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    assert same, rec
