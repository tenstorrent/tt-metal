# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Guard for the bias epilogue: the non-FLUX LOOSE cases (Panoptic-DeepLab layers: L1-sharded, LoFi, bf8_b weights)
through op_base and op_copy variants (test_bias_epi.VARIANTS). Per case: whether the program runs the epilogue walk
(op_copy's LAST_EPI), bit-identity vs the first variant, PCC vs torch; each call signposted `cmp|<layer>|<variant>|0`.
Env: BE_PAN_CASES (indices into the case list), BE_PAN_VARIANTS (default base,epi), BE_LOG."""

import json
import os

import pytest
import ttnn
from tracy import signpost

from ttnn.operations.conv2d_3x3.perf_experiments.bias_epilogue.test_bias_epi import (
    HERE,
    MODS,
    VARIANTS,
    _in_collect_pass,
    _set_knobs,
)


def _custom(layer, inputs, bias, relu):
    """HEIGHT-sharded ROW_MAJOR bf16 cells with face-direct geometry (stride * C_in == 16) and W_out % 32 != 0: off
    slab-direct, so the K-blocked walk runs with one K-block (indices after the LOOSE cases)."""
    return {
        "inputs": inputs,
        "dtype": ttnn.bfloat16,
        "weight_dtype": ttnn.bfloat16,
        "layout": ttnn.ROW_MAJOR_LAYOUT,
        "memory_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        "fp32_dest_acc_en": True,
        "bias_mode": "with_bias" if bias else "no_bias",
        "activation": "relu" if relu else "none",
        "extras": {"layer": layer},
    }


def _cases():
    from eval.golden_tests.conv2d_3x3 import feature_spec

    custom = [
        _custom(f"hs_rm{c}_s{s}{'' if b else '_nb'}{'_relu' if r else ''}", ((h, w, c), 64, 1, s, 1), b, r)
        for (h, w, c, s) in ((32, 48, 16, 1), (64, 96, 8, 2))
        for b in (True, False)
        for r in (False, True)
    ]
    return [c for c in feature_spec.LOOSE_CASES if "FLUX" not in c["extras"]["layer"]] + custom


CASES = _cases()
_ONLY = {int(i) for i in os.environ.get("BE_PAN_CASES", "").split(",") if i.strip()}


@pytest.mark.parametrize("device_params", [{"l1_small_size": 4 * 1024}], indirect=True)
@pytest.mark.parametrize("idx", range(len(CASES)), ids=[c["extras"]["layer"] for c in CASES])
def test_panoptic_be(device, idx):
    import torch
    from eval.golden_tests.conv2d_3x3.helpers import (
        build_input_memory_config,
        create_ttnn_input_tensor,
        make_activation,
        make_bias,
        make_compute_kernel_config,
        make_weight,
        output_hw,
        pytorch_conv2d_3x3,
    )

    if _ONLY and idx not in _ONLY:
        pytest.skip("not selected")
    case = CASES[idx]
    extras = case["extras"]
    (H, W, c_in), c_out, padding, stride, dilation = case["inputs"]
    torch.manual_seed(0)
    c_in_real = extras.get("c_in_real")
    w = make_weight(c_out, c_in, dtype=torch.bfloat16, c_in_real=c_in_real)
    b = make_bias(c_out, dtype=torch.bfloat16) if case["bias_mode"] == "with_bias" else None
    x = make_activation(H, W, c_in, dtype=torch.bfloat16, c_in_real=c_in_real)
    act = None if case["activation"] == "none" else case["activation"]
    expected = pytorch_conv2d_3x3(
        x, w, input_height=H, input_width=W, bias=b, padding=padding, stride=stride, dilation=dilation, activation=act
    )
    h_out, w_out = output_hw(H, W, padding=padding, stride=stride, dilation=dilation)
    outs, recs = {}, []
    for vname in os.environ.get("BE_PAN_VARIANTS", "base,epi").split(","):
        pkg, knobs = VARIANTS[vname]
        op, pd = MODS[pkg]
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
                activation=act,
                compute_kernel_config=make_compute_kernel_config(
                    case["fp32_dest_acc_en"], extras.get("math_fidelity", ttnn.MathFidelity.HiFi4)
                ),
            )

        out = run()
        ttnn.synchronize_device(device)
        ttnn.ReadDeviceProfiler(device)
        for i in range(int(os.environ.get("BE_ITERS", "1"))):
            ttnn.deallocate(out)
            signpost(header=f"cmp|{extras['layer']}|{vname}|{i}")
            out = run()
            ttnn.synchronize_device(device)
            ttnn.ReadDeviceProfiler(device)
        signpost(header=f"cmpend|{extras['layer']}|{vname}")
        rec = dict(layer=extras["layer"], idx=idx, variant=vname, memory_layout=str(case["memory_layout"]))
        if pkg == "op_copy":
            rec["epi"] = getattr(pd, "LAST_EPI", None)
        if not _in_collect_pass():
            got = ttnn.to_torch(out)
            outs[vname] = got.contiguous().view(torch.int16)
            g2 = got.reshape(-1, got.shape[-1])[: h_out * w_out, :c_out].double().flatten()
            rec["pcc"] = torch.corrcoef(torch.stack([g2, expected.reshape(-1, c_out).double().flatten()]))[0, 1].item()
        recs.append(rec)
        for t in (out, tx, tw) + ((tb,) if tb is not None else ()):
            ttnn.deallocate(t)
    _set_knobs({})
    if _in_collect_pass():
        return
    names = list(outs)
    for r in recs:
        r["bit_identical_to_first"] = bool(torch.equal(outs[names[0]], outs[r["variant"]]))
        print("BE_PAN:", json.dumps(r))
        with open(HERE / "runs" / f"{os.environ.get('BE_LOG', 'pan')}.jsonl", "a") as f:
            f.write(json.dumps(r) + "\n")
