# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Whole-op A/B on a private copy of conv2d_3x3 (op_copy/): the K-blocked walk's partials CB per-K-block push + drain
(baseline, = the committed op) vs the helper's caller-owned pack target (one reserve / one push per M-block).

    SBS_OWN=0|1 scripts/run_safe_pytest.sh --profile --run-all \
        "ttnn/ttnn/operations/conv2d_3x3/perf_experiments/matmul_subblock_shape/test_sbs_op.py::test_case[semantic]" \
        --import-mode=prepend

One dispatch per process (never profile the op next to its copy; both variants run the copy, so instrumentation is
identical). The output is saved to runs/out_<case>_<variant>.pt for the cross-process bit-exactness check
(compare_outputs.py). DEVICE KERNEL DURATION [ns] is column 20 of the profiler's ops_perf_results CSV.
"""

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest
import ttnn

HERE = Path(__file__).parent
OPC = "matmul_subblock_shape_op_copy"
RESULTS = HERE / "results_op.jsonl"
RUNS = HERE / "runs"


def _load_op_copy():
    if OPC not in sys.modules:
        root = HERE / "op_copy"
        spec = importlib.util.spec_from_file_location(OPC, root / "_pkg.py", submodule_search_locations=[str(root)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[OPC] = module
        spec.loader.exec_module(module)
    return sys.modules[OPC]


opc = _load_op_copy()
pd = sys.modules[OPC + ".conv2d_3x3_program_descriptor"]
op_mod = sys.modules[OPC + ".conv2d_3x3"]


def _cases():
    from eval.golden_tests.conv2d_3x3.feature_spec import LOOSE_CASES

    out = {}
    for c in LOOSE_CASES:
        layer = c["extras"]["layer"]
        if "[perf]" in layer:
            out[layer.split(" ")[0].replace(".", "_")] = c
    return out


CASES = _cases()


def _run(case, device):
    import torch
    from eval.golden_tests.conv2d_3x3 import helpers

    torch.manual_seed(0)
    (H, W, c_in), c_out, padding, stride, dilation = case["inputs"]
    extras = case["extras"]
    dtype, weight_dtype, layout = case["dtype"], case["weight_dtype"], case["layout"]
    memory_config = helpers.build_input_memory_config(
        (1, 1, H * W, c_in), case["memory_layout"], layout=layout, dtype=dtype, device=device, extras=extras
    )
    if memory_config.shard_spec is None or list(memory_config.shard_spec.shape) != list(extras["shard_shape"]):
        pytest.skip(f"{extras['layer']}: pinned shard geometry does not fit this device's grid")
    kwargs = dict(
        stride=stride,
        padding=padding,
        dilation=dilation,
        input_dtype=dtype,
        input_layout=layout,
        weight_dtype=weight_dtype,
        input_memory_config=memory_config,
        device=device,
    )
    torch_weight = helpers.make_weight(c_out, c_in, dtype=torch.bfloat16)
    has_bias = case["bias_mode"] == "with_bias"
    activation = None if case["activation"] == "none" else case["activation"]
    torch_bias = helpers.make_bias(c_out, dtype=torch.bfloat16) if has_bias else None
    tt_weight = opc.prepare_conv2d_weights(torch_weight, **kwargs)
    tt_bias = opc.prepare_conv2d_bias(torch_bias, **kwargs) if has_bias else None
    torch_input = helpers.make_activation(H, W, c_in, dtype=torch.bfloat16)
    tt_input = ttnn.from_torch(torch_input, dtype=dtype, layout=layout, device=device, memory_config=memory_config)
    tt_output = opc.conv2d_3x3(
        tt_input,
        tt_weight,
        input_height=H,
        input_width=W,
        bias=tt_bias,
        padding=padding,
        stride=stride,
        dilation=dilation,
        activation=activation,
        compute_kernel_config=helpers.make_compute_kernel_config(
            case["fp32_dest_acc_en"], extras.get("math_fidelity", ttnn.MathFidelity.HiFi4)
        ),
    )
    ref = helpers.pytorch_conv2d_3x3(
        torch_input,
        torch_weight,
        input_height=H,
        input_width=W,
        bias=torch_bias,
        padding=padding,
        stride=stride,
        dilation=dilation,
        activation=activation,
    )
    return ttnn.to_torch(tt_output).float().reshape(ref.shape), ref.float()


@pytest.mark.parametrize("case_id", list(CASES))
def test_case(case_id, device, monkeypatch):
    import torch

    own = os.environ.get("SBS_OWN", "0") == "1"
    variant = "own" if own else "base"
    monkeypatch.setattr(pd, "SBS_CALLER_OWNS_PARTIALS", own)
    info = {}
    real = op_mod.select_blocking

    def recording(g, grid, l1_budget):
        b = real(g, grid, l1_budget)
        info.update(
            bm=b.block_m_tiles,
            bk=b.block_k_tiles,
            bn=b.block_n_tiles,
            nkb=b.num_k_blocks,
            m_blocks=b.max_m_blocks_per_core,
            sb=[b.out_subblock_h, b.out_subblock_w],
            nslice=bool(b.nslice),
            k_split=bool(g.k_split),
        )
        return b

    monkeypatch.setattr(op_mod, "select_blocking", recording)
    out, ref = _run(CASES[case_id], device)
    RUNS.mkdir(exist_ok=True)
    torch.save(out, RUNS / f"out_{case_id}_{variant}.pt")
    pcc = float(torch.corrcoef(torch.stack([ref.flatten(), out.flatten()]))[0, 1])
    info.update(case=case_id, variant=variant, pcc=pcc, max_abs=float((out - ref).abs().max()))
    with RESULTS.open("a") as f:
        f.write(json.dumps(info) + "\n")
    tol = CASES[case_id]["extras"].get("tolerance") or (0.99, None)
    assert pcc >= tol[0], info
