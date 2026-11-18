# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Correctness of the integration patches (integration_A_sender_stagger.patch + integration_B_chunk_layout.patch,
applied to a scratch copy of the op in int_check/): the patched op's output must be bit-identical (torch.equal) to the
real op's on the same inputs, on the N-slice-outer walk and on the K-blocked walk (forced by making every N-slice
candidate not fit). Not profiled (the real op and its copy in one process would collide on zone ids under the profiler).

    scripts/run_safe_pytest.sh --run-all \\
        ttnn/ttnn/operations/conv2d_3x3/perf_experiments/weight_bank_spread/test_integration_check.py \\
        -p no:cacheprovider --import-mode=prepend
"""

import importlib
import importlib.util
import sys
import types
from pathlib import Path

import pytest
import ttnn

HERE = Path(__file__).parent
ICP = "weight_bank_spread_int_check"
if ICP not in sys.modules:
    _pkg = types.ModuleType(ICP)
    _pkg.__path__ = [str(HERE / "int_check")]
    sys.modules[ICP] = _pkg
new_pd = importlib.import_module(ICP + ".conv2d_3x3_program_descriptor")
new_op = importlib.import_module(ICP + ".conv2d_3x3")
old_pd = importlib.import_module("ttnn.operations.conv2d_3x3.conv2d_3x3_program_descriptor")
old_op = importlib.import_module("ttnn.operations.conv2d_3x3.conv2d_3x3")

CASES = ["block_unaligned", "focus:aspp_convs_1", "focus:res5_1_conv2", "focus:res4_0_conv2", "block:res5_0_conv2"]


def _harness():
    spec = importlib.util.spec_from_file_location("wbs_harness_for_check", HERE / "test_weight_bank_spread.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_H = _harness()


def _force_kblocked(monkeypatch, pd_mod):
    real = pd_mod._footprint

    def no_nslice(*args, **kwargs):
        fit = real(*args, **kwargs)
        nslice = kwargs.get("nslice", args[6] if len(args) > 6 else False)
        if nslice:
            fit = dict(fit, footprint_bytes=1 << 40)
        return fit

    monkeypatch.setattr(pd_mod, "_footprint", no_nslice)
    pd_mod.select_blocking.cache_clear()


def _run(op_mod, spec, torch_input, torch_weight, torch_bias, device):
    kwargs = dict(
        stride=spec["stride"],
        padding=spec["padding"],
        dilation=spec["dilation"],
        input_dtype=spec["dtype"],
        input_layout=spec["layout"],
        weight_dtype=spec["weight_dtype"],
        input_memory_config=spec["mem"],
        device=device,
    )
    tt_weight = op_mod.prepare_conv2d_weights(torch_weight, **kwargs)
    tt_bias = op_mod.prepare_conv2d_bias(torch_bias, **kwargs) if torch_bias is not None else None
    tt_input = ttnn.from_torch(
        torch_input, dtype=spec["dtype"], layout=spec["layout"], device=device, memory_config=spec["mem"]
    )
    out = op_mod.conv2d_3x3(
        tt_input,
        tt_weight,
        input_height=spec["H"],
        input_width=spec["W"],
        bias=tt_bias,
        padding=spec["padding"],
        stride=spec["stride"],
        dilation=spec["dilation"],
        activation=spec["activation"],
        compute_kernel_config=spec["cfg"],
    )
    return ttnn.to_torch(out).float()


@pytest.mark.parametrize("walk", ["nslice", "kblocked"])
@pytest.mark.parametrize("case_token", CASES)
def test_integration_bit_exact(case_token, walk, device, monkeypatch):
    import torch

    spec, torch_input, torch_weight, torch_bias = _H._build(case_token, device)
    if walk == "kblocked":
        _force_kblocked(monkeypatch, old_pd)
        _force_kblocked(monkeypatch, new_pd)
    else:
        old_pd.select_blocking.cache_clear()
        new_pd.select_blocking.cache_clear()
    device.clear_program_cache()
    ref = _run(old_op, spec, torch_input, torch_weight, torch_bias, device)
    device.clear_program_cache()
    got = _run(new_op, spec, torch_input, torch_weight, torch_bias, device)
    assert torch.equal(ref, got), f"{case_token} {walk}: patched op is not bit-identical to the op"
    old_pd.select_blocking.cache_clear()
    new_pd.select_blocking.cache_clear()
