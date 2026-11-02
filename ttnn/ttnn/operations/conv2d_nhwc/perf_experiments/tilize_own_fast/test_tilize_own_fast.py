# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""tilize_own_fast bake-off: real op (baseline) vs variant copy (candidate) on the perf-probe cases.

Env: TOF_IMPL=baseline|candidate, TOF_CASE=sdxl|b28|nobias|width_strided|r9_l3ds|r9_l3|r9_l4,
TOF_DEFINES=comma list of compute-kernel defines for the candidate (see variant kernel head).
Run via ./run.sh <tag> (one profiled case per run).
"""
import importlib.util
import os
import sys
from pathlib import Path

import pytest
import ttnn

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[5]
sys.path.insert(0, str(REPO / "tests/ttnn/unit_tests/operations/conv2d_nhwc"))
import test_conv2d_nhwc_sharded as base  # noqa: E402

ML = base.ML


def _load_variant():
    name = "conv2d_nhwc_tilize_own_fast_variant"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, HERE / "variant" / "__init__.py", submodule_search_locations=[str(HERE / "variant")]
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_bench(device):
    impl = os.environ.get("TOF_IMPL", "baseline")  # baseline | candidate
    case = os.environ.get("TOF_CASE", "sdxl")  # sdxl b28 nobias width_strided r9_l3ds r9_l3 r9_l4
    probe_path = REPO / "tests/ttnn/unit_tests/operations/conv2d_nhwc/test_conv2d_nhwc_perf_probe.py"
    spec = importlib.util.spec_from_file_location("perf_probe_mod", probe_path)
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    if os.environ.get("TOF_NOCHECK"):  # payload-stub diagnostics only: output is wrong by design
        base._TOL = {k: (-2.0, float("inf")) for k in base._TOL}
    if impl == "candidate":
        v = _load_variant()
        # run_sharded resolves these names from its module globals.
        base.conv2d_nhwc = v.conv2d_nhwc
        base.prepare_conv2d_weights = v.prepare_conv2d_weights
        base.prepare_conv2d_bias = v.prepare_conv2d_bias
        probe.run_sharded = base.run_sharded
    fn = {
        "sdxl": "test_perf_probe_block_sdxl_l1_bf16_bf8b",
        "b28": "test_perf_probe_block_28x28x256",
        "nobias": "test_guard_block_bf16_nobias",
        "width_strided": "test_guard_width_strided",
    }
    if case in fn:
        getattr(probe, fn[case])(device)
    else:
        probe._resnet_block(device, {"r9_l3ds": "layer3_ds", "r9_l3": "layer3_3x3", "r9_l4": "layer4_3x3"}[case])
