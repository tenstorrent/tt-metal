# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""fewer_rounds_bigger_kblock bake-off: the focus case + guards through either the real op
(FRB_VARIANT=0, baseline) or the variant copy in ./variant with knobs from env:
  FRB_BK (force bk), FRB_BM, FRB_TAP (0 = band walk), FRB_RECV (ACT_RECV_DEPTH),
  FRB_SEND (ACT_SEND_DEPTH), FRB_RMK (ACT_RM_KBLOCKS), FRB_WD (SHARDED_WEIGHT_DEPTH),
  FRB_LEAD (ACT_TILIZE_LEAD), FRB_ALIAS=1 (cb_act_send aliases the cb_act_tiles ring).
Correctness gate = run_sharded's (PCC >= 0.995, rel-RMS <= 0.05 for bf16)."""
import importlib.util
import os
import sys
from pathlib import Path

import pytest
import ttnn

HERE = Path(__file__).parent
REPO = HERE.parents[5]
TDIR = REPO / "tests/ttnn/unit_tests/operations/conv2d_nhwc"
sys.path.insert(0, str(TDIR))

import test_conv2d_nhwc_sharded as ts  # noqa: E402
from test_conv2d_nhwc_sharded import ML  # noqa: E402


def _variant():
    name = "frb_variant"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, HERE / "variant/__init__.py", submodule_search_locations=[str(HERE / "variant")]
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(autouse=True)
def _route(monkeypatch):
    if os.environ.get("FRB_VARIANT", "1") == "0":
        yield
        return
    v = _variant()
    pd = sys.modules["frb_variant.conv2d_nhwc_program_descriptor"]
    knobs = {
        "FRB_BK": "FORCE_BK",
        "FRB_BM": "FORCE_BM",
        "FRB_RECV": "ACT_RECV_DEPTH",
        "FRB_SEND": "ACT_SEND_DEPTH",
        "FRB_RMK": "ACT_RM_KBLOCKS",
        "FRB_WD": "SHARDED_WEIGHT_DEPTH",
        "FRB_LEAD": "ACT_TILIZE_LEAD",
    }
    for env, attr in knobs.items():
        if os.environ.get(env):
            monkeypatch.setattr(pd, attr, int(os.environ[env]))
    if os.environ.get("FRB_ALIAS") == "1":
        monkeypatch.setattr(pd, "ACT_SEND_ALIAS", True)
    if os.environ.get("FRB_TAP") == "0":
        monkeypatch.setattr(pd, "FORCE_TAP_MAJOR", False)
    monkeypatch.setattr(ts, "conv2d_nhwc", v.conv2d_nhwc)
    monkeypatch.setattr(ts, "prepare_conv2d_weights", v.prepare_conv2d_weights)
    monkeypatch.setattr(ts, "prepare_conv2d_bias", v.prepare_conv2d_bias)
    yield


def _sdxl(device):
    ts.run_sharded(
        device,
        (1, 64, 64, 640),
        640,
        3,
        1,
        1,
        ML.BLOCK_SHARDED,
        True,
        ttnn.bfloat16,
        ttnn.bfloat8_b,
        shard_shape=(410, 64),
        core_grid=(10, 10),
    )


def test_focus(device):
    _sdxl(device)


def _peak(device, fn):
    device.disable_and_clear_program_cache()
    ttnn.graph._cpp_begin_graph_capture(ttnn.graph.RunMode.NORMAL)
    try:
        fn(device)
    finally:
        trace = ttnn.graph._cpp_end_graph_capture()
        device.enable_program_cache()
    peak = int(ttnn.graph.extract_resource_usage_per_core(trace).peak_total)
    print(f"L1_PEAK_BYTES={peak}")


def test_l1_focus(device):
    _peak(device, _sdxl)


RESNET_BLOCK = {
    "layer3_ds": ((16, 28, 28, 256), 256, 3, 1, 2, (1280, 32), (8, 10)),
    "layer3_3x3": ((16, 14, 14, 256), 256, 3, 1, 1, (314, 32), (8, 10)),
    "layer4_3x3": ((16, 7, 7, 512), 512, 3, 1, 1, (96, 64), (8, 9)),
}


def _resnet(name):
    shape, C_out, k, pad, stride, shard, grid = RESNET_BLOCK[name]

    def f(device):
        ts.run_sharded(
            device,
            shape,
            C_out,
            k,
            pad,
            stride,
            ML.BLOCK_SHARDED,
            True,
            ttnn.bfloat16,
            ttnn.bfloat8_b,
            shard_shape=shard,
            core_grid=grid,
        )

    return f


@pytest.mark.parametrize("name", list(RESNET_BLOCK))
def test_resnet(device, name):
    _resnet(name)(device)


@pytest.mark.parametrize("name", list(RESNET_BLOCK))
def test_l1_resnet(device, name):
    _peak(device, _resnet(name))


def test_guard_block_bf16_nobias(device):
    ts.run_sharded(device, (1, 28, 28, 256), 256, 3, 1, 1, ML.BLOCK_SHARDED, False, ttnn.bfloat16, ttnn.bfloat8_b)


def test_guard_width_strided(device):
    ts.run_sharded(device, (1, 14, 14, 256), 512, 3, 1, 2, ML.WIDTH_SHARDED, True)


def test_block_28x28x256(device):
    ts.run_sharded(device, (1, 28, 28, 256), 256, 3, 1, 1, ML.BLOCK_SHARDED, True)
