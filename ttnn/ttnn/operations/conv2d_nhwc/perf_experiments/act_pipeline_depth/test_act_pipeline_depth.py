# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""act_pipeline_depth bake-off: BLOCK act exchange ack lead / landing depth.

Knobs (env, one variant per process so the JIT cache / program cache is fresh per variant):
  AP_CH      act channels (= receiver ack lead; Flag signal => one outstanding round per channel)
  AP_RD      ACT_RECV_DEPTH (cb_act_tiles depth = AP_RD + AP_CH - 1)
  AP_BUDGET  SHARDED_CB_BUDGET_BYTES in KiB (only to keep the same bm/bk plan when the ring deepens)
  AP_DEFINES comma list of writer defines (variant only), e.g. AP_NO_SEND_AHEAD
  AP_VARIANT 'real' (default: real op + monkeypatched host knobs) | 'nch' (variant/ copy with the
             N-channel writer)
Unset knobs = HEAD defaults (baseline)."""
import importlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TT = os.path.abspath(os.path.join(HERE, *[".."] * 6))
sys.path.insert(0, os.path.join(TT, "tests/ttnn/unit_tests/operations/conv2d_nhwc"))

import pytest  # noqa: E402
import ttnn  # noqa: E402
import test_conv2d_nhwc_sharded as ts  # noqa: E402

ML = ts.ML


def _patch(monkeypatch):
    variant = os.environ.get("AP_VARIANT", "real")
    if variant == "real":
        import ttnn.operations.conv2d_nhwc  # noqa: F401

        op = sys.modules["ttnn.operations.conv2d_nhwc.conv2d_nhwc"]
        pd = sys.modules["ttnn.operations.conv2d_nhwc.conv2d_nhwc_program_descriptor"]
    else:
        spec = importlib.util.spec_from_file_location(
            "ap_variant",
            os.path.join(HERE, "variant", "__init__.py"),
            submodule_search_locations=[os.path.join(HERE, "variant")],
        )
        pkg = importlib.util.module_from_spec(spec)
        sys.modules["ap_variant"] = pkg
        spec.loader.exec_module(pkg)
        op = sys.modules["ap_variant.conv2d_nhwc"]
        pd = sys.modules["ap_variant.conv2d_nhwc_program_descriptor"]
        monkeypatch.setattr(ts, "conv2d_nhwc", pkg.conv2d_nhwc)
        monkeypatch.setattr(ts, "prepare_conv2d_weights", pkg.prepare_conv2d_weights)
        monkeypatch.setattr(ts, "prepare_conv2d_bias", pkg.prepare_conv2d_bias)
    if os.environ.get("AP_DEFINES"):
        monkeypatch.setattr(pd, "WRITER_DEFINES", [(d, "1") for d in os.environ["AP_DEFINES"].split(",")])
    if "AP_TL" in os.environ:  # diagnostic: ACT_TILIZE_LEAD
        monkeypatch.setattr(pd, "ACT_TILIZE_LEAD", int(os.environ["AP_TL"]))
    if "AP_CH" in os.environ:
        monkeypatch.setattr(pd, "ACT_CHANNELS", int(os.environ["AP_CH"]))
    if "AP_RD" in os.environ:
        monkeypatch.setattr(pd, "ACT_RECV_DEPTH", int(os.environ["AP_RD"]))
    if "AP_BUDGET" in os.environ:
        monkeypatch.setattr(pd, "SHARDED_CB_BUDGET_BYTES", int(os.environ["AP_BUDGET"]) * 1024)
    orig = op.select_blocking_sharded

    def wrapped(*a, **k):
        p = orig(*a, **k)
        keys = [
            "block_m_tiles",
            "block_k_tiles",
            "block_n_tiles",
            "num_k_blocks",
            "act_channels",
            "act_recv_depth",
            "footprint",
        ]
        print("AP_PLAN", {kk: p.get(kk) for kk in keys if kk in p})
        return p

    monkeypatch.setattr(op, "select_blocking_sharded", wrapped)


def test_focus_sdxl(device, monkeypatch):
    _patch(monkeypatch)
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


def test_l1_peak_sdxl(device, monkeypatch):
    device.disable_and_clear_program_cache()
    ttnn.graph._cpp_begin_graph_capture(ttnn.graph.RunMode.NORMAL)
    try:
        test_focus_sdxl(device, monkeypatch)
    finally:
        trace = ttnn.graph._cpp_end_graph_capture()
        device.enable_program_cache()
    print(f"L1_PEAK_BYTES={int(ttnn.graph.extract_resource_usage_per_core(trace).peak_total)}")


RESNET_BLOCK = {
    "layer3_ds": ((16, 28, 28, 256), 256, 3, 1, 2, (1280, 32), (8, 10)),
    "layer3_3x3": ((16, 14, 14, 256), 256, 3, 1, 1, (314, 32), (8, 10)),
    "layer4_3x3": ((16, 7, 7, 512), 512, 3, 1, 1, (96, 64), (8, 9)),
}


@pytest.mark.parametrize("name", list(RESNET_BLOCK))
def test_resnet_block(device, monkeypatch, name):
    _patch(monkeypatch)
    shape, C_out, k, pad, stride, shard, grid = RESNET_BLOCK[name]
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


def test_guard_block_bf16_nobias(device, monkeypatch):
    _patch(monkeypatch)
    ts.run_sharded(device, (1, 28, 28, 256), 256, 3, 1, 1, ML.BLOCK_SHARDED, False, ttnn.bfloat16, ttnn.bfloat8_b)


def test_guard_block_28x28x256(device, monkeypatch):
    _patch(monkeypatch)
    ts.run_sharded(device, (1, 28, 28, 256), 256, 3, 1, 1, ML.BLOCK_SHARDED, True)


def test_guard_width_strided(device, monkeypatch):
    _patch(monkeypatch)
    ts.run_sharded(device, (1, 14, 14, 256), 512, 3, 1, 2, ML.WIDTH_SHARDED, True)
