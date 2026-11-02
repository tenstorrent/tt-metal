# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""gather_fast bake-off: BLOCK-path im2col gather issue rate / startup.

Env (one variant per process):
  GF_VARIANT  'real' (HEAD op, baseline) | 'variant' (variant/ copy)
  GF_MODE     variant reader define (see variant/kernels/conv2d_nhwc_gather.hpp): 0 = HEAD code."""
import importlib.util
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
    if os.environ.get("GF_VARIANT", "real") == "real":
        return
    vdir = os.path.join(HERE, "variant")
    spec = importlib.util.spec_from_file_location(
        "gf_variant", os.path.join(vdir, "__init__.py"), submodule_search_locations=[vdir]
    )
    pkg = importlib.util.module_from_spec(spec)
    sys.modules["gf_variant"] = pkg
    spec.loader.exec_module(pkg)
    monkeypatch.setattr(ts, "conv2d_nhwc", pkg.conv2d_nhwc)
    monkeypatch.setattr(ts, "prepare_conv2d_weights", pkg.prepare_conv2d_weights)
    monkeypatch.setattr(ts, "prepare_conv2d_bias", pkg.prepare_conv2d_bias)


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


# Correctness sweep of the sharded unit cases (edges, ragged, uneven, strided, 1x1) on the variant.
@pytest.mark.parametrize("args", [c.values for c in ts.CASES], ids=[c.id for c in ts.CASES])
def test_sharded_cases(device, monkeypatch, args):
    _patch(monkeypatch)
    shape, C_out, k, pad, stride, ml = args
    ts.run_sharded(device, shape, C_out, k, pad, stride, ml, True)


def test_guard_interleaved_bf16_relu(device, monkeypatch):
    _patch(monkeypatch)
    ts.run_sharded(
        device, (1, 28, 28, 256), 256, 3, 1, 1, ML.INTERLEAVED, True, ttnn.bfloat16, ttnn.bfloat8_b, activation="relu"
    )


def test_height_layer2_ds(device, monkeypatch):
    _patch(monkeypatch)
    ts.run_sharded(
        device,
        (16, 56, 56, 128),
        128,
        3,
        1,
        2,
        ML.HEIGHT_SHARDED,
        True,
        ttnn.bfloat16,
        ttnn.bfloat8_b,
        shard_shape=(457, 128),
        num_cores=110,
    )
