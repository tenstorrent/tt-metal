# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""weights_on_reader_risc bake-off: BLOCK weight ring on the reader (NCRISC) vs the writer (HEAD).

WR_VARIANT=real (HEAD op, baseline) | wr (variant/ copy: weight ring on the reader).
Correctness = run_sharded's PCC / rel-RMS gate. Measure under run_safe_pytest.sh --profile."""
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


@pytest.fixture(autouse=True)
def _variant(monkeypatch):
    if os.environ.get("WR_VARIANT", "real") == "real":
        return
    spec = importlib.util.spec_from_file_location(
        "wr_variant",
        os.path.join(HERE, "variant", "__init__.py"),
        submodule_search_locations=[os.path.join(HERE, "variant")],
    )
    pkg = importlib.util.module_from_spec(spec)
    sys.modules["wr_variant"] = pkg
    spec.loader.exec_module(pkg)
    monkeypatch.setattr(ts, "conv2d_nhwc", pkg.conv2d_nhwc)
    monkeypatch.setattr(ts, "prepare_conv2d_weights", pkg.prepare_conv2d_weights)
    monkeypatch.setattr(ts, "prepare_conv2d_bias", pkg.prepare_conv2d_bias)
    if "WR_ACT_CH" in os.environ:  # exploratory: act channels with the writer freed of weights
        monkeypatch.setattr(
            sys.modules["wr_variant.conv2d_nhwc_program_descriptor"], "ACT_CHANNELS", int(os.environ["WR_ACT_CH"])
        )


def _run(device, shape, C_out, k, pad, stride, ml, bias, dt=ttnn.bfloat16, wdt=ttnn.bfloat8_b, **cfg):
    ts.run_sharded(device, shape, C_out, k, pad, stride, ml, bias, dt, wdt, **cfg)


def test_focus_sdxl(device):
    _run(device, (1, 64, 64, 640), 640, 3, 1, 1, ML.BLOCK_SHARDED, True, shard_shape=(410, 64), core_grid=(10, 10))


RESNET_BLOCK = {
    "layer3_ds": ((16, 28, 28, 256), 256, 3, 1, 2, (1280, 32), (8, 10)),
    "layer3_3x3": ((16, 14, 14, 256), 256, 3, 1, 1, (314, 32), (8, 10)),
    "layer4_3x3": ((16, 7, 7, 512), 512, 3, 1, 1, (96, 64), (8, 9)),
}


@pytest.mark.parametrize("name", list(RESNET_BLOCK))
def test_resnet_block(device, name):
    shape, C_out, k, pad, stride, shard, grid = RESNET_BLOCK[name]
    _run(device, shape, C_out, k, pad, stride, ML.BLOCK_SHARDED, True, shard_shape=shard, core_grid=grid)


def test_guard_block_bf16_nobias(device):
    _run(device, (1, 28, 28, 256), 256, 3, 1, 1, ML.BLOCK_SHARDED, False)


def test_block_28x28x256(device):
    ts.run_sharded(device, (1, 28, 28, 256), 256, 3, 1, 1, ML.BLOCK_SHARDED, True)


def test_guard_width_strided(device):
    ts.run_sharded(device, (1, 14, 14, 256), 512, 3, 1, 2, ML.WIDTH_SHARDED, True)


# Correctness breadth on the BLOCK path (fp32 default dtypes, as the sharded acceptance suite).
@pytest.mark.parametrize("with_bias", [True, False], ids=["bias", "no_bias"])
@pytest.mark.parametrize(
    "input_shape, C_out, k, padding, stride",
    [
        ((1, 8, 8, 64), 64, 3, 1, 1),
        ((2, 14, 14, 128), 128, 3, 1, 1),
        ((1, 32, 32, 256), 320, 3, 1, 1),
        ((1, 28, 28, 256), 256, 3, 1, 2),
    ],
)
def test_block_correctness(device, input_shape, C_out, k, padding, stride, with_bias):
    ts.run_sharded(device, input_shape, C_out, k, padding, stride, ML.BLOCK_SHARDED, with_bias)
