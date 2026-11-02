# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""dual_noc_act_mcast bench: HEAD op (impl=baseline) vs the variant copy (impl=candidate).

Candidate knobs (env, read when the program is built): DNA_MODE=1 enables the raw dual-NoC act
exchange, DNA_B16 = sixteenths of each round's bytes sent on NOC0 (0 = raw protocol, NOC1 only),
DNA_CHUNK = bytes per interleaved burst. Correctness is run_sharded's PCC / rel-RMS gate.
Run one case per process under scripts/run_safe_pytest.sh --profile ... -k '<impl> and <case>'.
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
import test_conv2d_nhwc_sharded as ts  # noqa: E402

ML = ttnn.TensorMemoryLayout


def _load_variant():
    name = "dna_variant_conv2d_nhwc"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, HERE / "variant" / "__init__.py", submodule_search_locations=[str(HERE / "variant")]
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


CASES = {
    # focus: SDXL L1 3x3, BLOCK 10x10, bf16 x bf8b, bias
    "sdxl": ((1, 64, 64, 640), 640, 3, 1, 1, True, dict(shard_shape=(410, 64), core_grid=(10, 10))),
    "layer3_ds": ((16, 28, 28, 256), 256, 3, 1, 2, True, dict(shard_shape=(1280, 32), core_grid=(8, 10))),
    "layer3_3x3": ((16, 14, 14, 256), 256, 3, 1, 1, True, dict(shard_shape=(314, 32), core_grid=(8, 10))),
    "layer4_3x3": ((16, 7, 7, 512), 512, 3, 1, 1, True, dict(shard_shape=(96, 64), core_grid=(8, 9))),
    "block28_nobias": ((1, 28, 28, 256), 256, 3, 1, 1, False, {}),
    "block28_f32": ((1, 28, 28, 256), 256, 3, 1, 1, True, dict(_f32=True)),
}


@pytest.mark.parametrize("case", list(CASES))
@pytest.mark.parametrize("impl", ["baseline", "candidate"])
def test_dna(device, impl, case, monkeypatch):
    if impl == "candidate":
        v = _load_variant()
        monkeypatch.setattr(ts, "conv2d_nhwc", v.conv2d_nhwc)
        monkeypatch.setattr(ts, "prepare_conv2d_weights", v.prepare_conv2d_weights)
        monkeypatch.setattr(ts, "prepare_conv2d_bias", v.prepare_conv2d_bias)
        os.environ.setdefault("DNA_MODE", "1")
    shape, c_out, k, pad, stride, bias, cfg = CASES[case]
    cfg = dict(cfg)
    dts = (ttnn.float32, ttnn.float32) if cfg.pop("_f32", False) else (ttnn.bfloat16, ttnn.bfloat8_b)
    print(f"DNA impl={impl} case={case} DNA_MODE={os.environ.get('DNA_MODE')} B16={os.environ.get('DNA_B16', '8')}")
    ts.run_sharded(device, shape, c_out, k, pad, stride, ML.BLOCK_SHARDED, bias, *dts, **cfg)
