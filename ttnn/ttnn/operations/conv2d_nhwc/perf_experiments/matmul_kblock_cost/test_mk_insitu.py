# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""r2_breakdown bench: the BLOCK act-exchange per-round cost, baseline (real op) vs the variant copy.

Env:
  HF_IMPL    = op | variant            (op = the real conv2d_nhwc at HEAD; variant = ./variant)
  HF_DEFINES = comma list of kernel defines for the variant (stubs: STUB_ACT,STUB_W,STUB_WPROTO,
               STUB_GATHER,STUB_TILIZE,STUB_MATMUL; candidate switches, see variant kernels)
  HF_CASE    = focus | l3ds | l3 | l4 | g28 | g28nb | width
Correctness (PCC >= 0.995, rel-RMS <= 0.05) is asserted unless a payload stub is active.
Run: scripts/run_safe_pytest.sh --profile <this file>
"""
import importlib.util
import os
import sys
from pathlib import Path

import ttnn

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parents[5] / "tests/ttnn/unit_tests/operations/conv2d_nhwc"))
from test_conv2d_nhwc_sharded import _sharded_config, _TORCH, pcc  # noqa: E402

ML = ttnn.TensorMemoryLayout


def _load_variant():
    name = "mk_conv2d_variant"
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
    # shape, C_out, k, pad, stride, layout, bias, cfg
    "focus": ((1, 64, 64, 640), 640, 3, 1, 1, ML.BLOCK_SHARDED, True, dict(shard_shape=(410, 64), core_grid=(10, 10))),
    "l3ds": ((16, 28, 28, 256), 256, 3, 1, 2, ML.BLOCK_SHARDED, True, dict(shard_shape=(1280, 32), core_grid=(8, 10))),
    "l3": ((16, 14, 14, 256), 256, 3, 1, 1, ML.BLOCK_SHARDED, True, dict(shard_shape=(314, 32), core_grid=(8, 10))),
    "l4": ((16, 7, 7, 512), 512, 3, 1, 1, ML.BLOCK_SHARDED, True, dict(shard_shape=(96, 64), core_grid=(8, 9))),
    "g28": ((1, 28, 28, 256), 256, 3, 1, 1, ML.BLOCK_SHARDED, True, {}),
    "g28nb": ((1, 28, 28, 256), 256, 3, 1, 1, ML.BLOCK_SHARDED, False, {}),
    "width": ((1, 14, 14, 256), 512, 3, 1, 2, ML.WIDTH_SHARDED, True, {}),
}


def test_hf(device):
    impl = os.environ.get("HF_IMPL", "variant")
    defines = [d for d in os.environ.get("HF_DEFINES", "").split(",") if d]
    case = os.environ.get("HF_CASE", "focus")
    if impl == "op":
        import ttnn.operations.conv2d_nhwc as mod
    else:
        mod = _load_variant()
        pd = sys.modules["mk_conv2d_variant.conv2d_nhwc_program_descriptor"]
        pd.DEFINES = [(d.split("=")[0], d.split("=")[1] if "=" in d else "1") for d in defines]
        for kv in [x for x in os.environ.get("HF_SET", "").split(",") if x]:
            key, val = kv.split("=")
            setattr(pd, key, int(val))
    shape, C_out, k, padding, stride, ml, with_bias, cfg = CASES[case]
    dtype, wdtype = ttnn.bfloat16, ttnn.bfloat8_b
    if case in ("width", "g28"):
        dtype, wdtype = ttnn.float32, ttnn.float32
    import torch

    torch.manual_seed(0)
    x = torch.randn(shape).to(_TORCH[dtype]).float()
    w = torch.randn(C_out, shape[3], k, k).to(_TORCH[wdtype]).float()
    b = torch.randn(C_out).to(_TORCH[wdtype]).float() if with_bias else None
    ref = torch.nn.functional.conv2d(x.permute(0, 3, 1, 2), w, b, stride=stride, padding=padding)
    ref = ref.permute(0, 2, 3, 1).contiguous()
    mc = _sharded_config(ml, shape, device, **cfg)
    tt_x = ttnn.from_torch(x, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mc)
    kw = dict(
        kernel_size=k,
        stride=stride,
        padding=padding,
        input_dtype=dtype,
        weight_dtype=wdtype,
        input_memory_config=tt_x.memory_config(),
        device=device,
    )
    tt_w = mod.prepare_conv2d_weights(w, **kw)
    tt_b = mod.prepare_conv2d_bias(b, **kw) if with_bias else None
    out = mod.conv2d_nhwc(tt_x, tt_w, k, bias=tt_b, padding=padding, stride=stride)
    got = ttnn.to_torch(out).float()
    p = pcc(got, ref)
    rel = ((got - ref).pow(2).mean().sqrt() / ref.std()).item()
    print(
        f"HF impl={impl} case={case} defines={defines} set={os.environ.get('HF_SET', '')} PRECISION pcc={p:.6f} rel_rms={rel:.5f}"
    )
    if not any(d.startswith("STUB") for d in defines):
        assert p >= 0.995 and rel <= 0.05, f"pcc={p} rel_rms={rel}"
