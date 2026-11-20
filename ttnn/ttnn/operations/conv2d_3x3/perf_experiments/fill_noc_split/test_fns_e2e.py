# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""fill_noc_split (Perf 16) end to end: the interleaved ROW_MAJOR fill read over both NoCs on the K-blocked tap-direct
walk, on a private copy of the op (proto/), against the same copy without it.

FLUX.2 bh2x4 per-device convs (DRAM ROW_MAJOR bf16 input, padding (0, 0, 1, 1), HiFi4, fp32 dest, bias, TILE bf16 DRAM
output). Variant per FNS_VARIANTS (comma list), chosen by the FILL_NOC_SPLIT env var at descriptor time:
    base   proto with FILL_NOC_SPLIT=0 (= the op as of this round)
    split  proto with FILL_NOC_SPLIT=1 (reader NoC0 + writer NoC1, per-band claims, kernels/conv2d_3x3_fill_split.hpp)
    static proto with FILL_NOC_SPLIT=2 (same, but a fixed half of every band per RISC: FNS_STATIC)
    real   the real op (ttnn.operations.conv2d_3x3), a sanity anchor for base
Every variant after the first must match the first bit for bit; the first gets band PCC vs torch.

    FNS_CASES=41 FNS_VARIANTS=base,split scripts/run_safe_pytest.sh --profile --run-all \\
        ttnn/ttnn/operations/conv2d_3x3/perf_experiments/fill_noc_split/test_fns_e2e.py
    python3 ttnn/ttnn/operations/conv2d_3x3/perf_experiments/asym_padding/cmp_windows.py <ops_perf_results csv>
(or run_e2e.sh, which does both).
"""

import importlib
import json
import os
import sys
import types
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
RESULTS = os.environ.get("FNS_RESULTS", str(HERE / "runs" / "e2e_results.jsonl"))
MEASURED_ITERS = 3

# FLUX_CMP_CASES index -> (H incl. halo rows, W, C_in, C_out); padding (0, 0, 1, 1), stride 1.
CASES = {
    35: (1026, 2048, 128, 3),
    36: (514, 1024, 32, 128),
    41: (1026, 2048, 32, 128),
    42: (514, 1024, 64, 256),
    43: (258, 512, 128, 512),
    44: (1026, 2048, 64, 128),
    46: (1026, 2048, 64, 256),
    47: (514, 1024, 128, 512),
}
PAD = (0, 0, 1, 1)
_SEL = [int(i) for i in os.environ.get("FNS_CASES", "41").split(",") if i.strip()]
_VARIANTS = [v for v in os.environ.get("FNS_VARIANTS", "base,split").split(",") if v]
_ENV = {"base": "0", "split": "1", "static": "2"}


def _proto():
    name = "fns_proto"
    if name not in sys.modules:
        pkg = types.ModuleType(name)
        pkg.__path__ = [str(HERE / "proto")]
        sys.modules[name] = pkg
    return importlib.import_module(f"{name}.conv2d_3x3")


def _in_collect_pass():
    plugin = sys.modules.get("tests.plugins.up_front_collect")
    return plugin is not None and getattr(plugin, "_PASS", "real") == "collect"


def _ref_rows(x_hwc, weight, bias, r0, r1):
    import torch

    top, bottom, left, right = PAD
    h = x_hwc.shape[0]
    first, last = r0 - top, r1 - 1 - top + 3
    rows = x_hwc[max(first, 0) : min(last, h)].float().permute(2, 0, 1).unsqueeze(0)
    rows = torch.nn.functional.pad(rows, (left, right, max(0, -first), max(0, last - h)))
    return torch.nn.functional.conv2d(rows, weight.float(), bias.float())[0].permute(1, 2, 0)


def _pcc_bands(out_flat, x_hwc, weight, bias, h_out, w_out, c_out):
    import torch

    out = out_flat.reshape(-1, out_flat.shape[-1])[: h_out * w_out, :c_out].reshape(h_out, w_out, c_out)
    bands = [(0, 32), (h_out // 2 - 16, h_out // 2 + 16), (h_out - 32, h_out)]
    got = torch.cat([out[a:b].double().flatten() for a, b in bands])
    ref = torch.cat([_ref_rows(x_hwc, weight, bias, a, b).double().flatten() for a, b in bands])
    return torch.corrcoef(torch.stack([got, ref]))[0, 1].item()


@pytest.mark.parametrize("device_params", [{"l1_small_size": 65536}], indirect=True)
@pytest.mark.parametrize("idx", _SEL)
def test_fns_e2e(device, idx):
    import torch
    import ttnn
    from tracy import signpost
    from ttnn.operations.conv2d_3x3 import conv2d_3x3 as real_conv, prepare_conv2d_bias, prepare_conv2d_weights

    proto = _proto()
    H, W, c_in, c_out = CASES[idx]
    h_out, w_out = H - 2, W
    collect = _in_collect_pass()
    torch.manual_seed(0)
    x = torch.zeros(H, W, c_in, dtype=torch.bfloat16) if collect else torch.randn(H, W, c_in, dtype=torch.bfloat16)
    weight = (torch.randn(c_out, c_in, 3, 3) / (9 * c_in) ** 0.5).to(torch.bfloat16)
    bias = (0.1 * torch.randn(c_out)).to(torch.bfloat16)
    tt_in = ttnn.from_torch(
        x.reshape(1, 1, H * W, c_in),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    prep = dict(
        stride=1,
        padding=PAD,
        dilation=1,
        input_dtype=ttnn.bfloat16,
        input_layout=ttnn.ROW_MAJOR_LAYOUT,
        weight_dtype=ttnn.bfloat16,
        input_memory_config=tt_in.memory_config(),
        device=device,
    )
    tt_w = prepare_conv2d_weights(weight, **prep)
    tt_b = prepare_conv2d_bias(bias, **prep)
    cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False
    )
    label = f"fns{idx}"
    record = {"idx": idx, "shape": [H, W, c_in, c_out], "variants": {}}
    ref_out = None
    ref_name = None
    for variant in _VARIANTS:
        fn = real_conv if variant == "real" else proto.conv2d_3x3
        os.environ["FILL_NOC_SPLIT"] = _ENV.get(variant, "0")

        def run():
            return fn(
                tt_in,
                tt_w,
                input_height=H,
                input_width=W,
                bias=tt_b,
                padding=PAD,
                stride=1,
                compute_kernel_config=cfg,
            )

        out = run()
        ttnn.synchronize_device(device)
        ttnn.ReadDeviceProfiler(device)
        for i in range(MEASURED_ITERS):
            ttnn.deallocate(out)
            signpost(header=f"cmp|{label}|{variant}|{i}")
            out = run()
            ttnn.synchronize_device(device)
            ttnn.ReadDeviceProfiler(device)
        signpost(header=f"cmpend|{label}|{variant}")
        os.environ["FILL_NOC_SPLIT"] = "0"
        if collect:
            ttnn.deallocate(out)
            continue
        got = ttnn.to_torch(out)
        ttnn.deallocate(out)
        entry = {}
        if ref_out is None:
            ref_out, ref_name = got, variant
            entry["pcc_vs_torch"] = _pcc_bands(got, x, weight, bias, h_out, w_out, c_out)
        else:
            diff = (got.float() - ref_out.float()).abs()
            entry["bit_identical_to"] = ref_name
            entry["n_diff"] = int((got != ref_out).sum().item())
            entry["max_abs_diff"] = float(diff.max().item())
            entry["pcc_vs_torch"] = _pcc_bands(got, x, weight, bias, h_out, w_out, c_out)
        record["variants"][variant] = entry
        print(f"FILL_NOC_SPLIT {label} {variant}: {entry}")
    ttnn.deallocate(tt_in)
    if collect:
        return
    Path(RESULTS).parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS, "a") as f:
        f.write(json.dumps(record) + "\n")
    for variant, entry in record["variants"].items():
        if "n_diff" in entry:
            assert entry["n_diff"] == 0, f"{variant} differs from {ref_name}: {entry}"
