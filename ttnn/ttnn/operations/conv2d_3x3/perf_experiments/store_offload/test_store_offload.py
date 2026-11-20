# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""store_offload (Perf 16): who stores the output on the interleaved K-blocked tap-direct walk.

End to end on the FLUX.2 bh2x4 per-device convs (DRAM ROW_MAJOR bf16 input, padding (0, 0, 1, 1), HiFi4, fp32 dest,
bias), on a private copy of the op (proto/), variant chosen by the STORE_OFFLOAD env var at descriptor time:
    base    proto unchanged (the writer stores every row group, weight prefetch between row groups)
    reader  the reader (NCRISC / NoC0) stores every row group, polled from its fill / publish waits
    wserve  the writer stores from inside its weight-round waits, non-blocking
    l1diag  DIAGNOSTIC (wrong output): the writer's store writes go to its own L1, not DRAM
    real    the real op (ttnn.operations.conv2d_3x3), a sanity anchor for base
Every non-diagnostic variant must match the first variant's output bit for bit; the first gets band PCC vs torch.

    SO_CASES=41,47 SO_VARIANTS=base,reader scripts/run_safe_pytest.sh --profile --run-all \\
        ttnn/ttnn/operations/conv2d_3x3/perf_experiments/store_offload/test_store_offload.py
    python3 ttnn/ttnn/operations/conv2d_3x3/perf_experiments/asym_padding/cmp_windows.py <ops_perf_results csv>
"""

import importlib
import json
import os
import sys
import types
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
RESULTS = os.environ.get("SO_RESULTS", "/tmp/perf16/store_offload_results.jsonl")
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
# A variant is "+"-joined tokens (base = none): reader | wserve | l1diag | nofill | noweights; "real" = the real op.
DIAGNOSTIC_TOKENS = {"l1diag", "nofill", "noweights"}


def _tokens(variant):
    return (
        None if variant == "real" else ",".join(t for t in variant.split("+") if t != "base" and not t.startswith("od"))
    )


def _out_depth(variant):
    """odN: cb_output_tiles holds N row groups (the descriptor's OUT_DEPTH_ROW_GROUPS, 2 in the op)."""
    for t in variant.split("+"):
        if t.startswith("od"):
            return int(t[2:])
    return 2


def _diagnostic(variant):
    return bool(DIAGNOSTIC_TOKENS & set(variant.split("+")))


_SEL = [int(i) for i in os.environ.get("SO_CASES", "41").split(",") if i.strip()]
_VARIANTS = [v for v in os.environ.get("SO_VARIANTS", "base,reader").split(",") if v]


def _proto():
    name = "so_proto"
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
def test_store_offload(device, idx):
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
    label = f"so{idx}"
    record = {"idx": idx, "shape": [H, W, c_in, c_out], "variants": {}}
    ref_out = None
    ref_name = None
    for variant in _VARIANTS:
        tokens = _tokens(variant)
        fn = real_conv if tokens is None else proto.conv2d_3x3
        os.environ["STORE_OFFLOAD"] = tokens or ""
        desc = sys.modules["so_proto.conv2d_3x3_program_descriptor"]
        if desc.OUT_DEPTH_ROW_GROUPS != _out_depth(variant):
            desc.OUT_DEPTH_ROW_GROUPS = _out_depth(variant)
            desc.select_blocking.cache_clear()

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
        os.environ["STORE_OFFLOAD"] = ""
        if collect:
            ttnn.deallocate(out)
            continue
        got = ttnn.to_torch(out)
        ttnn.deallocate(out)
        entry = {}
        if ref_out is None and not _diagnostic(variant):
            ref_out, ref_name = got, variant
            entry["pcc_vs_torch"] = _pcc_bands(got, x, weight, bias, h_out, w_out, c_out)
        elif ref_out is not None:
            diff = (got.float() - ref_out.float()).abs()
            entry["bit_identical_to"] = ref_name
            entry["n_diff"] = int((got != ref_out).sum().item())
            entry["max_abs_diff"] = float(diff.max().item())
        record["variants"][variant] = entry
        print(f"STORE_OFFLOAD {label} {variant}: {entry}")
    ttnn.deallocate(tt_in)
    if collect:
        return
    Path(RESULTS).parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS, "a") as f:
        f.write(json.dumps(record) + "\n")
    for variant, entry in record["variants"].items():
        if not _diagnostic(variant) and "n_diff" in entry:
            assert entry["n_diff"] == 0, f"{variant} differs from {ref_name}: {entry}"
