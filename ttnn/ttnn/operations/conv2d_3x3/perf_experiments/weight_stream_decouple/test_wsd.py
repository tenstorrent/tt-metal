# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Bake-off: decouple the R1 weight multicast stream from the per-round lock-step (perf experiment, round 16,
`weight_stream_decouple`).

Re-run (from the repo root, after `export TT_METAL_HOME=$PWD PYTHONPATH=$PWD`):
    ttnn/ttnn/operations/conv2d_3x3/perf_experiments/weight_stream_decouple/run.sh <label> 43,47 base,rec
(results: runs/<label>.txt = device ns per case|variant, runs/<label>.jsonl = PCC / bit-identity / blocking;
rounds.py and blocks.py read runs/<label>_dev.csv.gz for per-round / per-M-block timelines). Panoptic guard:
test_panoptic_guard.py (base vs rec, bit-exact on every non-FLUX LOOSE case).

Recommended variant `rec` (op_copy knobs): WSD_KROT="all" (channel participant i walks every M-block's K-blocks from
offset i % num_k_blocks; leading / trailing rounds outside its own 9 x M are popped unused), WSD_SPLIT (the writer's store
loop acks / completes weight rounds without blocking and polls the output CB instead of cb_wait_front), WSD_FILL_LEAD=1
(a core with offset o issues its first slab fill once round o - 1 is in).

Variants (one dispatch per measured call, warm-up first):
  base   op_base/ (unmodified copy of the working-tree op at HEAD 5ac29b03f50)
  <name> op_copy/ with the knobs in VARIANTS (module constants of op_copy's descriptor, reset before every case)

Env: WSD_CASES (comma list of FLUX bh2x4 idx: 41,43,45,47,42,...), WSD_VARIANTS (comma list, run in that order),
WSD_ORDER=alt (reverse the variant order on every other case), WSD_LOG (run-log name), WSD_ITERS (measured calls,
default 1). Each variant's calls are signposted `cmp|<case>|<variant>|<i>` (cmp_windows.py sums them). Every
variant's output is compared bit-exactly with the base output of the same case and its PCC vs torch is recorded
(output-row bands, as the FLUX harness). Precision contract fixed: bf16, HiFi4, fp32 DEST, no approx, bias.
"""

import importlib
import json
import os
import time
from pathlib import Path

import pytest
import ttnn
from tracy import signpost

HERE = Path(__file__).parent
_BASE = "ttnn.operations.conv2d_3x3.perf_experiments.weight_stream_decouple."
_SUFFIX = "_nz" if os.environ.get("WSD_NZ") == "1" else ""
MODS = {}
for _k in ("op_base", "op_copy"):
    _p = _BASE + _k + _SUFFIX
    MODS[_k] = (
        importlib.import_module(_p + ".conv2d_3x3"),
        importlib.import_module(_p + ".conv2d_3x3_program_descriptor"),
    )
    assert Path(MODS[_k][1].KERNEL_DIR).parent.name == _k + _SUFFIX, MODS[_k][1].KERNEL_DIR

# FLUX.2 bh2x4 cases (test_flux_conv3x3_compare.py FLUX_CMP_CASES index -> (H, W, C_in, C_out)); padding (0, 0, 1, 1).
FLUX = {
    35: (1026, 2048, 128, 3),
    36: (514, 1024, 32, 128),
    41: (1026, 2048, 32, 128),
    42: (514, 1024, 64, 256),
    43: (258, 512, 128, 512),
    44: (1026, 2048, 64, 128),
    45: (514, 1024, 128, 256),
    46: (1026, 2048, 64, 256),
    47: (514, 1024, 128, 512),
    # 1x1-group-like extra regimes (not FLUX indices): the same shapes on other channel counts.
    101: (130, 256, 128, 512),
    102: (258, 512, 256, 512),
}
PAD = (0, 0, 1, 1)
BAND_ROWS = 32

KNOB_DEFAULTS = dict(
    WEIGHT_DEPTH=2,
    WSD_BK=None,  # force block_k_tiles (None: the solve)
    WSD_BM=None,  # force block_m_tiles (None: the solve)
    WSD_DIAG=False,  # split zones in the sender's round (ack wait / multicast)
    WSD_ROT=0,  # rotating-sender flag stream: number of senders (0 = off)
    WSD_ROT_LOOKAHEAD=None,  # rotating stream: rounds in flight (None: depth - 1)
    WSD_NOACK=False,  # flag-per-slot stream with per-slot reuse counters (single sender)
    WSD_EARLY_ACK=False,  # baseline protocol, but the receiver acks round k + 1 as soon as its slot frees
    WSD_KROT=(),  # K-rotation stagger offsets per channel participant (round-robin)
    WSD_SPLIT=False,  # split-phase weight rounds in the store loop
    WSD_FILL_LEAD=None,  # staggered first fill: rounds of lead (None = all fill at kernel start)
    WSD_NOREAD=False,  # timing-only ablation: the sender skips its DRAM read (wrong output)
)
VARIANTS = {
    "base": ("op_base", {}),
    "copy": ("op_copy", {}),
    "diag": ("op_copy", dict(WSD_DIAG=True)),
    "d3": ("op_copy", dict(WEIGHT_DEPTH=3)),
    "d4": ("op_copy", dict(WEIGHT_DEPTH=4)),
    "bk12": ("op_copy", dict(WSD_BK=12)),
    "rot": ("op_copy", dict(WSD_ROT=1)),
    "k9": ("op_copy", dict(WSD_KROT=tuple(range(9)))),
    "k5": ("op_copy", dict(WSD_KROT=(0, 2, 4, 6, 8))),
    "k3": ("op_copy", dict(WSD_KROT=(0, 3, 6))),
    "k3s": ("op_copy", dict(WSD_KROT=(0, 1, 2))),
    "k2": ("op_copy", dict(WSD_KROT=(0, 4))),
    "k9d3": ("op_copy", dict(WSD_KROT=tuple(range(9)), WEIGHT_DEPTH=3)),
    "k9d4": ("op_copy", dict(WSD_KROT=tuple(range(9)), WEIGHT_DEPTH=4)),
    "k3d3": ("op_copy", dict(WSD_KROT=(0, 3, 6), WEIGHT_DEPTH=3)),
    "sp": ("op_copy", dict(WSD_SPLIT=True)),
    "k9sp": ("op_copy", dict(WSD_KROT=tuple(range(9)), WSD_SPLIT=True)),
    "k9d3sp": ("op_copy", dict(WSD_KROT=tuple(range(9)), WEIGHT_DEPTH=3, WSD_SPLIT=True)),
    "d3sp": ("op_copy", dict(WEIGHT_DEPTH=3, WSD_SPLIT=True)),
    "k5csp": ("op_copy", dict(WSD_KROT=(0, 1, 2, 3, 4), WSD_SPLIT=True)),
    "k3csp": ("op_copy", dict(WSD_KROT=(0, 1, 2), WSD_SPLIT=True)),
    "k4sp": ("op_copy", dict(WSD_KROT=(0, 2, 4, 6), WSD_SPLIT=True)),
    "k9spf1": ("op_copy", dict(WSD_KROT=tuple(range(9)), WSD_SPLIT=True, WSD_FILL_LEAD=1)),
    "k9spf2": ("op_copy", dict(WSD_KROT=tuple(range(9)), WSD_SPLIT=True, WSD_FILL_LEAD=2)),
    "k9spf3": ("op_copy", dict(WSD_KROT=tuple(range(9)), WSD_SPLIT=True, WSD_FILL_LEAD=3)),
    # Recommended: one phase group per K-block, split-phase store-loop rounds, staggered first fill (lead 1).
    "rec": ("op_copy", dict(WSD_KROT="all", WSD_SPLIT=True, WSD_FILL_LEAD=1)),
    "recf0": ("op_copy", dict(WSD_KROT="all", WSD_SPLIT=True, WSD_FILL_LEAD=0)),
    "recd3": ("op_copy", dict(WSD_KROT="all", WSD_SPLIT=True, WSD_FILL_LEAD=1, WEIGHT_DEPTH=3)),
    "noread": ("op_copy", dict(WSD_NOREAD=True)),
    "recnoread": ("op_copy", dict(WSD_KROT="all", WSD_SPLIT=True, WSD_FILL_LEAD=1, WSD_NOREAD=True)),
    "recdiag": ("op_copy", dict(WSD_KROT="all", WSD_SPLIT=True, WSD_FILL_LEAD=1, WSD_DIAG=True)),
    "k9spdiag": ("op_copy", dict(WSD_KROT=tuple(range(9)), WSD_SPLIT=True, WSD_DIAG=True)),
    "k9d3spdiag": ("op_copy", dict(WSD_KROT=tuple(range(9)), WSD_SPLIT=True, WSD_DIAG=True, WEIGHT_DEPTH=3)),
}


def _bands(h_out):
    if h_out <= 3 * BAND_ROWS:
        return [(0, h_out)]
    mid = h_out // 2 - BAND_ROWS // 2
    return [(0, BAND_ROWS), (mid, mid + BAND_ROWS), (h_out - BAND_ROWS, h_out)]


def _ref_rows(x_hwc, weight, bias, pad, r0, r1):
    import torch

    top, bottom, left, right = pad
    h = x_hwc.shape[0]
    first = r0 - top
    last = (r1 - 1) - top + 3
    rows = x_hwc[max(first, 0) : min(last, h)].float().permute(2, 0, 1).unsqueeze(0)
    rows = torch.nn.functional.pad(rows, (left, right, max(0, -first), max(0, last - h)))
    return torch.nn.functional.conv2d(rows, weight.float(), bias.float())[0].permute(1, 2, 0)


def _pcc_bands(out, x_hwc, weight, bias, pad, h_out, w_out, c_out):
    import torch

    out = out.reshape(-1, out.shape[-1])[: h_out * w_out, :c_out].reshape(h_out, w_out, c_out)
    got, ref = [], []
    for r0, r1 in _bands(h_out):
        got.append(out[r0:r1].double().flatten())
        ref.append(_ref_rows(x_hwc, weight, bias, pad, r0, r1).double().flatten())
    got, ref = torch.cat(got), torch.cat(ref)
    return torch.corrcoef(torch.stack([got, ref]))[0, 1].item()


def _set_knobs(knobs):
    _, pd = MODS["op_copy"]
    for k, v in {**KNOB_DEFAULTS, **knobs}.items():
        setattr(pd, k, v)
    for name in dir(pd):
        f = getattr(pd, name)
        if hasattr(f, "cache_clear"):
            f.cache_clear()


def _in_collect_pass():
    import sys

    plugin = sys.modules.get("tests.plugins.up_front_collect")
    return plugin is not None and getattr(plugin, "_PASS", "real") == "collect"


def _cases():
    return [int(c) for c in os.environ.get("WSD_CASES", "43,47").split(",") if c.strip()]


def _variants():
    return [v for v in os.environ.get("WSD_VARIANTS", "base,copy").split(",") if v]


@pytest.mark.parametrize("device_params", [{"l1_small_size": 65536}], indirect=True)
@pytest.mark.parametrize("idx", _cases())
def test_wsd(device, idx):
    import torch

    H, W, c_in, c_out = FLUX[idx]
    h_out, w_out = H - 2, W
    torch.manual_seed(0)
    x = torch.randn(H, W, c_in, dtype=torch.bfloat16)
    weight = (torch.randn(c_out, c_in, 3, 3) / (9 * c_in) ** 0.5).to(torch.bfloat16)
    bias = (0.1 * torch.randn(c_out)).to(torch.bfloat16)
    tt_in = ttnn.from_torch(
        x.reshape(1, 1, H * W, c_in),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False
    )
    variants = _variants()
    log = os.environ.get("WSD_LOG", "wsd")
    if os.environ.get("WSD_ORDER") == "alt" and _cases().index(idx) % 2 == 1:
        variants = variants[::-1]
    iters = int(os.environ.get("WSD_ITERS", "1"))
    ref_out = None
    base_variant_out = {}
    for vname in variants:
        pkg, knobs = VARIANTS[vname]
        op, pd = MODS[pkg]
        _set_knobs(knobs if pkg == "op_copy" else {})
        device.clear_program_cache()
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
        tt_w = op.prepare_conv2d_weights(weight, **prep)
        tt_b = op.prepare_conv2d_bias(bias, **prep)

        def run():
            return op.conv2d_3x3(
                tt_in, tt_w, input_height=H, input_width=W, bias=tt_b, padding=PAD, stride=1, compute_kernel_config=cfg
            )

        rec = dict(case=idx, variant=vname, knobs={k: v for k, v in knobs.items()}, log=log, t=time.time())
        try:
            out = run()
            ttnn.synchronize_device(device)
            ttnn.ReadDeviceProfiler(device)
            for i in range(iters):
                ttnn.deallocate(out)
                signpost(header=f"cmp|{idx}|{vname}|{i}")
                out = run()
                ttnn.synchronize_device(device)
                ttnn.ReadDeviceProfiler(device)
            signpost(header=f"cmpend|{idx}|{vname}")
            if _in_collect_pass():
                ttnn.deallocate(out)
                continue
            got = ttnn.to_torch(out).float()
            ttnn.deallocate(out)
            rec["status"] = "ok"
            b = getattr(op, "LAST_BLOCKING", None)
            if b is not None:
                rec["blocking"] = dict(
                    bm=b.block_m_tiles,
                    bk=b.block_k_tiles,
                    depth=b.weight_depth,
                    m_blocks=b.max_m_blocks_per_core,
                    k_blocks=b.num_k_blocks,
                    footprint=b.footprint_bytes,
                )
            rec["pcc"] = _pcc_bands(got, x, weight, bias, PAD, h_out, w_out, c_out)
            if ref_out is None:
                ref_out, rec["ref"] = got, True
            rec["bit_identical_to_first"] = bool(torch.equal(got, ref_out))
            rec["max_abs_vs_first"] = float((got - ref_out).abs().max())
            valid = got.reshape(-1, got.shape[-1])[: h_out * w_out, :c_out]
            refv = ref_out.reshape(-1, ref_out.shape[-1])[: h_out * w_out, :c_out]
            gi = valid.to(torch.bfloat16).view(torch.int16).to(torch.int32)
            ri = refv.to(torch.bfloat16).view(torch.int16).to(torch.int32)
            ulp = (gi - ri).abs()
            rec["frac_diff_vs_first"] = float((ulp > 0).float().mean())
            rec["max_ulp_vs_first"] = int(ulp.max())
        except Exception as e:
            rec["status"] = "error"
            rec["error"] = str(e).splitlines()[0][:400] if str(e) else type(e).__name__
        ttnn.deallocate(tt_w)
        ttnn.deallocate(tt_b)
        print("WSD:", json.dumps(rec))
        with open(HERE / "runs" / f"{log}.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
    _set_knobs({})
    ttnn.deallocate(tt_in)
