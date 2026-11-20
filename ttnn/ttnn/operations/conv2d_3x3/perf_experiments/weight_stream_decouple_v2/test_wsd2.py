# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Bake-off (Perf 17, `weight_stream_decouple_v2`): Perf 16's weight_stream_decouple ported onto the tree with the
bank-run fill and weights resident, for the shapes whose weights still STREAM ([K, N] does not fit: 128->512).

Re-run (from the repo root): perf_experiments/weight_stream_decouple_v2/run.sh <label> <cases> <variants>
(results: runs/<label>.txt = device ns per case|variant, runs/<label>.jsonl = PCC / sha1 / bit-identity / blocking).

Variants: base = op_base/ (unmodified copy of the working-tree op); the others = op_copy/ with the knobs in VARIANTS
(module constants of op_copy's descriptor, reset before every case). One warm-up call, then WSD_ITERS (default 1)
measured calls signposted `cmp|<case>|<variant>|<i>`. Each variant's output is compared bit-exactly with the first
variant's output of the same case (sha1 of the bf16 bytes, ulp histogram) and its PCC vs torch is recorded (output-row
bands, as the FLUX harness). Precision contract fixed: bf16, HiFi4, fp32 DEST, no approx, bias.

Env: WSD_CASES (comma list of FLUX bh2x4 idx), WSD_VARIANTS (comma list, run in that order), WSD_ORDER=alt (reverse the
variant order on every other case), WSD_LOG (run-log name), WSD_ITERS (measured calls, default 1).
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
_BASE = "ttnn.operations.conv2d_3x3.perf_experiments.weight_stream_decouple_v2."
_SUFFIX = ""
MODS = {}
for _k in ("op_base", "op_copy", "op_final"):
    _p = _BASE + _k + _SUFFIX
    MODS[_k] = (
        importlib.import_module(_p + ".conv2d_3x3"),
        importlib.import_module(_p + ".conv2d_3x3_program_descriptor"),
    )
    assert Path(MODS[_k][1].KERNEL_DIR).parent.name == _k + _SUFFIX, MODS[_k][1].KERNEL_DIR

# FLUX.2 bh2x4 cases (test_flux_conv3x3_compare.py FLUX_CMP_CASES index -> (H, W, C_in, C_out)); padding (0, 0, 1, 1).
FLUX = {
    31: (66, 128, 32, 128),
    32: (130, 256, 32, 128),
    33: (514, 1024, 128, 3),
    34: (66, 128, 128, 512),
    35: (1026, 2048, 128, 3),
    36: (514, 1024, 32, 128),
    37: (258, 512, 64, 256),
    38: (130, 256, 128, 512),
    39: (514, 1024, 64, 128),
    40: (258, 512, 128, 256),
    41: (1026, 2048, 32, 128),
    42: (514, 1024, 64, 256),
    43: (258, 512, 128, 512),
    44: (1026, 2048, 64, 128),
    45: (514, 1024, 128, 256),
    46: (1026, 2048, 64, 256),
    47: (514, 1024, 128, 512),
    # Off-focus regimes (not FLUX indices): other streamed-weight shapes.
    101: (258, 512, 256, 512),
    102: (130, 256, 256, 512),
    103: (514, 1024, 256, 256),
    104: (164, 512, 128, 512),
    105: (218, 512, 128, 512),
    106: (98, 512, 256, 512),
    107: (130, 512, 256, 512),
    # Single-device FLUX.2 ROW_MAJOR guard: enc.conv_in 1024x1024 (C_in 8 lanes), padding 1.
    201: (1024, 1024, 8, 128, (1, 1, 1, 1)),
}
PAD = (0, 0, 1, 1)
BAND_ROWS = 32

KNOB_DEFAULTS = dict(
    WSD_KROT=(),  # K-rotation stagger offsets per channel participant (round-robin); "all" = range(num_k_blocks)
    WSD_SPLIT=False,  # split-phase weight rounds in the store loop
    WSD_SPLIT_TILE=False,  # WSD_SPLIT on TILE input too
    WSD_FILL_LEAD=None,  # staggered first fill: rounds of lead (None = all fill at kernel start)
    WSD_DIAG=False,  # split zones in the sender's round (read / ack wait / multicast)
    WSD_NOREAD=False,  # timing-only ablation: the sender skips its DRAM read (wrong output)
)
VARIANTS = {
    "base": ("op_base", {}),
    # op_final: the integration candidate (integration.patch = op_base -> op_final), recommended knobs as defaults.
    "final": ("op_final", {}),
    "copy": ("op_copy", {}),  # op_copy with every knob off (must equal base)
    # Bit-identical candidate: split-phase rounds only (K order unchanged).
    "sp": ("op_copy", dict(WSD_SPLIT=True)),
    # K rotation alone / with split / Perf 16's recommended (rotation + split + staggered first fill, lead 1).
    "k9": ("op_copy", dict(WSD_KROT="all")),
    "k9sp": ("op_copy", dict(WSD_KROT="all", WSD_SPLIT=True)),
    "rec": ("op_copy", dict(WSD_KROT="all", WSD_SPLIT=True, WSD_FILL_LEAD=1)),
    "recf0": ("op_copy", dict(WSD_KROT="all", WSD_SPLIT=True, WSD_FILL_LEAD=0)),
    "k3sp": ("op_copy", dict(WSD_KROT=(0, 3, 6), WSD_SPLIT=True)),
    "k3csp": ("op_copy", dict(WSD_KROT=(0, 1, 2), WSD_SPLIT=True)),
    "k4csp": ("op_copy", dict(WSD_KROT=(0, 1, 2, 3), WSD_SPLIT=True)),
    "k4sp": ("op_copy", dict(WSD_KROT=(0, 2, 4, 6), WSD_SPLIT=True)),
    "k5sp": ("op_copy", dict(WSD_KROT=(0, 2, 4, 6, 8), WSD_SPLIT=True)),
    "k4csf1": ("op_copy", dict(WSD_KROT=(0, 1, 2, 3), WSD_SPLIT=True, WSD_FILL_LEAD=1)),
    "k5sf1": ("op_copy", dict(WSD_KROT=(0, 2, 4, 6, 8), WSD_SPLIT=True, WSD_FILL_LEAD=1)),
    "k4sf1": ("op_copy", dict(WSD_KROT=(0, 2, 4, 6), WSD_SPLIT=True, WSD_FILL_LEAD=1)),
    # Recommended: offsets by the channel's M-blocks (descriptor "auto": 4 groups, step min(2, M - 1)), lead 1 at step 2.
    "auto": ("op_copy", dict(WSD_KROT="auto", WSD_SPLIT=True, WSD_FILL_LEAD="auto")),
    "auto2": ("op_copy", dict(WSD_KROT="auto2", WSD_SPLIT=True, WSD_FILL_LEAD="auto")),
    "k3esf1": ("op_copy", dict(WSD_KROT=(0, 2, 4), WSD_SPLIT=True, WSD_FILL_LEAD=1)),
    "k2esp": ("op_copy", dict(WSD_KROT=(0, 2), WSD_SPLIT=True)),
    "k2csp": ("op_copy", dict(WSD_KROT=(0, 1), WSD_SPLIT=True)),
    "k3csf1": ("op_copy", dict(WSD_KROT=(0, 1, 2), WSD_SPLIT=True, WSD_FILL_LEAD=1)),
    "recdiag": ("op_copy", dict(WSD_KROT="all", WSD_SPLIT=True, WSD_FILL_LEAD=1, WSD_DIAG=True)),
    "diag": ("op_copy", dict(WSD_DIAG=True)),
    "spdiag": ("op_copy", dict(WSD_SPLIT=True, WSD_DIAG=True)),
    "noread": ("op_copy", dict(WSD_NOREAD=True)),
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
    return [int(c) for c in os.environ.get("WSD_CASES", "43").split(",") if c.strip()]


def _variants():
    return [v for v in os.environ.get("WSD_VARIANTS", "base,copy").split(",") if v]


@pytest.mark.parametrize("device_params", [{"l1_small_size": 65536}], indirect=True)
@pytest.mark.parametrize("idx", _cases())
def test_wsd2(device, idx):
    import torch

    H, W, c_in, c_out = FLUX[idx][:4]
    pad = FLUX[idx][4] if len(FLUX[idx]) > 4 else PAD
    h_out, w_out = H + pad[0] + pad[1] - 2, W + pad[2] + pad[3] - 2
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
    # WSD_REPEAT=n: n passes over the variants, alternating their order (A B, B A, ...); cmp_windows.py takes the median.
    variants = [
        v for r in range(int(os.environ.get("WSD_REPEAT", "1"))) for v in (variants if r % 2 == 0 else variants[::-1])
    ]
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
            padding=pad,
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
                tt_in, tt_w, input_height=H, input_width=W, bias=tt_b, padding=pad, stride=1, compute_kernel_config=cfg
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
            got_raw = ttnn.to_torch(out)
            rec["sha1"] = (
                __import__("hashlib").sha1(got_raw.contiguous().view(torch.int16).numpy().tobytes()).hexdigest()
            )
            got = got_raw.float()
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
            rec["pcc"] = _pcc_bands(got, x, weight, bias, pad, h_out, w_out, c_out)
            if ref_out is None:
                ref_out, rec["ref"] = got, True
            rec["bit_identical_to_first"] = bool(torch.equal(got, ref_out))
            rec["max_abs_vs_first"] = float((got - ref_out).abs().max())
            valid = got.reshape(-1, got.shape[-1])[: h_out * w_out, :c_out]
            refv = ref_out.reshape(-1, ref_out.shape[-1])[: h_out * w_out, :c_out]
            gi = valid.to(torch.bfloat16).view(torch.int16).to(torch.int32)
            ri = refv.to(torch.bfloat16).view(torch.int16).to(torch.int32)
            # bf16 ulp distance on the ordered (sign-magnitude) encoding; sign flips counted apart.
            go = torch.where(gi < 0, -(gi & 0x7FFF), gi & 0x7FFF)
            ro = torch.where(ri < 0, -(ri & 0x7FFF), ri & 0x7FFF)
            ulp = (go - ro).abs()
            same_sign = (gi < 0) == (ri < 0)
            rec["frac_diff_vs_first"] = float((ulp > 0).float().mean())
            rec["n_diff_vs_first"] = int((ulp > 0).sum())
            rec["max_ulp_same_sign_vs_first"] = int(ulp[same_sign].max()) if bool(same_sign.any()) else 0
            rec["n_sign_flip_vs_first"] = int((~same_sign & (ulp > 0)).sum())
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
