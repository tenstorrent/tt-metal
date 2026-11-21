# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Bake-off: the per-unit / per-K-block cost of the K-blocked tap-direct walk (Perf 18, `tapdirect_unit_cost`).

Re-run (from the repo root):
    ttnn/ttnn/operations/conv2d_3x3/perf_experiments/tapdirect_unit_cost/run.sh <label> 42,41 base,c.GRP
(results: runs/<label>.txt = device ns per case|variant, runs/<label>.jsonl = PCC / bit-identity / ulp vs the first
variant of the case).

Cases: a FLUX.2 bh2x4 idx (test_flux_conv3x3_compare.py FLUX_CMP_CASES) with optional suffixes
  n  no bias      r  ReLU      t  TILE input (else ROW_MAJOR)
e.g. 42, 42n, 42r, 42nr, 41t. Padding (0, 0, 1, 1), bf16 activations + weights, HiFi4, fp32 DEST, no approx.

Variants: base = op_base/ (the working-tree op at HEAD d4414e00bd0 + uncommitted user changes, ablation hooks),
c.<TOK>.<TOK> = op_copy/ with defines TU_<TOK> (op_copy descriptor TU_DEFS, reset before every case); "@a+b" suffix = CONV2D_3X3_ABLATE=a,b.
Each measured call is signposted `cmp|<case>|<variant>|<i>` (cmp_windows.py sums the window).
Env: BE_CASES, BE_VARIANTS, BE_LOG, BE_ITERS (default 1), BE_ORDER=alt (reverse the variant order on odd cases).
"""

import hashlib
import importlib
import json
import os
import time
from pathlib import Path

import pytest
import ttnn
from tracy import signpost

HERE = Path(__file__).parent
_BASE = "ttnn.operations.conv2d_3x3.perf_experiments.tapdirect_unit_cost."
MODS = {}
for _k in [d for d in ("op_base", "op_copy", "op_final") if (HERE / d).is_dir()]:
    MODS[_k] = (
        importlib.import_module(_BASE + _k + ".conv2d_3x3"),
        importlib.import_module(_BASE + _k + ".conv2d_3x3_program_descriptor"),
    )
    assert Path(MODS[_k][1].KERNEL_DIR).parent.name == _k, MODS[_k][1].KERNEL_DIR

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
}
PAD = (0, 0, 1, 1)
# Single-device FLUX.2 convs (test_flux_conv3x3_compare.py group "1x1"), always TILE input: name -> (H, W, C_in, C_out,
# stride, (top, bottom, left, right)). The d* ones are the encoder's stride-2 downsamples (strided-window / S2 taps).
EXTRA = {
    "u512_256": (512, 512, 256, 256, 1, (1, 1, 1, 1)),
    "u1024_128": (1024, 1024, 128, 128, 1, (1, 1, 1, 1)),
    "d512_256s2": (512, 512, 256, 256, 2, (0, 1, 0, 1)),
    "d1024_128s2": (1024, 1024, 128, 128, 2, (0, 1, 0, 1)),
    # ROW_MAJOR, C_in 16: face-direct im2col, one K-block; W_out % 32 != 0 keeps it off slab-direct, so the K-blocked
    # walk runs its one-K-block case.
    "fd16": (32, 48, 16, 64, 1, (1, 1, 1, 1)),
    # ROW_MAJOR stride 2 (strided-window tilize).
    "s2rm": (64, 64, 64, 64, 2, (0, 1, 0, 1)),
}
EXTRA_RM = {"fd16", "s2rm", "fdk1"}
# ROW_MAJOR C_in 8 stride 2 (stem.conv1's face-direct geometry) with W_out % 32 != 0: off slab-direct, one K-block.
EXTRA["fdk1"] = (64, 96, 8, 64, 2, (1, 1, 1, 1))
BAND_ROWS = 32

KNOB_DEFAULTS = dict(
    TU_DEFS="",  # comma list of op_copy kernel tokens: each becomes the define TU_<token> on reader, writer, compute
)
VARIANTS = {
    "base": ("op_base", {}),
    "rec": ("op_final", {}),  # the recommended change on the unmodified working-tree op (integration.patch)
    "copy": ("op_copy", {}),
}


def _variant(vbase):
    """base / rec / copy, or c.<TOK>.<TOK>...: op_copy with TU_DEFS = those tokens."""
    if vbase in VARIANTS:
        return VARIANTS[vbase]
    assert vbase.startswith("c."), vbase
    return "op_copy", dict(TU_DEFS=",".join(vbase.split(".")[1:]))


def _parse_case(spec):
    name, _, flags = spec.partition(".")
    if name in EXTRA:
        return name, "n" not in flags, "r" in flags, name not in EXTRA_RM
    i = 0
    while i < len(spec) and spec[i].isdigit():
        i += 1
    idx, flags = int(spec[:i]), spec[i:]
    return idx, "n" not in flags, "r" in flags, "t" in flags


def _bands(h_out):
    if h_out <= 3 * BAND_ROWS:
        return [(0, h_out)]
    mid = h_out // 2 - BAND_ROWS // 2
    return [(0, BAND_ROWS), (mid, mid + BAND_ROWS), (h_out - BAND_ROWS, h_out)]


def _ref_rows(x_hwc, weight, bias, pad, r0, r1, relu):
    import torch

    top, bottom, left, right = pad
    h = x_hwc.shape[0]
    first = r0 - top
    last = (r1 - 1) - top + 3
    rows = x_hwc[max(first, 0) : min(last, h)].float().permute(2, 0, 1).unsqueeze(0)
    rows = torch.nn.functional.pad(rows, (left, right, max(0, -first), max(0, last - h)))
    out = torch.nn.functional.conv2d(rows, weight.float(), None if bias is None else bias.float())[0].permute(1, 2, 0)
    return torch.relu(out) if relu else out


def _pcc_bands(out, x_hwc, weight, bias, pad, h_out, w_out, c_out, relu):
    import torch

    out = out.reshape(-1, out.shape[-1])[: h_out * w_out, :c_out].reshape(h_out, w_out, c_out)
    got, ref = [], []
    for r0, r1 in _bands(h_out):
        got.append(out[r0:r1].double().flatten())
        ref.append(_ref_rows(x_hwc, weight, bias, pad, r0, r1, relu).double().flatten())
    got, ref = torch.cat(got), torch.cat(ref)
    return torch.corrcoef(torch.stack([got, ref]))[0, 1].item()


def _err_bands(out, x_hwc, weight, bias, pad, h_out, w_out, c_out, relu):
    import torch

    out = out.reshape(-1, out.shape[-1])[: h_out * w_out, :c_out].reshape(h_out, w_out, c_out)
    errs = []
    for r0, r1 in _bands(h_out):
        errs.append((out[r0:r1].double() - _ref_rows(x_hwc, weight, bias, pad, r0, r1, relu).double()).abs().flatten())
    e = torch.cat(errs)
    return float(e.mean()), float(e.max())


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
    return [c.strip() for c in os.environ.get("BE_CASES", "42").split(",") if c.strip()]


def _variants():
    return [v for v in os.environ.get("BE_VARIANTS", "base,copy").split(",") if v]


@pytest.mark.parametrize("device_params", [{"l1_small_size": 65536}], indirect=True)
@pytest.mark.parametrize("case", _cases())
def test_tu(device, case):
    import torch

    idx, has_bias, relu, tile = _parse_case(case)
    if idx in EXTRA:
        H, W, c_in, c_out, stride, pad = EXTRA[idx]
        h_out = (H + pad[0] + pad[1] - 3) // stride + 1
        w_out = (W + pad[2] + pad[3] - 3) // stride + 1
    else:
        H, W, c_in, c_out = FLUX[idx]
        stride, pad = 1, PAD
        h_out, w_out = H - 2, W
    torch.manual_seed(0)
    x = torch.randn(H, W, c_in, dtype=torch.bfloat16)
    weight = (torch.randn(c_out, c_in, 3, 3) / (9 * c_in) ** 0.5).to(torch.bfloat16)
    bias = (0.1 * torch.randn(c_out)).to(torch.bfloat16) if has_bias else None
    layout = ttnn.TILE_LAYOUT if tile else ttnn.ROW_MAJOR_LAYOUT
    tt_in = ttnn.from_torch(
        x.reshape(1, 1, H * W, c_in),
        dtype=ttnn.bfloat16,
        layout=layout,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False
    )
    variants = _variants()
    log = os.environ.get("BE_LOG", "tu")
    if os.environ.get("BE_ORDER") == "alt" and _cases().index(case) % 2 == 1:
        variants = variants[::-1]
    iters = int(os.environ.get("BE_ITERS", "1"))
    ref_out = None
    ref_cache = {}
    for vname in variants:
        # "<variant>@<ablate tokens joined by +>": CONV2D_3X3_ABLATE for this variant only (e.g. base@fill).
        vbase, _, abl = vname.partition("@")
        pkg, knobs = _variant(vbase)
        if abl:
            os.environ["CONV2D_3X3_ABLATE"] = abl.replace("+", ",")
        else:
            os.environ.pop("CONV2D_3X3_ABLATE", None)
        op, pd = MODS[pkg]
        _set_knobs(knobs if pkg == "op_copy" else {})
        device.clear_program_cache()
        prep = dict(
            stride=stride,
            padding=pad,
            dilation=1,
            input_dtype=ttnn.bfloat16,
            input_layout=layout,
            weight_dtype=ttnn.bfloat16,
            input_memory_config=tt_in.memory_config(),
            device=device,
        )
        tt_w = op.prepare_conv2d_weights(weight, **prep)
        tt_b = op.prepare_conv2d_bias(bias, **prep) if has_bias else None

        def run():
            return op.conv2d_3x3(
                tt_in,
                tt_w,
                input_height=H,
                input_width=W,
                bias=tt_b,
                padding=pad,
                stride=stride,
                activation="relu" if relu else None,
                compute_kernel_config=cfg,
            )

        rec = dict(
            case=case,
            variant=vname,
            knobs=dict(knobs),
            ablate=abl,
            log=log,
            t=time.time(),
        )
        try:
            out = run()
            ttnn.synchronize_device(device)
            ttnn.ReadDeviceProfiler(device)
            for i in range(iters):
                ttnn.deallocate(out)
                signpost(header=f"cmp|{case}|{vname}|{i}")
                out = run()
                ttnn.synchronize_device(device)
                ttnn.ReadDeviceProfiler(device)
            signpost(header=f"cmpend|{case}|{vname}")
            if _in_collect_pass():
                ttnn.deallocate(out)
                continue
            got = ttnn.to_torch(out).float()
            ttnn.deallocate(out)
            rec["status"] = "ok"
            gb = getattr(pd, "LAST_GB", None)
            if gb is not None:
                b = gb[1]
                rec["blocking"] = dict(
                    bm=b.block_m_tiles,
                    bk=b.block_k_tiles,
                    bn=b.block_n_tiles,
                    kb=b.num_k_blocks,
                    mb=b.max_m_blocks_per_core,
                    sb=(b.out_subblock_h, b.out_subblock_w),
                    res=b.weights_resident,
                    nslice=b.nslice,
                    fp=b.footprint_bytes,
                    face_direct=gb[0].face_direct_capable,
                    in0_stream_rows=b.in0_stream_rows,
                )
            valid = got.reshape(-1, got.shape[-1])[: h_out * w_out, :c_out]
            rec["sha1"] = hashlib.sha1(
                valid.to(torch.bfloat16).contiguous().view(torch.int16).numpy().tobytes()
            ).hexdigest()
            if not rec["ablate"] and idx in EXTRA:
                if "ref" not in ref_cache:
                    top, bottom, left, right = pad
                    xin = torch.nn.functional.pad(x.float().permute(2, 0, 1).unsqueeze(0), (left, right, top, bottom))
                    ref = torch.nn.functional.conv2d(
                        xin, weight.float(), bias.float() if has_bias else None, stride=stride
                    )
                    ref_cache["ref"] = ref[0].permute(1, 2, 0).reshape(-1, c_out).double()
                ref = ref_cache["ref"]
                if relu:
                    ref = torch.relu(ref)
                g2 = valid.double()
                rec["pcc"] = torch.corrcoef(torch.stack([g2.flatten(), ref.flatten()]))[0, 1].item()
                e = (g2 - ref).abs()
                rec["mean_abs_err"], rec["max_abs_err"] = float(e.mean()), float(e.max())
            elif not rec["ablate"]:
                rec["pcc"] = _pcc_bands(got, x, weight, bias, PAD, h_out, w_out, c_out, relu)
                rec["mean_abs_err"], rec["max_abs_err"] = _err_bands(
                    got, x, weight, bias, PAD, h_out, w_out, c_out, relu
                )
            if ref_out is None and not abl:
                ref_out, rec["ref"] = valid, True
            if ref_out is None:
                ref_out = valid
            # bf16 ulp distance in units of the ulp at the larger magnitude of the pair (a sign flip of a tiny value
            # counts |a - b| / ulp(max), not the raw bit distance through zero).
            diff = (valid - ref_out).abs()
            mag = torch.maximum(valid.abs(), ref_out.abs()).clamp_min(2.0**-126)
            ulp_at = torch.exp2(torch.floor(torch.log2(mag)) - 7)
            ulp = diff / ulp_at
            rec["bit_identical_to_first"] = bool(torch.equal(valid, ref_out))
            rec["max_abs_vs_first"] = float(diff.max())
            rec["frac_diff_vs_first"] = float((diff > 0).float().mean())
            rec["max_ulp_vs_first"] = float(ulp.max())
            rec["n_gt1ulp_vs_first"] = int((ulp > 1.0).sum())
            rec["n_diff_vs_first"] = int((diff > 0).sum())
            if relu:
                rec["min_out"] = float(valid.min())
        except Exception as e:
            rec["status"] = "error"
            rec["error"] = str(e).splitlines()[0][:400] if str(e) else type(e).__name__
        ttnn.deallocate(tt_w)
        if tt_b is not None:
            ttnn.deallocate(tt_b)
        print("TU:", json.dumps(rec))
        with open(HERE / "runs" / f"{log}.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
    _set_knobs({})
    os.environ.pop("CONV2D_3X3_ABLATE", None)
    ttnn.deallocate(tt_in)
