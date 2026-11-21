# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Bake-off: the flat row-streamed whole-K walk on the ROW_MAJOR ring, ported onto the round-20 tree (the whole-stream
fill split, band-first kh reuse, the routed store) (perf experiment, round 20, `whole_k_rowstream_v3`).

Re-run (from the repo root):
    ttnn/ttnn/operations/conv2d_3x3/perf_experiments/whole_k_rowstream_v3/run.sh <label> 47,45 base,rs
(results: runs/<label>.txt = device ns per case|variant, runs/<label>.jsonl = PCC / bit-identity / ulp / plan).

Variants (one warm-up call, then RS_ITERS measured calls, each signposted `cmp|<case>|<variant>|<i>`):
  base     op_base/ (unmodified copy of the working-tree op)
  rs       op_copy/ descriptor defaults: kh reuse kept where it applies, not where the output stream outruns DRAM
           (output_stream_bound: C_in 32 at bf16 HiFi4)
  rsa      rs on every ring walk with resident weights (no domain rule: C_in 32 too, with the routed store)
  rsx      rsa replacing kh reuse (the row stream on the plain ring where kh reuse would apply)
  rs1/rs8  batches of 1 / 8 tile-rows (no domain rule); rsw1 one row group per store; off: ROWSTREAM off (= base)

Env: RS_CASES (comma list of FLUX bh2x4 idx), RS_VARIANTS (comma list, run in that order), RS_ORDER=alt (reverse the
variant order on every other case), RS_LOG (run-log name), RS_ITERS (measured calls, default 1), RS_KNOBS (extra
`NAME=value` knobs for every op_copy variant, comma separated). Every variant's output is compared with the first
variant's (sha1 of the output bytes, bf16 ulp histogram) and its PCC vs torch is recorded (output-row bands, as the
FLUX harness). Precision contract fixed: bf16, HiFi4, fp32 DEST, no approx, bias.
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
_BASE = "ttnn.operations.conv2d_3x3.perf_experiments.whole_k_rowstream_v3."
MODS = {}
for _k in ("op_base", "op_copy") + (("op_base_abl", "op_copy_abl") if os.environ.get("RS_ABL") == "1" else ()):
    MODS[_k] = (
        importlib.import_module(_BASE + _k + ".conv2d_3x3"),
        importlib.import_module(_BASE + _k + ".conv2d_3x3_program_descriptor"),
    )
    assert Path(MODS[_k][1].KERNEL_DIR).parent.name == _k, MODS[_k][1].KERNEL_DIR

# FLUX.2 bh2x4 case n of test_flux_conv3x3_compare.py: (H, W, C_in, C_out), ROW_MAJOR input, padding (0, 0, 1, 1).
_BH = {
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
# Extra regimes: "r<n>" = (H, W, C_in, C_out, padding, bias, relu).
_EXTRA = {
    "r2": (258, 512, 64, 256, 1, False, True),  # no bias, ReLU, padding 1 (top / bottom halo zeros)
    "r3": (258, 512, 64, 256, 1, True, True),  # bias + ReLU, padding 1
    # Either side of the whole-op roofline (DRAM time / matmul time at HiFi4 on 110 cores); both stream their output at
    # 264 GB/s (C_in 64), inside the store-rate rule:
    "r4": (1026, 2048, 64, 64, (0, 0, 1, 1), True, False),  # 1.17: DRAM side
    "r5": (1026, 2048, 64, 96, (0, 0, 1, 1), True, False),  # 0.98: compute side
    # C_in 128 (not kh reuse): no bias + ReLU, and bias + ReLU, padding 1.
    "r6": (258, 512, 128, 256, 1, False, True),
    "r7": (258, 512, 128, 256, 1, True, True),
}
BAND_ROWS = 32

KNOB_DEFAULTS = dict(
    ROWSTREAM=True,
    ROWSTREAM_ROWS=4,
    ROWSTREAM_TABLE_ROWS=8,
    ROWSTREAM_OUT_ROWS=8,
    ROWSTREAM_KROW=True,
    ROWSTREAM_STORE_GROUPS=None,
    ROWSTREAM_KHR="keep",
    ROWSTREAM_STORE_RULE=True,
)
_ALL = dict(ROWSTREAM_STORE_RULE=False)
VARIANTS = {
    "base": ("op_base", {}),
    "rs": ("op_copy", {}),
    "rsa": ("op_copy", _ALL),
    "rsx": ("op_copy", dict(_ALL, ROWSTREAM_KHR="replace")),
    "rs1": ("op_copy", dict(_ALL, ROWSTREAM_ROWS=1)),
    "rs8": ("op_copy", dict(_ALL, ROWSTREAM_ROWS=8)),
    "rsw1": ("op_copy", dict(_ALL, ROWSTREAM_STORE_GROUPS=1)),
    "off": ("op_copy", dict(ROWSTREAM=False)),
    # Ablation copies (RS_ABL=1; CONV2D_3X3_ABLATE=<tokens>): wrong output, timing only.
    "base_abl": ("op_base_abl", {}),
    "rs_abl": ("op_copy_abl", _ALL),
}


def _case(idx):
    if idx in _EXTRA:
        H, W, c_in, c_out, pad, has_bias, relu = _EXTRA[idx]
        return H, W, c_in, c_out, pad, has_bias, relu
    H, W, c_in, c_out = _BH[int(idx)]
    return H, W, c_in, c_out, (0, 0, 1, 1), True, False


def _bands(h_out):
    if h_out <= 3 * BAND_ROWS:
        return [(0, h_out)]
    mid = h_out // 2 - BAND_ROWS // 2
    return [(0, BAND_ROWS), (mid, mid + BAND_ROWS), (h_out - BAND_ROWS, h_out)]


def _ref_rows(x_hwc, weight, bias, pad, relu, r0, r1):
    import torch

    top, bottom, left, right = pad
    h = x_hwc.shape[0]
    first = r0 - top
    last = (r1 - 1) - top + 3
    rows = x_hwc[max(first, 0) : min(last, h)].float().permute(2, 0, 1).unsqueeze(0)
    rows = torch.nn.functional.pad(rows, (left, right, max(0, -first), max(0, last - h)))
    y = torch.nn.functional.conv2d(rows, weight.float(), None if bias is None else bias.float())[0].permute(1, 2, 0)
    return torch.relu(y) if relu else y


def _pcc_bands(out, x_hwc, weight, bias, pad, relu, h_out, w_out, c_out):
    import torch

    out = out.reshape(-1, out.shape[-1])[: h_out * w_out, :c_out].reshape(h_out, w_out, c_out)
    got, ref = [], []
    for r0, r1 in _bands(h_out):
        got.append(out[r0:r1].double().flatten())
        ref.append(_ref_rows(x_hwc, weight, bias, pad, relu, r0, r1).double().flatten())
    got, ref = torch.cat(got), torch.cat(ref)
    return torch.corrcoef(torch.stack([got, ref]))[0, 1].item()


def _full_ref(x_hwc, weight, bias, pad, relu):
    """The whole output in fp32 on the host, [h_out * w_out, c_out]."""
    import torch

    top, bottom, left, right = pad
    xin = x_hwc.float().permute(2, 0, 1).unsqueeze(0)
    xin = torch.nn.functional.pad(xin, (left, right, top, bottom))
    y = torch.nn.functional.conv2d(xin, weight.float(), None if bias is None else bias.float())[0]
    y = torch.relu(y) if relu else y
    return y.permute(1, 2, 0).reshape(-1, y.shape[0])


def _extra_knobs():
    knobs = {}
    for kv in [k for k in os.environ.get("RS_KNOBS", "").split(",") if k]:
        name, val = kv.split("=")
        knobs[name] = val if not val.lstrip("-").isdigit() else int(val)
    return knobs


def _set_knobs(knobs):
    for pkg in [k for k in MODS if k.startswith("op_copy")]:
        _, pd = MODS[pkg]
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
    return [c.strip() for c in os.environ.get("RS_CASES", "41").split(",") if c.strip()]


def _variants():
    return [v for v in os.environ.get("RS_VARIANTS", "base,rs").split(",") if v]


@pytest.mark.parametrize("device_params", [{"l1_small_size": 65536}], indirect=True)
@pytest.mark.parametrize("idx", _cases())
def test_rs(device, idx):
    import torch

    H, W, c_in, c_out, pad, has_bias, relu = _case(idx)
    sides = pad if isinstance(pad, tuple) else (pad,) * 4
    h_out = H + sides[0] + sides[1] - 2
    w_out = W + sides[2] + sides[3] - 2
    torch.manual_seed(0)
    x = torch.randn(H, W, c_in, dtype=torch.bfloat16)
    weight = (torch.randn(c_out, c_in, 3, 3) / (9 * c_in) ** 0.5).to(torch.bfloat16)
    bias = (0.1 * torch.randn(c_out)).to(torch.bfloat16) if has_bias else None
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
    log = os.environ.get("RS_LOG", "rs")
    if os.environ.get("RS_ORDER") == "alt" and _cases().index(idx) % 2 == 1:
        variants = variants[::-1]
    iters = int(os.environ.get("RS_ITERS", "1"))
    extra = _extra_knobs()
    ref_out = None
    full_ref = None
    for vname in variants:
        pkg, knobs = VARIANTS[vname]
        op, pd = MODS[pkg]
        is_copy = pkg.startswith("op_copy")
        knobs = {**extra, **knobs} if is_copy else {}
        _set_knobs(knobs)
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
        tt_b = op.prepare_conv2d_bias(bias, **prep) if has_bias else None

        def run():
            return op.conv2d_3x3(
                tt_in,
                tt_w,
                input_height=H,
                input_width=W,
                bias=tt_b,
                padding=pad,
                stride=1,
                compute_kernel_config=cfg,
                **({"activation": "relu"} if relu else {}),
            )

        rec = dict(case=idx, variant=vname, knobs=knobs, log=log, t=time.time())
        try:
            # The plan this variant's descriptor takes (host only).
            if is_copy and not _in_collect_pass():
                captured = {}
                orig = pd.create_program_descriptor

                def spy(*a, **kw):
                    captured["g"], captured["b"] = kw["geometry"], kw["blocking"]
                    return orig(*a, **kw)

                op.create_program_descriptor = spy
            out = run()
            ttnn.synchronize_device(device)
            ttnn.ReadDeviceProfiler(device)
            if is_copy and not _in_collect_pass():
                op.create_program_descriptor = orig
                g, b = captured["g"], captured["b"]
                rec["blocking"] = dict(
                    bm=b.block_m_tiles,
                    bk=b.block_k_tiles,
                    bn=b.block_n_tiles,
                    nkb=b.num_k_blocks,
                    m_blocks=b.max_m_blocks_per_core,
                    ring=b.rm_ring_px,
                    resident=b.weights_resident,
                    ng=b.n_groups,
                    khr=b.khr_ring_rows,
                )
                rec["plan"] = pd.rowstream_plan(
                    g, b, has_bias, pd.l1_cb_budget_bytes(0, g.l1_small_bytes), cfg.math_fidelity
                )
                rec["rm_split"] = bool(b.rm_ring_px) and not pd.output_stream_bound(g, b, cfg.math_fidelity)
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
            rec["pcc"] = _pcc_bands(got, x, weight, bias, sides, relu, h_out, w_out, c_out)
            valid = got.reshape(-1, got.shape[-1])[: h_out * w_out, :c_out].to(torch.bfloat16).contiguous()
            if full_ref is None:
                full_ref = _full_ref(x, weight, bias, sides, relu)
            err = (valid.float() - full_ref).abs()
            rec["pcc_full"] = torch.corrcoef(torch.stack([valid.double().flatten(), full_ref.double().flatten()]))[
                0, 1
            ].item()
            rec["max_abs_err"] = float(err.max())
            rec["mean_abs_err"] = float(err.mean())
            # Worst error relative to the reference's bf16 ulp there (|ref| * 2^-7, floored at 2^-7 * mean |ref|).
            scale = torch.clamp(full_ref.abs(), min=float(full_ref.abs().mean())) * 2.0**-7
            rec["max_err_in_ref_ulps"] = float((err / scale).max())
            rec["sha1"] = hashlib.sha1(valid.view(torch.int16).numpy().tobytes()).hexdigest()[:16]
            if ref_out is None:
                ref_out, rec["ref"] = valid, True
            gi = valid.view(torch.int16).to(torch.int32)
            ri = ref_out.view(torch.int16).to(torch.int32)
            # bf16 ulp distance (sign-magnitude ordering).
            gs = torch.where(gi < 0, -(gi & 0x7FFF), gi)
            rs_ = torch.where(ri < 0, -(ri & 0x7FFF), ri)
            ulp = (gs - rs_).abs()
            flip = (gi < 0) != (ri < 0)
            rec["sign_flips_vs_first"] = int((flip & (ulp > 0)).sum())
            rec["sign_flip_max_abs"] = (
                float(valid.float()[flip & (ulp > 0)].abs().max()) if bool((flip & (ulp > 0)).any()) else 0.0
            )
            ulp = torch.where(flip, torch.zeros_like(ulp), ulp)
            rec["bit_identical_to_first"] = bool(torch.equal(gi, ri))
            rec["n_diff_vs_first"] = int((ulp > 0).sum())
            rec["frac_diff_vs_first"] = float((ulp > 0).float().mean())
            rec["max_ulp_vs_first"] = int(ulp.max())
            rec["ulp_hist"] = {str(u): int((ulp == u).sum()) for u in (1, 2, 3)}
            rec["ulp_gt3"] = int((ulp > 3).sum())
            # Away from zero (|reference| at least 1/16 of its mean magnitude), where a bf16 ulp is meaningful.
            big = full_ref.abs() >= full_ref.abs().mean() / 16
            rec["ulp_gt1_significant"] = int(((ulp > 1) & big).sum())
            rec["max_ulp_significant"] = int(ulp[big].max()) if bool(big.any()) else 0
        except Exception as e:
            rec["status"] = "error"
            rec["error"] = str(e).splitlines()[0][:400] if str(e) else type(e).__name__
        ttnn.deallocate(tt_w)
        if tt_b is not None:
            ttnn.deallocate(tt_b)
        print("RS:", json.dumps(rec))
        if not _in_collect_pass():
            with open(HERE / "runs" / f"{log}.jsonl", "a") as f:
                f.write(json.dumps(rec) + "\n")
    _set_knobs({})
    ttnn.deallocate(tt_in)
