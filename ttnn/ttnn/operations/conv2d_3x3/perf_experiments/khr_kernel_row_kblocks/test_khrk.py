# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Perf 20 khr_kernel_row_kblocks: the round-start op (op_base/) against kernel-row K-blocks on the kh-reuse walk
(op_copy/), per FLUX.2 bh2x4 case: device ns per call (signposts `cmp|<case>|<variant>|<i>`), PCC vs torch on
output-row bands, and bit-identity (sha1 of the output bytes, ulp diff) against the first variant run. From
graduation_r19/test_final.py.

    KR_LOG=<label> KR_CASES=42,38 KR_VARIANTS=base,krow [KR_ORDER=alt KR_ITERS=3] scripts/run_safe_pytest.sh \
        --profile --run-all ttnn/ttnn/operations/conv2d_3x3/perf_experiments/khr_kernel_row_kblocks/test_khrk.py
Results: runs/<KR_LOG>.jsonl.
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
_BASE = "ttnn.operations.conv2d_3x3.perf_experiments.khr_kernel_row_kblocks."
MODS = {}
LAST_BLOCKING = {}


def _recording(pkg, create):
    def wrapped(*args, **kwargs):
        LAST_BLOCKING[pkg] = kwargs.get("blocking")
        return create(*args, **kwargs)

    return wrapped


# The variants other than op_base / op_copy / op_ftr live as patches (variants/<name>.patch against op_base): recreate
# one with `cp -r op_base op_<name> && patch -p1 -d op_<name> < variants/<name>.patch`.
for _k in [d for d in ("op_base", "op_copy", "op_ft", "op_ftr", "op_rt3", "op_ftc", "op_sb2") if (HERE / d).is_dir()]:
    MODS[_k] = (
        importlib.import_module(_BASE + _k + ".conv2d_3x3"),
        importlib.import_module(_BASE + _k + ".conv2d_3x3_program_descriptor"),
    )
    assert Path(MODS[_k][1].KERNEL_DIR).parent.name == _k, MODS[_k][1].KERNEL_DIR
    # The blocking each call ran with (for the record).
    MODS[_k][0].create_program_descriptor = _recording(_k, MODS[_k][0].create_program_descriptor)

# Cases: (H, W, C_in, C_out, stride, padding (top, bottom, left, right), TILE input).
# "<n>": FLUX.2 bh2x4 case n of test_flux_conv3x3_compare.py (ROW_MAJOR input, padding (0, 0, 1, 1));
# "t<n>": the same on TILE input (FLUX_CMP_BH_LAYOUT=TILE); "x<n>": its 1x1 (single-device, TILE input) case n;
# "r<n>": extra ROW_MAJOR regimes (not FLUX cases).
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
FLUX = {str(k): (*v, 1, (0, 0, 1, 1), False) for k, v in _BH.items()}
FLUX.update({f"t{k}": (*v, 1, (0, 0, 1, 1), True) for k, v in _BH.items()})
FLUX.update(
    {
        "x2": (1024, 1024, 128, 3, 1, (1, 1, 1, 1), True),
        "x18": (1024, 1024, 256, 128, 1, (1, 1, 1, 1), True),
        "x10": (256, 256, 256, 512, 1, (1, 1, 1, 1), True),
        "x12": (512, 512, 256, 256, 1, (1, 1, 1, 1), True),
        "x15": (512, 512, 256, 256, 2, (0, 1, 0, 1), True),
        "x20": (512, 512, 256, 512, 1, (1, 1, 1, 1), True),
        "x22": (1024, 1024, 256, 256, 1, (1, 1, 1, 1), True),
        "r1": (258, 1024, 128, 512, 1, (0, 0, 1, 1), False),
        "r2": (130, 512, 128, 512, 1, (0, 0, 1, 1), False),
        "r3": (258, 512, 256, 512, 1, (0, 0, 1, 1), False),
        # A core's range shorter than an image row: its three bands are separate stream segments.
        "r6": (66, 2048, 32, 512, 1, (1, 1, 1, 1), False),
        # kh reuse with streamed (non-resident) weights.
        "r7": (258, 64, 128, 1024, 1, (0, 0, 1, 1), False),
        # kh reuse, other blockings (host sweep): N split bn 16, bn 2, bm 15, pad 1 all round.
        "r8": (258, 256, 64, 1024, 1, (0, 0, 1, 1), False),
        "r9": (1026, 256, 128, 64, 1, (0, 0, 1, 1), False),
        "r10": (1026, 256, 64, 256, 1, (0, 0, 1, 1), False),
        "r11": (130, 256, 128, 512, 1, (1, 1, 1, 1), False),
    }
)
BAND_ROWS = 32

# base: the round-start op; krow: kernel-row K-blocks (recommended); ftr: kernel rows past the first M-block (runtime
# taps); ft: the same with two walk instantiations; rt3: ftr's runtime walk at taps 3 everywhere (control); ftc: ftr
# with compile-time k loops; sb2: krow with 2x2 subblocks where bm, W / 32 are even.
VARIANTS = {
    "base": ("op_base", {}),
    "krow": ("op_copy", {}),
    "ft": ("op_ft", {}),
    "ftr": ("op_ftr", {}),
    "rt3": ("op_rt3", {}),
    "ftc": ("op_ftc", {}),
    "sb2": ("op_sb2", {}),
}


def _bands(h_out):
    if h_out <= 3 * BAND_ROWS:
        return [(0, h_out)]
    mid = h_out // 2 - BAND_ROWS // 2
    return [(0, BAND_ROWS), (mid, mid + BAND_ROWS), (h_out - BAND_ROWS, h_out)]


def _ref_rows(x_hwc, weight, bias, stride, pad, r0, r1):
    import torch

    top, bottom, left, right = pad
    h = x_hwc.shape[0]
    first = r0 * stride - top
    last = (r1 - 1) * stride - top + 3
    rows = x_hwc[max(first, 0) : min(last, h)].float().permute(2, 0, 1).unsqueeze(0)
    rows = torch.nn.functional.pad(rows, (left, right, max(0, -first), max(0, last - h)))
    return torch.nn.functional.conv2d(rows, weight.float(), bias.float(), stride=stride)[0].permute(1, 2, 0)


def _pcc_bands(out, x_hwc, weight, bias, stride, pad, h_out, w_out, c_out, activation=None):
    import torch

    out = out.reshape(-1, out.shape[-1])[: h_out * w_out, :c_out].reshape(h_out, w_out, c_out)
    got, ref = [], []
    for r0, r1 in _bands(h_out):
        got.append(out[r0:r1].double().flatten())
        r = _ref_rows(x_hwc, weight, bias, stride, pad, r0, r1)
        if activation == "relu":
            r = torch.relu(r)
        ref.append(r.double().flatten())
    got, ref = torch.cat(got), torch.cat(ref)
    err = (got - ref).abs()
    return torch.corrcoef(torch.stack([got, ref]))[0, 1].item(), err.mean().item(), err.max().item()


def _in_collect_pass():
    import sys

    plugin = sys.modules.get("tests.plugins.up_front_collect")
    return plugin is not None and getattr(plugin, "_PASS", "real") == "collect"


def _cases():
    return [c.strip() for c in os.environ.get("KR_CASES", "43").split(",") if c.strip()]


def _variants():
    return [v for v in os.environ.get("KR_VARIANTS", "base,krow").split(",") if v]


@pytest.mark.parametrize("device_params", [{"l1_small_size": 65536}], indirect=True)
@pytest.mark.parametrize("idx", _cases())
def test_khrk(device, idx):
    import torch

    # "<case>[:nb][:relu][:w8][:lofi][:nofp32]": no bias / ReLU / bfloat8_b weights / LoFi / fp32_dest_acc_en off.
    key, *mods = idx.split(":")
    H, W, c_in, c_out, stride, pad, tile = FLUX[key]
    use_bias = "nb" not in mods
    activation = "relu" if "relu" in mods else None
    h_out = (H + pad[0] + pad[1] - 3) // stride + 1
    w_out = (W + pad[2] + pad[3] - 3) // stride + 1
    layout = ttnn.TILE_LAYOUT if tile else ttnn.ROW_MAJOR_LAYOUT
    pad_arg = pad if len(set(pad)) > 1 else pad[0]
    torch.manual_seed(0)
    x = torch.randn(H, W, c_in, dtype=torch.bfloat16)
    weight = (torch.randn(c_out, c_in, 3, 3) / (9 * c_in) ** 0.5).to(torch.bfloat16)
    bias = (0.1 * torch.randn(c_out)).to(torch.bfloat16)
    tt_in = ttnn.from_torch(
        x.reshape(1, 1, H * W, c_in),
        dtype=ttnn.bfloat16,
        layout=layout,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.LoFi if "lofi" in mods else ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en="nofp32" not in mods,
        math_approx_mode=False,
    )
    w_dtype = ttnn.bfloat8_b if "w8" in mods else ttnn.bfloat16
    variants = _variants()
    log = os.environ.get("KR_LOG", "final")
    if os.environ.get("KR_ORDER") == "alt" and _cases().index(idx) % 2 == 1:
        variants = variants[::-1]
    iters = int(os.environ.get("KR_ITERS", "1"))
    ref_out = None
    for vname in variants:
        pkg, knobs = VARIANTS[vname]
        op, pd = MODS[pkg]
        device.clear_program_cache()
        prep = dict(
            stride=stride,
            padding=pad_arg,
            dilation=1,
            input_dtype=ttnn.bfloat16,
            input_layout=layout,
            weight_dtype=w_dtype,
            input_memory_config=tt_in.memory_config(),
            device=device,
        )
        tt_w = op.prepare_conv2d_weights(weight, **prep)
        tt_b = op.prepare_conv2d_bias(bias, **prep)

        def run():
            return op.conv2d_3x3(
                tt_in,
                tt_w,
                input_height=H,
                input_width=W,
                bias=tt_b if use_bias else None,
                padding=pad_arg,
                stride=stride,
                activation=activation,
                compute_kernel_config=cfg,
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
            b = LAST_BLOCKING.get(pkg)
            if b is not None:
                rec["blocking"] = dict(
                    ng=getattr(b, "n_groups", 1),
                    cores=b.num_m_groups * getattr(b, "n_groups", 1),
                    core_m=b.core_m_tiles,
                    bm=b.block_m_tiles,
                    bk=b.block_k_tiles,
                    bn=b.block_n_tiles,
                    m_blocks=b.max_m_blocks_per_core,
                    k_blocks=b.num_k_blocks,
                    resident=b.weights_resident,
                    footprint=b.footprint_bytes,
                    depth=b.im2col_rm_depth_rows,
                    ring_px=getattr(b, "rm_ring_px", 0),
                    split=getattr(b, "rm_split", False),
                    stage=(getattr(b, "rm_stage_n", 0), getattr(b, "rm_stage_px", 0)),
                )
            ref_bias = bias if use_bias else torch.zeros_like(bias)
            rec["pcc"], rec["mae_vs_torch"], rec["maxerr_vs_torch"] = _pcc_bands(
                got, x, weight, ref_bias, stride, pad, h_out, w_out, c_out, activation
            )
            valid = got.reshape(-1, got.shape[-1])[: h_out * w_out, :c_out].to(torch.bfloat16).contiguous()
            rec["sha1"] = hashlib.sha1(valid.view(torch.int16).numpy().tobytes()).hexdigest()[:16]
            if ref_out is None:
                ref_out, rec["ref"] = valid, True
            gi = valid.view(torch.int16).to(torch.int32)
            ri = ref_out.view(torch.int16).to(torch.int32)
            ulp = (gi - ri).abs()
            rec["bit_identical_to_first"] = bool(torch.equal(gi, ri))
            rec["n_diff_vs_first"] = int((ulp > 0).sum())
            rec["max_ulp_vs_first"] = int(ulp.max())
            if rec["n_diff_vs_first"]:
                # Differences in bf16 ulps of the larger magnitude (sign-safe), and their magnitudes.
                g32, r32 = valid.float(), ref_out.float()
                d = (g32 - r32).abs()
                mask = d > 0
                mag = torch.maximum(g32.abs(), r32.abs())[mask]
                ulp_size = torch.exp2(torch.floor(torch.log2(mag)) - 7)
                u = (d[mask] / ulp_size).round().to(torch.int64)
                rec["true_ulp_hist"] = {
                    "1": int((u <= 1).sum()),
                    "2": int((u == 2).sum()),
                    "3-4": int(((u >= 3) & (u <= 4)).sum()),
                    "5-16": int(((u >= 5) & (u <= 16)).sum()),
                    ">16": int((u > 16).sum()),
                }
                rec["max_true_ulp"] = int(u.max())
                rec["max_abs_diff_vs_first"] = float(d.max())
                rec["out_absmax"] = float(r32.abs().max())
                rec["n_out"] = int(valid.numel())
        except Exception as e:
            rec["status"] = "error"
            rec["error"] = str(e)[:3000] if str(e) else type(e).__name__
        ttnn.deallocate(tt_w)
        ttnn.deallocate(tt_b)
        print("KR:", json.dumps(rec))
        if not _in_collect_pass():
            with open(HERE / "runs" / f"{log}.jsonl", "a") as f:
                f.write(json.dumps(rec) + "\n")
    ttnn.deallocate(tt_in)
