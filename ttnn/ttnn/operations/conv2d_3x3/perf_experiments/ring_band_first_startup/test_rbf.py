# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Bake-off: band-first startup of the ROW_MAJOR ring fill (perf experiment, round 19, `ring_band_first_startup`).

Re-run (from the repo root):
    ttnn/ttnn/operations/conv2d_3x3/perf_experiments/ring_band_first_startup/run.sh <label> 42,41 base,bf
(results: runs/<label>.txt = device ns per case|variant, runs/<label>.jsonl = PCC / bit-identity / blocking).

Variants (one dispatch per measured call, one warm-up call first):
  base     op_base/ (unmodified copy of the working-tree op: monotone kh-reuse order, FIFO groups across tilize steps)
  bf       op_copy/ (the recommendation: band-first kh-reuse order, FIFO groups ending at each tilize step)
  align    op_variants/: monotone order, FIFO groups ending at each tilize step only (KHR_ALIGN)
  bf_unal  op_variants/: band-first order, FIFO groups as base
  lanes    op_variants/: band-first fill on the ring walk without kh reuse (RM_LANES, RmLaneRing)
  mono, mono_r, mono_c, v_off: op_variants/ controls (band-first code with the monotone order on both / the reader /
           compute only; band-first off)
Cases: FLUX.2 bh2x4 idx (ROW_MAJOR), t<idx> (TILE input), k<n> (extra kh-reuse regimes: symmetric padding, a core
spanning less than an image row).

Env: RBF_CASES (comma list of FLUX bh2x4 idx), RBF_VARIANTS (comma list, run in that order), RBF_ORDER=alt (reverse the
variant order on every other case), RBF_LOG (run-log name), RBF_ITERS (measured calls, default 1). Each variant's calls
are signposted `cmp|<case>|<variant>|<i>` (cmp_windows.py sums them). Every variant's output is compared bit-exactly
with the first variant's output of the same case (sha1 of the output bytes, ulp diff) and its PCC vs torch is recorded
(output-row bands, as the FLUX harness). Precision contract fixed: bf16, HiFi4, fp32 DEST, no approx, bias.
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
_BASE = "ttnn.operations.conv2d_3x3.perf_experiments.ring_band_first_startup."
MODS = {}
LAST_BLOCKING = {}


def _recording(pkg, create):
    def wrapped(*args, **kwargs):
        LAST_BLOCKING[pkg] = kwargs.get("blocking")
        return create(*args, **kwargs)

    return wrapped


for _k in ("op_base", "op_copy", "op_variants"):
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
        "x10": (256, 256, 256, 512, 1, (1, 1, 1, 1), True),
        "x12": (512, 512, 256, 256, 1, (1, 1, 1, 1), True),
        "x15": (512, 512, 256, 256, 2, (0, 1, 0, 1), True),
        "x20": (512, 512, 256, 512, 1, (1, 1, 1, 1), True),
        "x22": (1024, 1024, 256, 256, 1, (1, 1, 1, 1), True),
        "r1": (258, 1024, 128, 512, 1, (0, 0, 1, 1), False),
        "r2": (130, 512, 128, 512, 1, (0, 0, 1, 1), False),
        "r3": (258, 512, 256, 512, 1, (0, 0, 1, 1), False),
        # kh-reuse regimes beyond the focus set: a core spanning less than an image row, symmetric padding (rows
        # above and below the image), a partial last M-block.
        "k1": (34, 1024, 64, 256, 1, (0, 0, 1, 1), False),
        "k2": (514, 1024, 64, 256, 1, (1, 1, 1, 1), False),
        "k3": (66, 256, 64, 64, 1, (1, 1, 1, 1), False),
        "k4": (130, 512, 128, 128, 1, (0, 0, 1, 1), False),
        "k5": (200, 1024, 64, 128, 1, (1, 1, 1, 1), False),
        "k6": (66, 1024, 64, 512, 1, (0, 0, 1, 1), False),
        "k7": (34, 1024, 64, 512, 1, (0, 0, 1, 1), False),
        "k8": (66, 512, 64, 512, 1, (1, 1, 1, 1), False),
        "k9": (98, 2048, 64, 256, 1, (0, 0, 1, 1), False),
        "k10": (98, 512, 64, 512, 1, (0, 0, 1, 1), False),
        "k11": (98, 512, 64, 384, 1, (0, 0, 1, 1), False),
        "k12": (90, 512, 64, 512, 1, (1, 1, 1, 1), False),
    }
)
BAND_ROWS = 32

KNOB_DEFAULTS = dict(
    RM_BAND_FIRST=True,
    RM_BAND_MONO=False,
    RM_BAND_SIDES="both",
    RM_LANES=False,
    RM_IDLE_WAIT=0,
    RM_PAD_NOPS=0,
    RM_EXP=(),
    RM_ALIGN_ONLY=False,
)
VARIANTS = {
    "base": ("op_base", {}),
    "bf": ("op_copy", {}),
    # The menu (op_variants/, knobs above):
    "align": ("op_variants", dict(RM_ALIGN_ONLY=True)),
    "bf_unal": ("op_variants", dict(RM_EXP=("KHR_EXP_UNALIGNED",))),
    "lanes": ("op_variants", dict(RM_LANES=True)),
    "mono": ("op_variants", dict(RM_BAND_MONO=True, RM_EXP=("KHR_EXP_UNALIGNED",))),
    "mono_r": ("op_variants", dict(RM_BAND_MONO=True, RM_BAND_SIDES="reader", RM_EXP=("KHR_EXP_UNALIGNED",))),
    "mono_c": ("op_variants", dict(RM_BAND_MONO=True, RM_BAND_SIDES="compute")),
    "v_off": ("op_variants", dict(RM_BAND_FIRST=False)),
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


def _pcc_bands(out, x_hwc, weight, bias, stride, pad, h_out, w_out, c_out):
    import torch

    out = out.reshape(-1, out.shape[-1])[: h_out * w_out, :c_out].reshape(h_out, w_out, c_out)
    got, ref = [], []
    for r0, r1 in _bands(h_out):
        got.append(out[r0:r1].double().flatten())
        ref.append(_ref_rows(x_hwc, weight, bias, stride, pad, r0, r1).double().flatten())
    got, ref = torch.cat(got), torch.cat(ref)
    return torch.corrcoef(torch.stack([got, ref]))[0, 1].item()


def _set_knobs(knobs):
    for pkg in [k for k in MODS if k == "op_variants"]:
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
    return [c.strip() for c in os.environ.get("RBF_CASES", "42").split(",") if c.strip()]


def _variants():
    return [v for v in os.environ.get("RBF_VARIANTS", "base,bf").split(",") if v]


@pytest.mark.parametrize("device_params", [{"l1_small_size": 65536}], indirect=True)
@pytest.mark.parametrize("idx", _cases())
def test_rbf(device, idx):
    import torch

    H, W, c_in, c_out, stride, pad, tile = FLUX[idx]
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
        math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False
    )
    variants = _variants()
    log = os.environ.get("RBF_LOG", "rbf")
    if os.environ.get("RBF_ORDER") == "alt" and _cases().index(idx) % 2 == 1:
        variants = variants[::-1]
    iters = int(os.environ.get("RBF_ITERS", "1"))
    ref_out = None
    for vname in variants:
        pkg, knobs = VARIANTS[vname]
        op, pd = MODS[pkg]
        _set_knobs(knobs if pkg == "op_variants" else {})
        device.clear_program_cache()
        prep = dict(
            stride=stride,
            padding=pad_arg,
            dilation=1,
            input_dtype=ttnn.bfloat16,
            input_layout=layout,
            weight_dtype=ttnn.bfloat16,
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
                bias=tt_b,
                padding=pad_arg,
                stride=stride,
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
                    khr=getattr(b, "khr_ring_rows", 0),
                )
            rec["pcc"] = _pcc_bands(got, x, weight, bias, stride, pad, h_out, w_out, c_out)
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
        except Exception as e:
            rec["status"] = "error"
            rec["error"] = str(e).splitlines()[0][:400] if str(e) else type(e).__name__
        ttnn.deallocate(tt_w)
        ttnn.deallocate(tt_b)
        print("RBF:", json.dumps(rec))
        if not _in_collect_pass():
            with open(HERE / "runs" / f"{log}.jsonl", "a") as f:
                f.write(json.dumps(rec) + "\n")
    _set_knobs({})
    ttnn.deallocate(tt_in)
