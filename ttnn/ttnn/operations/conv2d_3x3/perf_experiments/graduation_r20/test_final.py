# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Perf 20 graduation check: the round-start op (op_base/, the working tree before this round) against the graduated op
(op_cand/, a snapshot of the real op after the round's graduations), per FLUX.2 bh2x4 case: device ns per call
(signposts `cmp|<case>|<variant>|<i>`), PCC vs torch on output-row bands, and bit-identity (sha1 of the output bytes,
ulp diff) against the first variant run. Adapted from ring_fill_steady_split/test_rfss.py.

    GR_CASES=41,36 GR_VARIANTS=base,cand scripts/run_safe_pytest.sh --profile --run-all \
        ttnn/ttnn/operations/conv2d_3x3/perf_experiments/graduation_r19/test_final.py
Results: runs/<GR_LOG>.jsonl.
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
_BASE = "ttnn.operations.conv2d_3x3.perf_experiments.graduation_r20."
MODS = {}
LAST_BLOCKING = {}


def _recording(pkg, create):
    def wrapped(*args, **kwargs):
        LAST_BLOCKING[pkg] = kwargs.get("blocking")
        return create(*args, **kwargs)

    return wrapped


for _k in ("op_base", "op_prev", "op_cand"):
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
        "x4": (256, 256, 32, 512, 1, (1, 1, 1, 1), True),
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
    }
)
BAND_ROWS = 32

VARIANTS = {"base": ("op_base", {}), "prev": ("op_prev", {}), "cand": ("op_cand", {})}


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


def _in_collect_pass():
    import sys

    plugin = sys.modules.get("tests.plugins.up_front_collect")
    return plugin is not None and getattr(plugin, "_PASS", "real") == "collect"


def _cases():
    return [c.strip() for c in os.environ.get("GR_CASES", "43").split(",") if c.strip()]


def _variants():
    return [v for v in os.environ.get("GR_VARIANTS", "base,cand").split(",") if v]


@pytest.mark.parametrize("device_params", [{"l1_small_size": 65536}], indirect=True)
@pytest.mark.parametrize("idx", _cases())
def test_final(device, idx):
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
    log = os.environ.get("GR_LOG", "final")
    if os.environ.get("GR_ORDER") == "alt" and _cases().index(idx) % 2 == 1:
        variants = variants[::-1]
    iters = int(os.environ.get("GR_ITERS", "1"))
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
        print("GR:", json.dumps(rec))
        if not _in_collect_pass():
            with open(HERE / "runs" / f"{log}.jsonl", "a") as f:
                f.write(json.dumps(rec) + "\n")
    ttnn.deallocate(tt_in)
