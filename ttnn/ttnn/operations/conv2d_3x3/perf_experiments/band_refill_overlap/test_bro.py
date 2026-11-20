# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Isolated bake-off: overlap the ROW_MAJOR bank-run band refill with compute (perf experiment band_refill_overlap,
Perf 17).

Two private copies of conv2d_3x3 (working tree at HEAD ec60e2fa9f2 plus the uncommitted user changes):
  op_base/  unmodified: per M-block the reader fills all three kh bands (bank-run chunks, staging aliasing cb_im2col_rm)
            only after compute popped every tap-direct entry of the previous block, so fill and compute serialize.
  op_copy/  BRO_OVERLAP: the bank-run fill runs as a background engine in the reader across M-blocks. Each chunk is
            de-interleaved into the slab once compute popped every unit of the previous block's kernel rows that read the
            slab range it overwrites (kernel row min(2, x / band_stride_px) for slab pixel x); the K-block loop pumps it
            only up to the band its next K-block needs and polls it while compute's unit table is full. The staging
            ring gets its own L1 (the slab CB's tail), carved from the L1 left at the solved (bm, bk) before the deeper
            im2col ring (bm / bk never change).

Variants (VARIANTS): base (op_base), rec (op_copy: the recommended design, the integration patch), and the bake-off's
knob sweeps on op_knobs/ (the experiment revision: bro = polled background refill everywhere, lazy = per-band fill, ...). Every variant's output is compared bit-exactly with the first variant's (sha1 of the output bytes, ulp
counts if not equal) and its PCC vs torch (3 output-row bands) is recorded. Precision contract fixed: bf16, HiFi4,
fp32 DEST, no approx, bias.

Re-run from the repo root (after `export TT_METAL_HOME=$PWD PYTHONPATH=$PWD`):
    ttnn/ttnn/operations/conv2d_3x3/perf_experiments/band_refill_overlap/run.sh <label> idx41,idx35 base,bro
results: runs/<label>.txt (device ns per case|variant via cmp_windows.py), runs/<label>.jsonl (blocking, PCC, sha1).
Env: BRO_CASES, BRO_VARIANTS, BRO_ITERS (measured calls, default 1), BRO_ORDER=alt (reverse the variant order on every
other case), BRO_LOG.
"""

import hashlib
import importlib
import json
import os
import sys
import time
from pathlib import Path

import pytest
import ttnn
from tracy import signpost

HERE = Path(__file__).parent
_PKG = "ttnn.operations.conv2d_3x3.perf_experiments.band_refill_overlap."
MODS = {}
for _k in ("op_base", "op_copy", "op_knobs"):
    MODS[_k] = (
        importlib.import_module(_PKG + _k + ".conv2d_3x3"),
        importlib.import_module(_PKG + _k + ".conv2d_3x3_program_descriptor"),
    )
    assert Path(MODS[_k][1].KERNEL_DIR).parent.name == _k, MODS[_k][1].KERNEL_DIR

PAD = (0, 0, 1, 1)
# FLUX.2 bh2x4 per-device decoder convs (test_flux_conv3x3_compare.py FLUX_CMP_CASES index): (H, W, C_in, C_out, pad).
CASES = {
    "idx41": (1026, 2048, 32, 128, PAD),
    "idx35": (1026, 2048, 128, 3, PAD),
    "idx43": (258, 512, 128, 512, PAD),
    "idx36": (514, 1024, 32, 128, PAD),
    "idx38": (130, 256, 128, 512, PAD),
    "idx33": (514, 1024, 128, 3, PAD),
    "idx42": (514, 1024, 64, 256, PAD),
    "idx44": (1026, 2048, 64, 128, PAD),
    "idx37": (258, 512, 64, 256, PAD),
    "idx47": (514, 1024, 128, 512, PAD),
    "idx34": (66, 128, 128, 512, PAD),
    "idx46": (1026, 2048, 64, 256, PAD),
    "idx39": (514, 1024, 64, 128, PAD),
    "idx45": (514, 1024, 128, 256, PAD),
    "idx40": (258, 512, 128, 256, PAD),
    "idx32": (130, 256, 32, 128, PAD),
    "idx31": (66, 128, 32, 128, PAD),
    # Correctness regimes (ROW_MAJOR DRAM interleaved): symmetric padding, W not a multiple of 32 (tile-rows straddle
    # output rows), dilation 2 (overlapping bands at stride 2W), overlapping bands (bm * 32 >= W), one M-block per core,
    # a partial last block, C_in 96 / 256.
    "s128x100_c64_p1": (128, 100, 64, 64, 1),
    "s64x100_c32_p1": (64, 100, 32, 64, 1),
    "s64x200_c64_d2p2": (64, 200, 64, 32, 2),
    "s258x512_c32_p1": (258, 512, 32, 128, 1),
    "s130x1024_c128_p1": (130, 1024, 128, 64, 1),
    "s98x96_c64_p0011": (98, 96, 64, 64, (0, 0, 1, 1)),
    "s200x72_c96_p1": (200, 72, 96, 64, 1),
    "s64x300_c256_p1": (64, 300, 256, 32, 1),
    # Multi-block regimes (H * W a multiple of 32, the op's contract): W not a multiple of 32 with disjoint (516x1000)
    # and overlapping bands spanning > 2 W (1028x1000 C_in 32), C_in 96 with overlapping bands, dilation 2 (bands at
    # stride 2W), C_in 256 (the leftover staging would be 8-pixel chunks: the serialized fill).
    "s516x1000_c64_p1": (516, 1000, 64, 128, 1),
    "s1028x1000_c32_p1": (1028, 1000, 32, 64, 1),
    "s304x700_c96_p1": (304, 700, 96, 64, 1),
    "s400x600_c32_d2p2": (400, 600, 32, 64, 2),
    "s514x1024_c256_p1": (514, 1024, 256, 64, 1),
}
DILATION = {"s64x200_c64_d2p2": 2, "s400x600_c32_d2p2": 2}
BAND_ROWS = 32

# op_copy: the recommended design (BRO_POLL "auto"; the integration patch). op_knobs: the experiment revision with every
# knob the bake-off swept (its defaults: polled refill everywhere = "bro").
KNOB_DEFAULTS = {
    "op_copy": dict(BRO_OVERLAP=True, BRO_POLL="auto"),
    "op_knobs": dict(
        BRO_OVERLAP=True,
        BRO_POLL_BANKS=8,
        BRO_STAGE_MODE="leftover",
        BRO_NSTG_MAX=4,
        BRO_WRITE_YIELD=-1,
        BRO_WX=False,
        BRO_POLL=True,
        BRO_TABLE_KBLOCKS=1,
        BRO_MIN_CHUNK_PX=16,
    ),
}
VARIANTS = {
    "base": ("op_base", {}),
    # Recommended: per-band fill (lazy) by default, polled refill where the output writes are light (conv_out), the
    # serialized fill where the dedicated staging would be under 16-pixel chunks.
    "rec": ("op_copy", {}),
    "rec_off": ("op_copy", dict(BRO_OVERLAP=False)),
    # Experiment revision (op_knobs). bro: polled background refill everywhere (the brief's design).
    "bro": ("op_knobs", {}),
    "off": ("op_knobs", dict(BRO_OVERLAP=False)),
    "pb1": ("op_knobs", dict(BRO_POLL_BANKS=1)),
    "pb2": ("op_knobs", dict(BRO_POLL_BANKS=2)),
    "pb4": ("op_knobs", dict(BRO_POLL_BANKS=4)),
    "n2": ("op_knobs", dict(BRO_NSTG_MAX=2)),
    "wy0": ("op_knobs", dict(BRO_WRITE_YIELD=0)),
    "n2wy0": ("op_knobs", dict(BRO_NSTG_MAX=2, BRO_WRITE_YIELD=0)),
    "wx": ("op_knobs", dict(BRO_WX=True)),
    "n2wx": ("op_knobs", dict(BRO_NSTG_MAX=2, BRO_WX=True)),
    # lazy: no polled refill, each band filled when its first K-block needs it.
    "lazy": ("op_knobs", dict(BRO_POLL=False)),
    "n2lazy": ("op_knobs", dict(BRO_NSTG_MAX=2, BRO_POLL=False)),
    "lazy_t2": ("op_knobs", dict(BRO_POLL=False, BRO_TABLE_KBLOCKS=2)),
    "lazy_t3": ("op_knobs", dict(BRO_POLL=False, BRO_TABLE_KBLOCKS=3)),
    "bro_t2": ("op_knobs", dict(BRO_TABLE_KBLOCKS=2)),
    "auto": ("op_knobs", dict(BRO_POLL="auto")),
}


def _set_knobs(pkg, knobs):
    if pkg not in KNOB_DEFAULTS:
        return
    _, pd = MODS[pkg]
    for k, v in {**KNOB_DEFAULTS[pkg], **knobs}.items():
        assert hasattr(pd, k), (pkg, k)
        setattr(pd, k, v)
    for name in dir(pd):
        f = getattr(pd, name)
        if hasattr(f, "cache_clear"):
            f.cache_clear()


def _out_hw(h, w, pad, d):
    top, bottom, left, right = pad if isinstance(pad, tuple) else (pad,) * 4
    return h + top + bottom - 2 * d, w + left + right - 2 * d


def _bands(h_out):
    if h_out <= 3 * BAND_ROWS:
        return [(0, h_out)]
    mid = h_out // 2 - BAND_ROWS // 2
    return [(0, BAND_ROWS), (mid, mid + BAND_ROWS), (h_out - BAND_ROWS, h_out)]


def _ref_rows(x_hwc, weight, bias, pad, d, r0, r1):
    import torch

    top, bottom, left, right = pad if isinstance(pad, tuple) else (pad,) * 4
    h = x_hwc.shape[0]
    first = r0 - top
    last = (r1 - 1) - top + 2 * d + 1
    rows = x_hwc[max(first, 0) : min(last, h)].float().permute(2, 0, 1).unsqueeze(0)
    rows = torch.nn.functional.pad(rows, (left, right, max(0, -first), max(0, last - h)))
    return torch.nn.functional.conv2d(rows, weight.float(), bias.float(), dilation=d)[0].permute(1, 2, 0)


def _pcc(out, x_hwc, weight, bias, pad, d, h_out, w_out, c_out):
    import torch

    out = out.reshape(-1, out.shape[-1])[: h_out * w_out, :c_out].reshape(h_out, w_out, c_out)
    got, ref = [], []
    for r0, r1 in _bands(h_out):
        got.append(out[r0:r1].double().flatten())
        ref.append(_ref_rows(x_hwc, weight, bias, pad, d, r0, r1).double().flatten())
    got, ref = torch.cat(got), torch.cat(ref)
    return torch.corrcoef(torch.stack([got, ref]))[0, 1].item()


def _in_collect_pass():
    plugin = sys.modules.get("tests.plugins.up_front_collect")
    return plugin is not None and getattr(plugin, "_PASS", "real") == "collect"


def _cases():
    return [c for c in os.environ.get("BRO_CASES", "idx41").split(",") if c]


def _variants():
    return [v for v in os.environ.get("BRO_VARIANTS", "base,bro").split(",") if v]


@pytest.mark.parametrize("device_params", [{"l1_small_size": 65536}], indirect=True)
@pytest.mark.parametrize("case", _cases())
def test_bro(device, case):
    import torch

    H, W, c_in, c_out, pad = CASES[case]
    d = DILATION.get(case, 1)
    h_out, w_out = _out_hw(H, W, pad, d)
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
    cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False
    )
    variants = _variants()
    if os.environ.get("BRO_ORDER") == "alt" and _cases().index(case) % 2 == 1:
        variants = variants[::-1]
    iters = int(os.environ.get("BRO_ITERS", "1"))
    log = os.environ.get("BRO_LOG", "bro")
    ref_out = None
    recs = []
    for vname in variants:
        pkg, knobs = VARIANTS[vname]
        op, pd = MODS[pkg]
        _set_knobs(pkg, knobs)
        device.clear_program_cache()
        prep = dict(
            stride=1,
            padding=pad,
            dilation=d,
            input_dtype=ttnn.bfloat16,
            input_layout=ttnn.ROW_MAJOR_LAYOUT,
            weight_dtype=ttnn.bfloat16,
            input_memory_config=tt_in.memory_config(),
            device=device,
        )
        tt_w = op.prepare_conv2d_weights(weight, **prep)
        tt_b = op.prepare_conv2d_bias(bias, **prep)
        captured = {}
        real_cpd = op.create_program_descriptor

        def capture(*args, geometry, blocking, **kwargs):
            captured["b"] = blocking
            return real_cpd(*args, geometry=geometry, blocking=blocking, **kwargs)

        op.create_program_descriptor = capture

        def run():
            return op.conv2d_3x3(
                tt_in,
                tt_w,
                input_height=H,
                input_width=W,
                bias=tt_b,
                padding=pad,
                stride=1,
                dilation=d,
                compute_kernel_config=cfg,
            )

        rec = dict(case=case, variant=vname, knobs=dict(knobs), log=log, t=time.time())
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
            if collect:
                ttnn.deallocate(out)
                continue
            got = ttnn.to_torch(out)
            ttnn.deallocate(out)
            rec["status"] = "ok"
            b = captured.get("b")
            if b is not None:
                rec["blocking"] = dict(
                    bm=b.block_m_tiles,
                    bk=b.block_k_tiles,
                    m_blocks=b.max_m_blocks_per_core,
                    resident=b.weights_resident,
                    im2col_depth=b.im2col_rm_depth_rows,
                    band_len=b.band_len_px,
                    band_stride=b.band_stride_px,
                    bro_nstg=getattr(b, "bro_nstg", 0),
                    bro_chunk_px=getattr(b, "bro_chunk_px", 0),
                    foot_kb=round(b.footprint_bytes / 1024, 1),
                )
            valid = got.reshape(-1, got.shape[-1])[: h_out * w_out, :c_out].contiguous()
            rec["sha1"] = hashlib.sha1(valid.view(torch.int16).numpy().tobytes()).hexdigest()
            rec["pcc"] = _pcc(got.float(), x, weight, bias, pad, d, h_out, w_out, c_out)
            if ref_out is None:
                ref_out, rec["ref"] = valid, vname
            rec["bit_identical"] = bool(torch.equal(valid.view(torch.int16), ref_out.view(torch.int16)))
            if not rec["bit_identical"]:
                ulp = (valid.view(torch.int16).to(torch.int32) - ref_out.view(torch.int16).to(torch.int32)).abs()
                rec["frac_diff"] = float((ulp > 0).float().mean())
                rec["max_ulp"] = int(ulp.max())
        except Exception as e:
            rec["status"] = "error"
            rec["error"] = str(e).splitlines()[0][:400] if str(e) else type(e).__name__
        finally:
            op.create_program_descriptor = real_cpd
        ttnn.deallocate(tt_w)
        ttnn.deallocate(tt_b)
        recs.append(rec)
        print("BRO:", json.dumps(rec))
        (HERE / "runs").mkdir(exist_ok=True)
        with open(HERE / "runs" / f"{log}.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
    for pkg in KNOB_DEFAULTS:
        _set_knobs(pkg, {})
    ttnn.deallocate(tt_in)
    if not collect:
        bad = [(r["variant"], r.get("error", "not bit-identical")) for r in recs if not r.get("bit_identical")]
        assert not bad, bad
