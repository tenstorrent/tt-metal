# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Isolated bake-off: the ROW_MAJOR halo slab as a ring across M-blocks (perf experiment rm_ring_slab, round 16).

Runs a private copy of conv2d_3x3 (op_copy/), switched by descriptor knobs (VARIANTS below):
  base         the op's current path: per M-block the reader reads three kh bands (disjoint when bm * 32 < W, so each
               input pixel is read ~3x per core), one page per pixel, and before refilling waits for compute to pop every
               tap-direct entry (fill and compute serialize).
  ring         (recommended) cb_halo_slab is a FIFO ring of consecutive input pixels (kernels/conv2d_3x3_rm_ring.hpp):
               each pixel is read once per core, still one page per pixel; the next block's pixels are issued in 8-pixel
               steps while the unit FIFO is full (compute behind), ring slots freed K-block by K-block. The ring counts
               in the L1 solve at the slab solve's bk (bm shrinks where it does not fit; it displaces the deep im2col
               ring first). No ring where a core runs one M-block.
  ring_serial  same ring and blocking, each block's new pixels issued at its start after compute popped every entry
               (isolates the read-count win from the overlap).
  ring_fit     ring only where it fits at the slab solve's bm (else the base program).
  ring_anybk   the solve may also shrink bk for the ring (not bit-identical where it does).
  ring_ch*/mf* fill step size / in-flight bound sweeps.  *_bm<N>: correctness coverage with bm capped at N.

Precision contract (fixed): bf16 act / weights, HiFi4, fp32 DEST, bias. Every ring variant must be bit-identical to base
(torch.equal, asserted unless CONV2D_3X3_ABLATE is set), and each run's PCC vs torch (3 output-row bands) is recorded.

One process per invocation; signposts `cmp|<case>|<variant>|<i>` (cmp_windows.py-compatible). See run.sh:
  ./run.sh <tag> idx41,idx35 base,ring        (ABLATE=fill|weights|store|... for the ablation hooks)
Never profile this copy next to the real op in one process (zone-id hash collisions).
"""

import json
import os
import sys
import time
from pathlib import Path

import pytest
import ttnn
from tracy import signpost

sys.path.insert(0, str(Path(__file__).parent))
import rmr_load  # noqa: E402

opc, pd, op_mod = rmr_load.load()

RESULTS = Path(os.environ.get("RMR_RESULTS", str(Path(__file__).parent / "results.jsonl")))
MEASURED_ITERS = int(os.environ.get("RMR_ITERS", "3"))
BAND_ROWS = 32

# FLUX.2 bh2x4 per-device decoder convs (ROW_MAJOR DRAM input, padding (0, 0, 1, 1)); idx = FLUX_CMP_CASES index.
FOCUS = {
    "idx41": (1026, 2048, 32, 128, (0, 0, 1, 1)),
    "idx42": (514, 1024, 64, 256, (0, 0, 1, 1)),
    "idx35": (1026, 2048, 128, 3, (0, 0, 1, 1)),
    "idx36": (514, 1024, 32, 128, (0, 0, 1, 1)),
    "idx44": (1026, 2048, 64, 128, (0, 0, 1, 1)),
    "idx46": (1026, 2048, 64, 256, (0, 0, 1, 1)),
    "idx43": (258, 512, 128, 512, (0, 0, 1, 1)),
    "idx47": (514, 1024, 128, 512, (0, 0, 1, 1)),
}
# The rest of the bh2x4 ROW_MAJOR set (idx 31, 32, 34 run one M-block per core: no ring, the base program).
FLUX_REST = {
    "idx31": (66, 128, 32, 128, (0, 0, 1, 1)),
    "idx32": (130, 256, 32, 128, (0, 0, 1, 1)),
    "idx33": (514, 1024, 128, 3, (0, 0, 1, 1)),
    "idx34": (66, 128, 128, 512, (0, 0, 1, 1)),
    "idx37": (258, 512, 64, 256, (0, 0, 1, 1)),
    "idx38": (130, 256, 128, 512, (0, 0, 1, 1)),
    "idx39": (514, 1024, 64, 128, (0, 0, 1, 1)),
    "idx40": (258, 512, 128, 256, (0, 0, 1, 1)),
    "idx45": (514, 1024, 128, 256, (0, 0, 1, 1)),
}
# Correctness regimes (ROW_MAJOR interleaved): symmetric padding, W not a multiple of 32 (tile-rows straddle output
# rows), dilation 2, window units (bk = 3 * c_in_tiles), one M-block per core, wide C_out.
DOMAIN = {
    "s128x100_c64_p1": (128, 100, 64, 64, 1),
    "s64x100_c32_p1": (64, 100, 32, 64, 1),
    "s64x200_c64_d2p2": (64, 200, 64, 32, 2),
    "s258x512_c32_p1": (258, 512, 32, 128, 1),
    "s130x1024_c128_p1": (130, 1024, 128, 64, 1),
    "s98x96_c64_p0011": (98, 96, 64, 64, (0, 0, 1, 1)),
}
DILATION = {"s64x200_c64_d2p2": 2}

_OFF = dict(
    RM_RING=False,
    RM_RING_OVERLAP=True,
    RM_RING_IN_SOLVE=True,
    RM_RING_KEEP_BK=True,
    RM_RING_MIN_FREE=128,
    RM_RING_CHUNK=8,
)
VARIANTS = {
    "base": dict(_OFF),
    # Recommended: overlapped fill, 8-pixel steps, ring counted in the L1 solve.
    "ring": dict(_OFF, RM_RING=True),
    "ring_ch32": dict(_OFF, RM_RING=True, RM_RING_CHUNK=32),
    "ring_serial": dict(_OFF, RM_RING=True, RM_RING_OVERLAP=False),
    "ring_fit": dict(_OFF, RM_RING=True, RM_RING_IN_SOLVE=False),
    # The ring may also shrink bk (not bit-identical where it does: the K-block count sets the fp32 accumulation order).
    "ring_anybk": dict(_OFF, RM_RING=True, RM_RING_KEEP_BK=False),
    # In-flight bound: up to 255 - RM_RING_MIN_FREE fill reads per core.
    "ring_mf64": dict(_OFF, RM_RING=True, RM_RING_MIN_FREE=64, RM_RING_CHUNK=32),
    "ring_mf32": dict(_OFF, RM_RING=True, RM_RING_MIN_FREE=32, RM_RING_CHUNK=32),
    "ring_serial_mf32": dict(_OFF, RM_RING=True, RM_RING_OVERLAP=False, RM_RING_MIN_FREE=32, RM_RING_CHUNK=32),
    "ring_mf32_ch16": dict(_OFF, RM_RING=True, RM_RING_MIN_FREE=32, RM_RING_CHUNK=16),
    # Pixels per opportunistic fill step (the reader returns to the unit FIFO between steps).
    "ring_ch16": dict(_OFF, RM_RING=True, RM_RING_CHUNK=16),
    "ring_ch4": dict(_OFF, RM_RING=True, RM_RING_CHUNK=4),
}


def _variant_knobs(name):
    bm_cap = 0
    if "_bm" in name:
        name, cap = name.split("_bm")
        bm_cap = int(cap)
    return VARIANTS[name], bm_cap


def _base_of(variant):
    """The base variant a ring variant must match bit for bit: same bm cap."""
    return "base" + (variant[variant.index("_bm") :] if "_bm" in variant else "")


def _selected(env, default):
    v = os.environ.get(env)
    return [x for x in v.split(",") if x] if v else default


LAST = {}


def apply_variant(variant, monkeypatch):
    knobs, bm_cap = _variant_knobs(variant)
    for k, v in knobs.items():
        monkeypatch.setattr(pd, k, v)
    pd.select_blocking.cache_clear()
    pd.rm_ring_plan.cache_clear()
    if bm_cap:
        monkeypatch.setattr(op_mod, "select_blocking", lambda g, grid, budget: _capped(g, grid, budget, bm_cap))
    real_cpd = pd.create_program_descriptor

    def capture(*args, geometry, blocking, **kwargs):
        LAST.clear()
        LAST.update(
            bm=blocking.block_m_tiles,
            bk=blocking.block_k_tiles,
            nb=blocking.max_m_blocks_per_core,
            slab_px=blocking.slab_px,
            rm_ring_px=blocking.rm_ring_px,
            im2col_depth=blocking.im2col_rm_depth_rows,
            foot_kb=round(blocking.footprint_bytes / 1024, 1),
        )
        return real_cpd(*args, geometry=geometry, blocking=blocking, **kwargs)

    monkeypatch.setattr(op_mod, "create_program_descriptor", capture)


def _capped(g, grid, l1_budget, bm_cap):
    """select_blocking with every bm candidate above bm_cap dropped (same solve otherwise)."""
    core_m_tiles, groups = pd.m_split(g, grid[0] * grid[1])
    bn = g.tensor_n_tiles
    kkh = g.tensor_kkh_tiles
    tap_floor = g.c_in_p // 32 if g.c_in_p % 32 == 0 else kkh
    bks = [d for d in range(kkh, 0, -1) if kkh % d == 0 and d >= tap_floor]
    if pd.tap_major_capable(g):
        bks = [d for d in bks if d % g.c_in_tiles == 0]
    for bm in range(bm_cap, 0, -1):
        for bk in bks:
            fit = pd._footprint(g, core_m_tiles, bm, bk, bn)
            rm_px = pd.rm_ring_plan(g, core_m_tiles, bm) if pd.rm_ring_capable(g, bm, bk) else None
            if rm_px:
                slab_px = 2 * fit["band_stride_px"] + fit["band_len_px"]
                delta = (rm_px - slab_px) * g.c_in_p * 2
                if fit["footprint_bytes"] + delta > l1_budget:
                    continue
                fit = dict(fit, rm_ring_px=rm_px, footprint_bytes=fit["footprint_bytes"] + delta)
            if fit["footprint_bytes"] <= l1_budget:
                return pd.Conv3x3Blocking(
                    grid_x=grid[0],
                    core_m_tiles=core_m_tiles,
                    num_m_groups=groups,
                    block_m_tiles=bm,
                    block_k_tiles=bk,
                    block_n_tiles=bn,
                    out_depth_row_groups=pd.OUT_DEPTH_ROW_GROUPS,
                    **fit,
                )
    raise ValueError("no capped blocking")


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


def _pcc(out_flat, x_hwc, weight, bias, pad, d, h_out, w_out, c_out):
    import torch

    out = out_flat.reshape(-1, out_flat.shape[-1])[: h_out * w_out, :c_out].reshape(h_out, w_out, c_out)
    got, ref = [], []
    for r0, r1 in _bands(h_out):
        got.append(out[r0:r1].double().flatten())
        ref.append(_ref_rows(x_hwc, weight, bias, pad, d, r0, r1).double().flatten())
    got, ref = torch.cat(got), torch.cat(ref)
    return torch.corrcoef(torch.stack([got, ref]))[0, 1].item()


def _in_collect_pass():
    plugin = sys.modules.get("tests.plugins.up_front_collect")
    return plugin is not None and getattr(plugin, "_PASS", "real") == "collect"


ALL_CASES = {**FOCUS, **FLUX_REST, **DOMAIN}
CASE_IDS = _selected("RMR_CASES", list(FOCUS))
VARIANT_IDS = _selected("RMR_VARIANTS", ["base", "ring"])


@pytest.mark.parametrize("device_params", [{"l1_small_size": 65536}], indirect=True)
@pytest.mark.parametrize("case_id", CASE_IDS)
def test_rm_ring(device, case_id, monkeypatch):
    import torch

    H, W, c_in, c_out, pad = ALL_CASES[case_id]
    d = DILATION.get(case_id, 1)
    collect = _in_collect_pass()
    h_out, w_out = _out_hw(H, W, pad, d)
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
        padding=pad,
        dilation=d,
        input_dtype=ttnn.bfloat16,
        input_layout=ttnn.ROW_MAJOR_LAYOUT,
        weight_dtype=ttnn.bfloat16,
        input_memory_config=tt_in.memory_config(),
        device=device,
    )
    tt_w = opc.prepare_conv2d_weights(weight, **prep)
    tt_b = opc.prepare_conv2d_bias(bias, **prep)
    cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False
    )
    outs = {}
    for variant in VARIANT_IDS:
        with monkeypatch.context() as mp:
            apply_variant(variant, mp)

            def run():
                return opc.conv2d_3x3(
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

            t0 = time.time()
            out = run()
            ttnn.synchronize_device(device)
            ttnn.ReadDeviceProfiler(device)
            for i in range(MEASURED_ITERS):
                ttnn.deallocate(out)
                signpost(header=f"cmp|{case_id}|{variant}|{i}")
                out = run()
                ttnn.synchronize_device(device)
                ttnn.ReadDeviceProfiler(device)
            signpost(header=f"cmpend|{case_id}|{variant}")
            blocking = dict(LAST)
            if collect:
                ttnn.deallocate(out)
                continue
            got = ttnn.to_torch(out)
            ttnn.deallocate(out)
            outs[variant] = got
            rec = dict(case=case_id, variant=variant, shape=[H, W, c_in, c_out], pad=pad, dilation=d, **blocking)
            rec["pcc"] = _pcc(got, x, weight, bias, pad, d, h_out, w_out, c_out)
            ref_v = _base_of(variant)
            if ref_v in outs and variant != ref_v:
                rec["identical_to_base"] = bool(torch.equal(got, outs[ref_v]))
                if not rec["identical_to_base"]:
                    diff = (got.float() - outs[ref_v].float()).abs()
                    rec["max_abs_diff"] = diff.max().item()
                    rec["n_diff"] = int((diff > 0).sum().item())
            rec["wall_s"] = round(time.time() - t0, 1)
            rec["tag"] = os.environ.get("RMR_TAG", "")
            rec["ablate"] = os.environ.get("CONV2D_3X3_ABLATE", "")
            with open(RESULTS, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print("RMR:", json.dumps(rec))
    ttnn.deallocate(tt_in)
    if not collect and os.environ.get("CONV2D_3X3_ABLATE", "") == "":
        for variant, got in outs.items():
            ref_v = _base_of(variant)
            if variant != ref_v and ref_v in outs:
                assert torch.equal(got, outs[ref_v]), f"{case_id}: {variant} differs from {ref_v}"
