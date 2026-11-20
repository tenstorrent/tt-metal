# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Bake-off (perf 16, `weights_resident`): weights resident across M-blocks on the interleaved K-blocked walk.

Packages (copies of the op, loaded as private packages; the dirs have no __init__.py):
  op_base/  conv2d_3x3 at the round-16 baseline (HEAD 5ac29b03f50 + the working tree's Refinement 7 WIP), untouched
  op_copy/  the same + WR_MODE / WR_NOACK (conv2d_3x3_program_descriptor.py) and the WEIGHTS_RESIDENT kernel paths

Variants (name -> package, WR_MODE, WR_NOACK):
  base       op_base                                  the current per-K-block weight rounds (cb_weights depth 2)
  off        op_copy, "off"                           must equal base (same blocking, same kernels)
  lo         op_copy, "leftover", acked rounds        resident only where the L1 left at the streaming solve fits it
  lo_na      op_copy, "leftover", no acks (Counter)
  trim       op_copy, "trim", acked rounds            resident at the streaming bm / bk (may take im2col ring depth)
  trim_na    op_copy, "trim", no acks
  solve      op_copy, "solve", acked rounds           resident counted in the solve (bm shrinks for it)
  solve_na   op_copy, "solve", no acks
  base_abl / solve_abl   op_base_abl / op_copy_abl: base / solve + the ablation_r16 reader fill hook (diagnostic only)

The FLUX.2 bh2x4 per-device convs, exactly as test_flux_conv3x3_compare.py runs conv2d_3x3: DRAM-interleaved ROW_MAJOR
bf16 [1, 1, H*W, C_in] input, padding (0, 0, 1, 1), bf16 weights / bias, HiFi4, fp32 DEST, math_approx_mode False.
Inputs seeded as the harness (torch.manual_seed(0)). Per variant: one warm-up call, WR_ITERS signposted calls
(`cmp|<case>|<variant>|<i>`, cmp_windows.py-compatible). Every variant's output is hashed and compared bit-for-bit
with base's; base's (and any non-identical variant's) PCC is against a fp32 torch conv on bands of output rows.

Env: WR_CASES (comma list of keys of CASES), WR_VARIANTS (comma list, run in that order), WR_ITERS (default 1),
WR_LOG (jsonl, default runs/wr.jsonl).
"""

import hashlib
import importlib
import json
import os
import sys
import types
from pathlib import Path

import pytest
import ttnn
from tracy import signpost

HERE = Path(__file__).resolve().parent
pytestmark = pytest.mark.use_module_device({"l1_small_size": 65536})


def _pkg(sub):
    name = "wr_" + sub
    if name not in sys.modules:
        pkg = types.ModuleType(name)
        pkg.__path__ = [str(HERE / sub)]
        sys.modules[name] = pkg
    return (
        importlib.import_module(f"{name}.conv2d_3x3"),
        importlib.import_module(f"{name}.conv2d_3x3_program_descriptor"),
    )


PKGS = {k: _pkg(k) for k in ("op_base", "op_copy", "op_base_abl", "op_copy_abl") if (HERE / k).exists()}
for _k, (_o, _p) in PKGS.items():
    assert Path(_p.KERNEL_DIR).parent.name == _k, _p.KERNEL_DIR

VARIANTS = {
    "base": ("op_base", None, None),
    "off": ("op_copy", "off", False),
    "lo": ("op_copy", "leftover", False),
    "lo_na": ("op_copy", "leftover", True),
    "trim": ("op_copy", "trim", False),
    "trim_na": ("op_copy", "trim", True),
    "solve": ("op_copy", "solve", False),
    "solve_na": ("op_copy", "solve", True),
    # Composition diagnostics: the same with the reader's ROW_MAJOR fill hook of ablation_r16 (CONV2D_3X3_ABLATE=fill in
    # the session's env stubs the DRAM reads; outputs are wrong by construction and not compared).
    "base_abl": ("op_base_abl", None, None),
    "solve_abl": ("op_copy_abl", "solve", False),
    # Diagnostics: streaming with the deeper im2col ring capped at n tile-rows (the depth "solve" leaves it).
    **{f"off_cap{n}": ("op_copy", "off", False, dict(WR_IM2COL_CAP=n)) for n in (2, 4, 6, 8, 12, 15, 38)},
}

# FLUX_CMP_CASES index -> (H, W, C_in, C_out[, layout, padding, stride]); default ROW_MAJOR, padding (0, 0, 1, 1),
# stride 1 (the bh2x4 per-device convs). "_T" = the same conv on TILE input; "1x1_*" = FLUX.2 single-device convs (TILE).
RM, TL = "RM", "TILE"
CASES = {
    "idx41": (1026, 2048, 32, 128),
    "idx42": (514, 1024, 64, 256),
    "idx35": (1026, 2048, 128, 3),
    "idx36": (514, 1024, 32, 128),
    "idx44": (1026, 2048, 64, 128),
    "idx46": (1026, 2048, 64, 256),
    "idx43": (258, 512, 128, 512),
    "idx47": (514, 1024, 128, 512),
    # Smaller bh2x4 convs of the same decoder (domain sweep).
    "c66x128": (66, 128, 32, 128),
    "c130x256": (130, 256, 32, 128),
    "c258x512_64": (258, 512, 64, 256),
    "c258x512_128_256": (258, 512, 128, 256),
    "c514x1024_128_3": (514, 1024, 128, 3),
    # TILE input.
    "idx41_T": (1026, 2048, 32, 128, TL),
    "idx42_T": (514, 1024, 64, 256, TL),
    "c258x512_128_256_T": (258, 512, 128, 256, TL),
    "1x1_1024_128_3": (1024, 1024, 128, 3, TL, 1),
    "1x1_512_128_256": (512, 512, 128, 256, TL, 1),
    "1x1_1024_128_128": (1024, 1024, 128, 128, TL, 1),
    "1x1_1024_256_128": (1024, 1024, 256, 128, TL, 1),
    "1x1_1024_128_128_s2": (1024, 1024, 128, 128, TL, (0, 1, 0, 1), 2),
}
BAND_ROWS = 32
ITERS = int(os.environ.get("WR_ITERS", "1"))
SEL = [c for c in os.environ.get("WR_CASES", "idx41").split(",") if c]
VAR_ORDER = [v for v in os.environ.get("WR_VARIANTS", "base,off,lo").split(",") if v]
LOG = HERE / "runs" / os.environ.get("WR_LOG", "wr.jsonl")
# Contract knobs for the domain sweep (every variant of the session): bf8w (bfloat8_b weights), nobias, relu.
OPTS = {o for o in os.environ.get("WR_OPTS", "").split(",") if o}


def _in_collect_pass():
    plugin = sys.modules.get("tests.plugins.up_front_collect")
    return plugin is not None and getattr(plugin, "_PASS", "real") == "collect"


def _bands(h_out):
    if h_out <= 3 * BAND_ROWS:
        return [(0, h_out)]
    mid = h_out // 2 - BAND_ROWS // 2
    return [(0, BAND_ROWS), (mid, mid + BAND_ROWS), (h_out - BAND_ROWS, h_out)]


def _sides(pad):
    return (pad,) * 4 if isinstance(pad, int) else pad


def _ref_rows(x_hwc, weight, bias, r0, r1, pad, stride):
    import torch

    top, bottom, left, right = _sides(pad)
    h = x_hwc.shape[0]
    first = r0 * stride - top
    last = (r1 - 1) * stride - top + 3
    rows = x_hwc[max(first, 0) : min(last, h)].float().permute(2, 0, 1).unsqueeze(0)
    rows = torch.nn.functional.pad(rows, (left, right, max(0, -first), max(0, last - h)))
    return torch.nn.functional.conv2d(rows, weight.float(), bias.float(), stride=stride)[0].permute(1, 2, 0)


def _pcc_bands(out, x_hwc, weight, bias, h_out, w_out, c_out, pad, stride, relu=False):
    import torch

    out = out.reshape(-1, out.shape[-1])[: h_out * w_out, :c_out].reshape(h_out, w_out, c_out)
    got, ref = [], []
    for r0, r1 in _bands(h_out):
        got.append(out[r0:r1].double().flatten())
        r = _ref_rows(x_hwc, weight, bias, r0, r1, pad, stride)
        ref.append((torch.relu(r) if relu else r).double().flatten())
    got, ref = torch.cat(got), torch.cat(ref)
    return torch.corrcoef(torch.stack([got, ref]))[0, 1].item()


def _set_variant(variant):
    pkg, mode, noack, *extra = VARIANTS[variant]
    op, pd = PKGS[pkg]
    if mode is not None:
        pd.WR_MODE = mode
        pd.WR_NOACK = noack
        pd.WR_IM2COL_CAP = 1 << 30
        for k, v in (extra[0] if extra else {}).items():
            setattr(pd, k, v)
    for name in dir(pd):
        m = getattr(pd, name)
        if callable(m) and hasattr(m, "cache_clear"):
            m.cache_clear()
    return op, pd


@pytest.mark.parametrize("case", SEL)
def test_wr(case, device):
    import torch

    collect = _in_collect_pass()
    H, W, c_in, c_out, layout, pad, stride = (CASES[case] + (RM, (0, 0, 1, 1), 1)[len(CASES[case]) - 4 :])[:7]
    top, bottom, left, right = _sides(pad)
    h_out, w_out = (H + top + bottom - 3) // stride + 1, (W + left + right - 3) // stride + 1
    in_layout = ttnn.TILE_LAYOUT if layout == TL else ttnn.ROW_MAJOR_LAYOUT
    torch.manual_seed(0)
    x = torch.zeros(H, W, c_in, dtype=torch.bfloat16) if collect else torch.randn(H, W, c_in, dtype=torch.bfloat16)
    weight = (torch.randn(c_out, c_in, 3, 3) / (9 * c_in) ** 0.5).to(torch.bfloat16)
    bias = (0.1 * torch.randn(c_out)).to(torch.bfloat16)
    base_op, _ = PKGS["op_base"]
    tt_x = ttnn.from_torch(
        x.reshape(1, 1, H * W, c_in),
        dtype=ttnn.bfloat16,
        layout=in_layout,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    prep = dict(
        stride=stride,
        padding=pad,
        dilation=1,
        input_dtype=ttnn.bfloat16,
        input_layout=in_layout,
        weight_dtype=ttnn.bfloat8_b if "bf8w" in OPTS else ttnn.bfloat16,
        input_memory_config=tt_x.memory_config(),
        device=device,
    )
    tt_w = base_op.prepare_conv2d_weights(weight, **prep)
    if "nobias" in OPTS:
        bias = torch.zeros_like(bias)
        tt_b = None
    else:
        tt_b = base_op.prepare_conv2d_bias(bias, **prep)
    cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False
    )

    base_sha = None
    for variant in VAR_ORDER:
        op, pd = _set_variant(variant)
        seen = {}
        orig = op.select_blocking

        def recording(g, grid, budget, _orig=orig):
            b = _orig(g, grid, budget)
            seen.update(
                bm=b.block_m_tiles,
                bk=b.block_k_tiles,
                nkb=b.num_k_blocks,
                m_blocks=b.max_m_blocks_per_core,
                core_m=b.core_m_tiles,
                weight_depth=b.weight_depth,
                resident=getattr(b, "weights_resident", False),
                footprint=b.footprint_bytes,
                budget=budget,
                band_len=b.band_len_px,
                im2col=b.im2col_rm_depth_rows,
            )
            return b

        op.select_blocking = recording
        try:

            def run():
                return op.conv2d_3x3(
                    tt_x,
                    tt_w,
                    input_height=H,
                    input_width=W,
                    bias=tt_b,
                    padding=pad,
                    stride=stride,
                    activation="relu" if "relu" in OPTS else None,
                    compute_kernel_config=cfg,
                )

            out = run()
            ttnn.synchronize_device(device)
            ttnn.ReadDeviceProfiler(device)
            for i in range(ITERS):
                ttnn.deallocate(out)
                signpost(header=f"cmp|{case}|{variant}|{i}")
                out = run()
                ttnn.synchronize_device(device)
                ttnn.ReadDeviceProfiler(device)
            signpost(header=f"cmpend|{case}|{variant}")
        finally:
            op.select_blocking = orig
        if collect:
            ttnn.deallocate(out)
            continue
        got = ttnn.to_torch(out)
        ttnn.deallocate(out)
        sha = hashlib.sha256(got.contiguous().view(torch.int16).numpy().tobytes()).hexdigest()[:16]
        rec = dict(case=case, variant=variant, sha=sha, opts=sorted(OPTS), **seen)
        if variant == "base" or base_sha is None:
            base_sha = base_sha or (sha if variant == "base" else None)
        rec["identical_to_base"] = (sha == base_sha) if base_sha else None
        if variant == "base" or not rec["identical_to_base"]:
            rec["pcc"] = _pcc_bands(got, x, weight, bias, h_out, w_out, c_out, pad, stride, relu="relu" in OPTS)
        print("WR:", json.dumps(rec))
        LOG.parent.mkdir(exist_ok=True)
        with open(LOG, "a") as f:
            f.write(json.dumps(rec) + "\n")
    ttnn.deallocate(tt_x)
    ttnn.deallocate(tt_w)
    if tt_b is not None:
        ttnn.deallocate(tt_b)
