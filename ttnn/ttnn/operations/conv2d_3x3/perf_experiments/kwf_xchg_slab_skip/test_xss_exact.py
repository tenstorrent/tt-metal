# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Bit-exactness of the dead-slab-row skip (op_copy/) against HEAD (op_base/), in one process (no profiler).

The skipped slab slots stay unwritten, so a stale slab from an identical earlier run would hide a wrong skip. Each
check therefore runs op_base first on a POISON input (another seed, x64) of the same config, which writes every slab
slot (dead ones included) at the same L1 addresses, then op_copy on the real input (a dead slot it wrongly reads now
holds poison), then op_base on the real input (the reference). Covered:
- every kw-fold [perf] focus case, ReLU on;
- the forced kw-fold unit shapes and the halo-exchange unit shapes (tests/.../test_conv2d_3x3_kw_fold.py) with the
  exchange on (one M-block per core) and off (every halo recomputed), plus extra multi-block shapes whose block
  boundaries fall inside / on image rows.
Each record also logs the kw-fold blocking taken and the host mirror's filled vs live slab rows. Appends exact.jsonl.

    scripts/run_safe_pytest.sh --run-all --import-mode=prepend \
        ttnn/ttnn/operations/conv2d_3x3/perf_experiments/kwf_xchg_slab_skip/test_xss_exact.py
"""

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from xss_copy import copy_modules  # noqa: E402

KWF_LAYERS = {
    "stem.conv2": "stem_conv2",
    "stem.conv3": "stem_conv3",
    "res2.{0,1,2}.conv2": "res2",
    "res3.{1,2,3}.conv2": "res3_x",
    "semantic decoder.res3.fuse_conv.0": "sem_dec_res3_fc0",
    "instance decoder.res3.fuse_conv.0": "inst_dec_res3_fc0",
    "semantic decoder.res3.fuse_conv.1": "sem_dec_res3_fc1",
    "semantic decoder.res2.fuse_conv.0": "sem_dec_res2_fc0",
    "instance decoder.res2.fuse_conv.0": "inst_dec_res2_fc0",
    "semantic res2.fuse_conv.1": "sem_res2_fc1",
    "instance res2.fuse_conv.1": "inst_res2_fc1",
    "center_head.1": "center_head_1",
}
OUT = Path(__file__).resolve().parent / "exact.jsonl"


def _cases():
    from eval.golden_tests.conv2d_3x3.feature_spec import LOOSE_CASES

    out, ids = [], []
    for c in LOOSE_CASES:
        layer = c["extras"]["layer"]
        for k, tag in KWF_LAYERS.items():
            if "[perf]" in layer and layer.startswith(k):
                out.append(c)
                ids.append(tag)
    return out, ids


_CASES, _IDS = _cases()


def _slab_rows(H, W, d, per_core, bm, xchg):
    """Host mirror of the fill: (slab rows filled, live rows) summed over every core and M-block."""
    tpr, m_tiles = W // 32, H * W // 32
    r = d * tpr
    filled = live = 0
    for t_core in range(0, m_tiles, per_core):
        core_tiles = min(per_core, m_tiles - t_core)
        for t0 in range(t_core, t_core + core_tiles, bm):
            t_end = min(t0 + bm, t_core + core_tiles)
            ht = not xchg and t0 % tpr != 0
            hb = not xchg and t_end % tpr != 0 and t_end < m_tiles
            band = set()
            used = set()
            for kh in range(3):
                band |= {x for x in range(t0 - 1 + (kh - 1) * r, t_end + 1 + (kh - 1) * r) if 0 <= x < m_tiles}
                used |= {x for x in range(t0 - ht + (kh - 1) * r, t_end + hb + (kh - 1) * r) if 0 <= x < m_tiles}
            assert used <= band
            filled += len(band)
            live += len(used)
    return filled, live


def _run_three(build_input, conv_kwargs, prep_kwargs, tw, tb, force=None, ref="base"):
    """The reference (op_base, or op_basefix where HEAD hangs) on the poison input, op_copy then the reference on the
    real one. Returns (outs, kw-fold plan per copy)."""
    import ttnn

    outs, plans = {}, {}
    for name, poison in (("base", True), ("copy", False), ("base", False)):
        # XSS_COPY=neg: a deliberately wrong skip (op_neg/), the gate's negative control.
        mod, pd = copy_modules(os.environ.get("XSS_COPY", "copy") if name == "copy" else ref)
        taken = []
        saved = {k: getattr(pd, k) for k in ("kw_fold_plan", "_kw_fold_pays", "kw_fold_halo_exchange")}
        orig = pd.kw_fold_plan

        def spy(*a, _orig=orig, _taken=taken, **k):
            r = _orig(*a, **k)
            _taken.append(r)
            return r

        pd.kw_fold_plan = spy
        if force is not None:
            pd._kw_fold_pays = lambda *a: True
            pd.kw_fold_halo_exchange = lambda b, _x=force: _x and b.max_m_blocks_per_core == 1
        try:
            y = mod.conv2d_3x3(
                build_input(poison),
                ttnn.operations.conv2d_3x3.prepare_conv2d_weights(tw, **prep_kwargs),
                bias=ttnn.operations.conv2d_3x3.prepare_conv2d_bias(tb, **prep_kwargs),
                **conv_kwargs,
            )
            xchg = pd.kw_fold_halo_exchange
            kb = next((p for p in taken if p is not None), None)
            plan = None if kb is None else (kb.block_m_tiles, kb.max_m_blocks_per_core, bool(xchg(kb)))
        finally:
            for k, v in saved.items():
                setattr(pd, k, v)
        if not poison:
            plans[name] = plan
            outs[name] = ttnn.to_torch(y).float()
        ttnn.deallocate(y)
    return outs, plans


def _record(rec, outs, ref, c_out):
    import torch

    for name, o in outs.items():
        a, b = o.flatten(), ref.flatten()
        rec[name] = {
            "pcc": round(torch.corrcoef(torch.stack([a, b]))[0, 1].item(), 6),
            "mae": round((a - b).abs().mean().item(), 5),
        }
    d = (outs["copy"] - outs["base"]).reshape(-1, c_out)
    rec["identical_frac"] = round((d == 0).float().mean().item(), 6)
    rec["max_base_copy_diff"] = round(d.abs().max().item(), 5)
    if rec["identical_frac"] < 1:
        rec["first_bad"] = (d != 0).nonzero()[:6].tolist()
    with OUT.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec))


def _rows(H, W, d, per_core, plan):
    if plan is None:
        return None
    bm, _, xchg = plan
    filled, live = _slab_rows(H, W, d, per_core, bm, xchg)
    return {"filled": filled, "live": live, "dead_frac": round(1 - live / filled, 4)}


@pytest.mark.parametrize("case", _CASES, ids=_IDS)
def test_xss_exact_focus(case, device):
    import torch
    import ttnn
    from eval.golden_tests.conv2d_3x3 import helpers

    torch.manual_seed(0)
    (H, W, c_in), c_out, padding, stride, dilation = case["inputs"]
    extras = case["extras"]
    dtype, weight_dtype, layout = case["dtype"], case["weight_dtype"], case["layout"]
    mc = helpers.build_input_memory_config(
        (1, 1, H * W, c_in), case["memory_layout"], layout=layout, dtype=dtype, device=device, extras=extras
    )
    prep = dict(
        stride=stride,
        padding=padding,
        dilation=dilation,
        input_dtype=dtype,
        input_layout=layout,
        weight_dtype=weight_dtype,
        input_memory_config=mc,
        device=device,
    )
    tw = helpers.make_weight(c_out, c_in, dtype=torch.bfloat16)
    tb = helpers.make_bias(c_out, dtype=torch.bfloat16)
    tx = helpers.make_activation(H, W, c_in, dtype=torch.bfloat16)
    gen = torch.Generator().manual_seed(1234)
    tp = (torch.randn(tx.shape, generator=gen) * 64).to(torch.bfloat16)
    xq = ttnn.to_torch(ttnn.from_torch(tx, dtype=dtype, layout=ttnn.TILE_LAYOUT)).float()
    wq = ttnn.to_torch(ttnn.from_torch(tw, dtype=weight_dtype, layout=ttnn.TILE_LAYOUT)).float()
    bq = ttnn.to_torch(ttnn.from_torch(tb.reshape(1, c_out), dtype=weight_dtype, layout=ttnn.TILE_LAYOUT)).float()[0]
    ref = torch.nn.functional.conv2d(
        xq.reshape(1, H, W, c_in).permute(0, 3, 1, 2), wq, bias=bq, stride=stride, padding=padding, dilation=dilation
    )
    ref = torch.relu(ref).permute(0, 2, 3, 1).reshape(1, 1, -1, c_out)
    conv = dict(
        input_height=H,
        input_width=W,
        padding=padding,
        stride=stride,
        dilation=dilation,
        activation="relu",
        compute_kernel_config=helpers.make_compute_kernel_config(case["fp32_dest_acc_en"], extras["math_fidelity"]),
    )
    outs, plans = _run_three(
        lambda poison: ttnn.from_torch(
            tp if poison else tx, dtype=dtype, layout=layout, device=device, memory_config=mc
        ),
        conv,
        prep,
        tw,
        tb,
    )
    outs = {k: v.reshape(ref.shape) for k, v in outs.items()}
    per_core = extras["shard_shape"][0] // 32
    rec = {"case": extras["layer"], "plan": plans, "rows": _rows(H, W, dilation, per_core, plans["copy"])}
    _record(rec, outs, ref, c_out)
    assert plans["base"] is not None and plans["copy"] is not None, rec
    assert rec["identical_frac"] == 1.0, rec


def _unit_cases():
    from tests.ttnn.unit_tests.operations.conv2d_3x3 import test_conv2d_3x3_kw_fold as u

    # Multi-block shapes (exchange off by construction) whose block boundaries fall inside image rows (W 96: 3 tile-rows
    # per image row) or on them, a ragged last core, d 2 / 3.
    extra = [
        ("mb_w96_mid_row", 128, 96, 64, 64, 1, "bf8", "bf8", "LoFi", True, 4),
        ("mb_w96_mid_row_d2", 128, 96, 64, 64, 2, "bf8", "bf8", "LoFi", True, 4),
        ("mb_ragged_w64", 96, 64, 64, 64, 1, "bf8", "bf8", "LoFi", True, 5),
        ("n2_one_row_cores", 32, 64, 64, 64, 1, "bf8", "bf8", "LoFi", True, 64),
        ("n2_d3", 48, 96, 64, 64, 3, "bf16", "bf8", "LoFi", True, 36),
        ("kblocked_n2_multiblock", 64, 128, 256, 64, 1, "bf8", "bf8", "LoFi", True, 8),
    ]
    cases = (
        [("forced", c, True) for c in u.CASES]
        + [("xchg", c, True) for c in u.XCHG_CASES]
        + [("recompute", c, False) for c in u.XCHG_CASES]
        + [("extra", c, True) for c in extra]
    )
    return cases, [f"{k}-{c[0]}" for k, c, _ in cases]


_UNIT, _UNIT_IDS = _unit_cases()


@pytest.mark.parametrize("kind,case,xchg", _UNIT, ids=_UNIT_IDS)
def test_xss_exact_unit(kind, case, xchg, device):
    import torch
    import ttnn

    _, H, W, c_in, c_out, d, act, wdt, fidelity, relu, cores = case
    DT = {"bf8": ttnn.bfloat8_b, "bf16": ttnn.bfloat16}
    dtype, weight_dtype = DT[act], DT[wdt]
    grid = device.compute_with_storage_grid_size()
    m_tiles = H * W // 32
    per_core = -(-m_tiles // cores)
    cores = -(-m_tiles // per_core)
    if cores > grid.x * grid.y:
        pytest.skip("grid")
    mc = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            ttnn.num_cores_to_corerangeset(cores, grid, row_wise=True),
            [per_core * 32, c_in],
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )
    torch.manual_seed(0)
    tx = torch.randn(1, 1, H * W, c_in).to(torch.bfloat16)
    tp = (torch.randn(1, 1, H * W, c_in) * 64).to(torch.bfloat16)
    tw = (torch.randn(c_out, c_in, 3, 3) / (3 * c_in**0.5)).to(torch.bfloat16)
    tb = torch.randn(c_out).to(torch.bfloat16)
    prep = dict(
        stride=1,
        padding=d,
        dilation=d,
        input_dtype=dtype,
        input_layout=ttnn.TILE_LAYOUT,
        weight_dtype=weight_dtype,
        input_memory_config=mc,
        device=device,
    )
    conv = dict(
        input_height=H,
        input_width=W,
        padding=d,
        stride=1,
        dilation=d,
        activation="relu" if relu else None,
        compute_kernel_config=ttnn.init_device_compute_kernel_config(
            device.arch(), math_fidelity=getattr(ttnn.MathFidelity, fidelity), fp32_dest_acc_en=True
        ),
    )
    x = tx.float().reshape(1, H, W, c_in).permute(0, 3, 1, 2)
    ref = torch.nn.functional.conv2d(x, tw.float(), tb.float(), padding=d, dilation=d)
    if relu:
        ref = torch.relu(ref)
    ref = ref.permute(0, 2, 3, 1).reshape(1, 1, -1, c_out)
    outs, plans = _run_three(
        lambda poison: ttnn.from_torch(
            tp if poison else tx, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc
        ),
        conv,
        prep,
        tw,
        tb,
        force=xchg,
        # HEAD's early writer fill deadlocks where a core's first M-block has no writer batch (see op_basefix/).
        ref="basefix" if kind == "extra" else "base",
    )
    outs = {k: v.reshape(ref.shape) for k, v in outs.items()}
    rec = {
        "case": f"{kind}-{case[0]}",
        "ref": "basefix" if kind == "extra" else "base",
        "plan": plans,
        "rows": _rows(H, W, d, per_core, plans["copy"]),
    }
    _record(rec, outs, ref, c_out)
    assert plans["base"] is not None and plans["copy"] is not None, rec
    assert rec["identical_frac"] == 1.0, rec
