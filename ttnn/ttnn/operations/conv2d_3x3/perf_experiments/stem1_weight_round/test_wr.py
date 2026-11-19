# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Bake-off: land stem.conv1's slab-direct weight round earlier (perf experiment, round 13, idea `stem1_weight_round`).

Variants pick a package and knobs: `base` = op_base_uz/ (HEAD 4f2f03f6d20, zones stripped), `cand` = op_copy_uz/ (the
candidate, zones stripped), `base_z` / `cand_z` the zoned op_base/ / op_copy/ (never in one process with each other:
zone-hash collisions), plus knob variants of the candidate (VARIANTS below).

Env: WR_CASES (comma list of focus ids, e.g. `stem_conv1`, or generic specs
`g:<mem>:<H>x<W>x<Ci>x<Co>:s<stride>:p<pad>:d<dil>:<bias>:<act>[:<fid>[:<wdt>]]` with mem in {dram, height}, bias in
{b, nb}, act in {relu, none}, fid in {lofi, hifi4}, wdt in {bf8, bf16}; generic cases use bf16 RM activation, fp32
DEST), WR_VARIANTS (comma list, run in that order per case), WR_ORDER=alt (reverse the order on every other case),
WR_LOG (run-log name). One dispatch per test, for `run_safe_pytest.sh --profile`; each dispatch appends a JSON record
(blocking, PCC, output hash) to runs/<log>.jsonl, which report.py joins with the profiler CSV's GenericOp rows in call
order. The precision contract is fixed per case: no variant touches it.
"""

import hashlib
import importlib
import json
import os
import time
from pathlib import Path

import pytest
import ttnn

from tests.ttnn.utils_for_testing import comp_pcc

HERE = Path(__file__).parent
_ROOT = "ttnn.operations.conv2d_3x3.perf_experiments.stem1_weight_round."
_MODS = {}


def _pkg(sub):
    if sub not in _MODS:
        o = importlib.import_module(_ROOT + sub + ".conv2d_3x3")
        p = importlib.import_module(_ROOT + sub + ".conv2d_3x3_program_descriptor")
        assert Path(p.KERNEL_DIR).parent == HERE / sub, p.KERNEL_DIR
        _MODS[sub] = (o, p)
    return _MODS[sub]


# name -> (package, descriptor knobs)
VARIANTS = {
    "base": ("op_base_uz", {}),
    "cand": ("op_copy_uz", {}),
    "base_z": ("op_base", {}),
    "cand_z": ("op_copy", {}),
    # options of the candidate
    "grad": ("op_grad_uz", {}),  # the graduation version (op_grad/, zones stripped)
    "grad_z": ("op_grad", {}),
    "cand_off": ("op_copy_uz", dict(SD_W0_POLL=False, SD_W0_TOP=False, SD_ONE_SHARD_WEIGHTS=False)),  # = base
    "cand_off_z": ("op_copy", dict(SD_W0_POLL=False, SD_W0_TOP=False, SD_ONE_SHARD_WEIGHTS=False)),
    "cand_p": ("op_copy_uz", dict(SD_W0_TOP=False, SD_ONE_SHARD_WEIGHTS=False)),  # polled round 0 only
    "cand_pt": ("op_copy_uz", dict(SD_ONE_SHARD_WEIGHTS=False)),  # + issued at kernel entry
    "cand_ps": ("op_copy_uz", dict(SD_W0_TOP=False)),  # + one-shard weights, issued where HEAD issues
    "cand_pt_z": ("op_copy", dict(SD_ONE_SHARD_WEIGHTS=False)),
    "cand_r1": ("op_copy_uz", dict(SD_MCAST_FIRST_RECT=True)),  # + multicast to the first rectangle only
    "cand_r1i": ("op_copy_uz", dict(SD_MCAST_FIRST_RECT=True, SD_ONE_SHARD_WEIGHTS=False)),
    "cand_r1_z": ("op_copy", dict(SD_MCAST_FIRST_RECT=True)),
    "cand_n0": ("op_copy_uz", dict(SD_R0_NOC0=True)),  # + round 0 multicast on NoC0 by the sender's reader
    "cand_n0i": ("op_copy_uz", dict(SD_R0_NOC0=True, SD_ONE_SHARD_WEIGHTS=False)),
    "cand_n0_z": ("op_copy", dict(SD_R0_NOC0=True)),
    "cand_n0_nt": ("op_copy_uz", dict(SD_R0_NOC0=True, SD_W0_TOP=False)),  # NoC0, round 0 issued where HEAD issues
    "cand_n0_min": ("op_copy_uz", dict(SD_R0_NOC0=True, SD_W0_TOP=False, SD_ONE_SHARD_WEIGHTS=False)),  # NoC0 only
    "v_pipe": ("op_copy_uz", dict(SD_R0_NOC0=False)),  # entry issue + sender-only polling, NoC1 pipe multicast
    "v_n0_nt": ("op_copy_uz", dict(SD_W0_TOP=False)),  # NoC0 multicast, round 0 issued where HEAD issues
    "cand_n0r": ("op_copy_uz", dict(SD_R0_NOC0=True, SD_W0_TOP_RECV=True)),  # receivers ack at kernel entry too
    "diag_n0_z": ("op_copy", dict(SD_R0_NOC0=True, SD_W0_DIAG=True)),
    "diag_z": ("op_copy", dict(SD_W0_DIAG=True)),
    # diagnostics (wrong output by design, WR_NOASSERT=1): no LoFi bias split / no weight multicast at all
    "dg_nosplit": ("op_copy_uz", dict(SD_W0_DIAG_NOSPLIT=True)),
    "dg_stub": ("op_copy_uz", dict(SD_W0_DIAG_STUB=True)),
    "dg_late": ("op_copy_uz", dict(SD_W0_DIAG_LATE=True)),  # correct output: round 0 multicast after the sender's slab
    "dg_late_z": ("op_copy", dict(SD_W0_DIAG_LATE=True)),
    "dg_nosplit_z": ("op_copy", dict(SD_W0_DIAG_NOSPLIT=True, SD_W0_DIAG=True)),
    "dg_stub_z": ("op_copy", dict(SD_W0_DIAG_STUB=True, SD_W0_DIAG=True)),
    "dg_stub_zz": ("op_copy", dict(SD_W0_DIAG_STUB=True)),
}
_EXTRA = os.environ.get("WR_KNOBS", "")
# WR_KNOBS="name=pkg:KNOB=val:KNOB=val;name2=..." adds ad-hoc knob variants (ints / True / False).
for _spec in [s for s in _EXTRA.split(";") if s]:
    _name, _rest = _spec.split("=", 1)
    _parts = _rest.split(":")
    _knobs = {}
    for _kv in _parts[1:]:
        _kk, _vv = _kv.split("=")
        _knobs[_kk] = {"True": True, "False": False}.get(_vv, None) if _vv in ("True", "False") else int(_vv)
    VARIANTS[_name] = (_parts[0], _knobs)


def apply_variant(variant, monkeypatch):
    sub, knobs = VARIANTS[variant]
    op, pd = _pkg(sub)
    for k, v in knobs.items():
        assert hasattr(pd, k), k
        monkeypatch.setattr(pd, k, v)
    pd.select_blocking.cache_clear()
    return op, pd


def _focus_cases():
    from eval.golden_tests.conv2d_3x3.feature_spec import LOOSE_CASES

    cases = [c for c in LOOSE_CASES if "[perf]" in c["extras"]["layer"]]
    ids = [c["extras"]["layer"].split(" ")[0].replace(".", "_") for c in cases]
    seen = {}
    out = {}
    for c, i in zip(cases, ids):
        if ids.count(i) > 1:
            n = seen.get(i, 0)
            seen[i] = n + 1
            i = f"{i}{n}"
        out[i] = c
    return out


FOCUS = _focus_cases()
SEL_CASES = [c for c in os.environ.get("WR_CASES", "stem_conv1").split(",") if c]
SEL_VARIANTS = [v for v in os.environ.get("WR_VARIANTS", "base,cand").split(",") if v]
PARAMS = []
for i, c in enumerate(SEL_CASES):
    vs = SEL_VARIANTS if (os.environ.get("WR_ORDER") != "alt" or i % 2 == 0) else list(reversed(SEL_VARIANTS))
    PARAMS += [(c, v) for v in vs]
RUN_LOG = HERE / "runs" / os.environ.get("WR_LOG", f"run_{time.strftime('%Y%m%d_%H%M%S')}.jsonl")


def _resolve(case_id):
    if case_id in FOCUS:
        return FOCUS[case_id]
    hits = [k for k in FOCUS if k.startswith(case_id)]
    assert len(hits) == 1, (case_id, hits, list(FOCUS))
    return FOCUS[hits[0]]


def _generic(spec):
    _, mem, dims, s, p, d, bias, act, *rest = spec.split(":")
    H, W, c_in, c_out = (int(v) for v in dims.split("x"))
    fid = (rest[0] if rest else "lofi").lower()
    wdt = ttnn.bfloat16 if (len(rest) > 1 and rest[1] == "bf16") else ttnn.bfloat8_b
    return dict(
        inputs=((H, W, c_in), c_out, int(p[1:]), int(s[1:]), int(d[1:])),
        dtype=ttnn.bfloat16,
        weight_dtype=wdt,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED
        if mem == "height"
        else ttnn.TensorMemoryLayout.INTERLEAVED,
        fp32_dest_acc_en=True,
        bias_mode="with_bias" if bias == "b" else "no_bias",
        activation=act,
        extras=dict(
            layer=spec,
            math_fidelity=ttnn.MathFidelity.LoFi if fid == "lofi" else ttnn.MathFidelity.HiFi4,
            tolerance=(0.996, 0.08) if fid == "lofi" else (0.999, None),
        ),
    )


def _run(case, device, op):
    import torch
    from eval.golden_tests.conv2d_3x3 import helpers

    torch.manual_seed(0)
    (H, W, c_in), c_out, padding, stride, dilation = case["inputs"]
    extras = case["extras"]
    dtype, weight_dtype, layout = case["dtype"], case["weight_dtype"], case["layout"]
    pinned = "shard_shape" in extras
    if case["memory_layout"] == ttnn.TensorMemoryLayout.INTERLEAVED:
        memory_config = ttnn.DRAM_MEMORY_CONFIG
    else:
        memory_config = helpers.build_input_memory_config(
            (1, 1, H * W, c_in), case["memory_layout"], layout=layout, dtype=dtype, device=device, extras=extras
        )
    if pinned and (
        memory_config.shard_spec is None or list(memory_config.shard_spec.shape) != list(extras["shard_shape"])
    ):
        pytest.skip(f"{extras['layer']}: pinned shard geometry does not fit this device's grid")
    kwargs = dict(
        stride=stride,
        padding=padding,
        dilation=dilation,
        input_dtype=dtype,
        input_layout=layout,
        weight_dtype=weight_dtype,
        input_memory_config=memory_config,
        device=device,
    )
    torch_weight = helpers.make_weight(c_out, c_in, dtype=torch.bfloat16)
    has_bias = case["bias_mode"] == "with_bias"
    activation = None if case["activation"] == "none" else case["activation"]
    torch_bias = helpers.make_bias(c_out, dtype=torch.bfloat16) if has_bias else None
    tt_weight = op.prepare_conv2d_weights(torch_weight, **kwargs)
    tt_bias = op.prepare_conv2d_bias(torch_bias, **kwargs) if has_bias else None
    torch_input = helpers.make_activation(H, W, c_in, dtype=torch.bfloat16)
    tt_input = ttnn.from_torch(torch_input, dtype=dtype, layout=layout, device=device, memory_config=memory_config)
    tt_output = op.conv2d_3x3(
        tt_input,
        tt_weight,
        input_height=H,
        input_width=W,
        bias=tt_bias,
        padding=padding,
        stride=stride,
        dilation=dilation,
        activation=activation,
        compute_kernel_config=helpers.make_compute_kernel_config(
            case["fp32_dest_acc_en"], extras.get("math_fidelity", ttnn.MathFidelity.HiFi4)
        ),
    )
    ref = helpers.pytorch_conv2d_3x3(
        torch_input.float(),
        torch_weight.float(),
        input_height=H,
        input_width=W,
        bias=torch_bias.float() if torch_bias is not None else None,
        padding=padding,
        stride=stride,
        dilation=dilation,
        activation=activation,
    )
    pcc = (extras.get("tolerance") or (0.99, None))[0]
    return tt_output, ref.float(), pcc


def _record_blocking(monkeypatch, info, op, pd):
    real = op.select_blocking

    def recording(g, grid, l1_budget):
        b = real(g, grid, l1_budget)
        info["blocking"] = dict(
            bm=b.block_m_tiles,
            bk=b.block_k_tiles,
            nkb=b.num_k_blocks,
            sb=(b.out_subblock_h, b.out_subblock_w),
            face_direct=bool(pd.face_direct_im2col(g, b.block_k_tiles)),
            sd=bool(pd.slab_direct_capable(g, b.block_k_tiles)),
            footprint=b.footprint_bytes,
            budget=l1_budget,
        )
        return b

    monkeypatch.setattr(op, "select_blocking", recording)
    real_servers = pd._weight_servers

    def servers(*a, **k):
        out = real_servers(*a, **k)
        info["servers"] = out is not None
        return out

    monkeypatch.setattr(pd, "_weight_servers", servers)
    force_bm = int(os.environ.get("WR_FORCE_BM", "0"))
    if force_bm:
        # Force M-blocks of force_bm tile-rows (several M-blocks per core: weight rounds after round 0).
        def blocks(g, grid, l1_budget):
            b = real(g, grid, l1_budget)
            fit = pd._footprint(g, b.core_m_tiles, force_bm, b.block_k_tiles, b.block_n_tiles)
            out = pd.Conv3x3Blocking(
                grid_x=b.grid_x,
                core_m_tiles=b.core_m_tiles,
                num_m_groups=b.num_m_groups,
                block_m_tiles=force_bm,
                block_k_tiles=b.block_k_tiles,
                block_n_tiles=b.block_n_tiles,
                out_depth_row_groups=b.out_depth_row_groups,
                **fit,
            )
            info["blocking"] = dict(info.get("blocking", {}), bm=out.block_m_tiles, forced=True)
            return out

        monkeypatch.setattr(op, "select_blocking", blocks)


@pytest.mark.parametrize("case_id,variant", PARAMS, ids=[f"{c}-{v}" for c, v in PARAMS])
def test_wr(case_id, variant, device, monkeypatch):
    import torch

    op, pd = apply_variant(variant, monkeypatch)
    info = dict(case=case_id, variant=variant)
    _record_blocking(monkeypatch, info, op, pd)
    if case_id.startswith("g:"):
        case = _generic(case_id)
    elif case_id.startswith("h:"):
        # h:<focus id>: the focus case at HiFi4 (its precision contract: tolerance 0.999).
        base_case = _resolve(case_id[2:])
        case = dict(
            base_case,
            extras=dict(base_case["extras"], math_fidelity=ttnn.MathFidelity.HiFi4, tolerance=(0.999, None)),
        )
    elif case_id.startswith("f:"):
        # f:<focus id>:<bias b|nb>:<act relu|none>: a focus case with the bias / activation overridden.
        _, fid, bias, act = case_id.split(":")
        case = dict(_resolve(fid), bias_mode="with_bias" if bias == "b" else "no_bias", activation=act)
    else:
        case = _resolve(case_id)
    tt_output, ref, pcc_threshold = _run(case, device, op)
    out = ttnn.to_torch(tt_output).float().reshape(ref.shape)
    passed, pcc = comp_pcc(ref, out, pcc_threshold)
    info.update(
        pcc=float(pcc),
        pcc_threshold=pcc_threshold,
        mae=float((out - ref).abs().mean()),
        max_abs=float((out - ref).abs().max()),
        finite=bool(torch.isfinite(out).all()),
        out_sha=hashlib.sha256(out.numpy().tobytes()).hexdigest()[:16],
    )
    if os.environ.get("WR_DUMP"):
        torch.save(
            dict(out=out, ref=ref, case=case_id, variant=variant),
            f"/tmp/wr_dump_{abs(hash(case_id)) % 10**8}_{variant}.pt",
        )
    RUN_LOG.parent.mkdir(exist_ok=True)
    with open(RUN_LOG, "a") as f:
        f.write(json.dumps(info) + "\n")
    if not os.environ.get("WR_NOASSERT"):
        assert info["finite"] and passed, info


# The unit tests' face-direct shapes, with their shard rows (footprints across shards), optionally forced to 4-tile-row
# M-blocks (2 M-blocks per core); every output dumped for an exact cross-variant comparison (compare_dumps.py).
UNIT_SHAPES = [
    ((64, 128, 8), 64, 1, 2, 1, 1000),
    ((32, 64, 16), 32, 1, 1, 1, 230),
    ((48, 64, 8), 64, 1, 2, 1, 700),
]
UNIT_PARAMS = [
    (i, mb, bias, relu, v)
    for i in range(len(UNIT_SHAPES))
    for mb in (0, 4)
    for bias, relu in ((True, True), (False, True), (True, False))
    for v in ("base", "cand")
    if not (mb and i != 0)
]


@pytest.mark.skipif(not os.environ.get("WR_UNIT_SHAPES"), reason="WR_UNIT_SHAPES=1 to run")
@pytest.mark.parametrize(
    "idx,mb,bias,relu,variant", UNIT_PARAMS, ids=[f"u{i}-mb{m}-b{int(b)}r{int(r)}-{v}" for i, m, b, r, v in UNIT_PARAMS]
)
def test_wr_unit_shapes(idx, mb, bias, relu, variant, device, monkeypatch):
    import torch
    from tests.ttnn.unit_tests.operations.conv2d_3x3 import test_conv2d_3x3_height_sharded as hs
    from eval.golden_tests.conv2d_3x3 import helpers

    op, pd = apply_variant(variant, monkeypatch)
    (H, W, c_in), c_out, padding, stride, dilation, shard_rows = UNIT_SHAPES[idx]
    if mb:
        real_select = pd.select_blocking

        def blocks(g, grid, l1_budget):
            b = real_select(g, grid, l1_budget)
            fit = pd._footprint(g, b.core_m_tiles, mb, b.block_k_tiles, b.block_n_tiles)
            return pd.Conv3x3Blocking(
                grid_x=b.grid_x,
                core_m_tiles=b.core_m_tiles,
                num_m_groups=b.num_m_groups,
                block_m_tiles=mb,
                block_k_tiles=b.block_k_tiles,
                block_n_tiles=b.block_n_tiles,
                out_depth_row_groups=b.out_depth_row_groups,
                **fit,
            )

        monkeypatch.setattr(op, "select_blocking", blocks)
    torch.manual_seed(11)
    config = hs._row_major_config(device, H * W, c_in, ttnn.ROW_MAJOR_LAYOUT, shard_rows)
    kwargs = dict(
        stride=stride,
        padding=padding,
        dilation=dilation,
        input_dtype=ttnn.bfloat16,
        input_layout=ttnn.ROW_MAJOR_LAYOUT,
        weight_dtype=ttnn.bfloat8_b,
        input_memory_config=config,
        device=device,
    )
    w = helpers.make_weight(c_out, c_in, dtype=torch.bfloat16)
    b = helpers.make_bias(c_out, dtype=torch.bfloat16) if bias else None
    x = helpers.make_activation(H, W, c_in, dtype=torch.bfloat16)
    out = op.conv2d_3x3(
        ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=config),
        op.prepare_conv2d_weights(w, **kwargs),
        input_height=H,
        input_width=W,
        bias=op.prepare_conv2d_bias(b, **kwargs) if bias else None,
        padding=padding,
        stride=stride,
        dilation=dilation,
        activation="relu" if relu else None,
        compute_kernel_config=helpers.make_compute_kernel_config(True, ttnn.MathFidelity.LoFi),
    )
    out = ttnn.to_torch(out).float()
    ref = helpers.pytorch_conv2d_3x3(
        x.float(),
        w.float(),
        input_height=H,
        input_width=W,
        bias=b.float() if bias else None,
        padding=padding,
        stride=stride,
        dilation=dilation,
        activation="relu" if relu else None,
    ).float()
    out = out.reshape(ref.shape)
    os.makedirs("/tmp/wr_unit_dumps", exist_ok=True)
    torch.save(dict(out=out, ref=ref), f"/tmp/wr_unit_dumps/u{idx}_mb{mb}_b{int(bias)}r{int(relu)}_{variant}.pt")
    passed, pcc = comp_pcc(ref, out, 0.996)
    assert passed and bool(torch.isfinite(out).all()), pcc


@pytest.mark.skipif(not os.environ.get("WR_STRESS"), reason="WR_STRESS=<n> to run")
@pytest.mark.parametrize("case_id,variant", PARAMS, ids=[f"stress-{c}-{v}" for c, v in PARAMS])
def test_wr_stress(case_id, variant, device, monkeypatch):
    """The same dispatch WR_STRESS times (program-cache hits after the first): every output identical to the first."""
    op, pd = apply_variant(variant, monkeypatch)
    case = _generic(case_id) if case_id.startswith("g:") else _resolve(case_id)
    first = None
    for i in range(int(os.environ["WR_STRESS"])):
        tt_output, ref, pcc_threshold = _run(case, device, op)
        out = ttnn.to_torch(tt_output).float().reshape(ref.shape)
        if first is None:
            first = out
            passed, pcc = comp_pcc(ref, out, pcc_threshold)
            assert passed, pcc
        else:
            assert bool((out == first).all()), f"dispatch {i} differs"
