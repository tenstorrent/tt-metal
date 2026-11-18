# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Bake-off: keep the N-slice-outer K split's slice sends from stalling the walk and setting the tail (perf round 5,
idea `nslice_send_decouple`).

Every variant runs on this dir's op copy (op_copy/: HEAD 2ac42333eb2 + this experiment's knobs), whose knobs default to
the committed op's behaviour (variant `baseline`). Never profile it in one process with the real op (zone-name hash
collisions).

Env: NSD_CASES (focus ids, `block:<layer>`, or `focus` / `block` / `all`), NSD_VARIANTS (comma list, run in that order
per case), NSD_LOG (run-log name). One dispatch per test, for `run_safe_pytest.sh --profile`; each dispatch appends a
JSON record (blocking, PCC, output hash) to runs/<log>.jsonl, which report.py joins with the profiler CSV's GenericOp
rows in call order. Precision contract fixed per case (LoFi / HiFi4 as the case says, fp32 DEST, bf8_b, fp32
partials): no variant touches it.
"""

import hashlib
import importlib
import json
import os
from pathlib import Path

import pytest
import ttnn

HERE = Path(__file__).parent
# NSD_PKG=graduated runs graduated/ (the real op with graduation.patch applied: the ring is its only N-slice path).
_COPY = os.environ.get("NSD_PKG", "op_copy")
_PKG = "ttnn.operations.conv2d_3x3.perf_experiments.nslice_send_decouple." + _COPY
op = importlib.import_module(_PKG + ".conv2d_3x3")
pd = importlib.import_module(_PKG + ".conv2d_3x3_program_descriptor")
assert Path(pd.KERNEL_DIR).parent == HERE / _COPY, pd.KERNEL_DIR

KNOB_DEFAULTS = dict(NSD_DEFINES=(), NSD_SLICE_DEPTH=None, NSD_RING_GROUP="sb", NSD_RING_INFLIGHT=2)
VARIANTS = {
    "baseline": {},
    # graduated/ as is (run with NSD_PKG=graduated)
    "grad": {},
    # diagnostic: the slice sends are not issued (semaphores kept; wrong output by construction, PCC not gated)
    "nosend": dict(_ablate="scatter"),
    # slice CBs deepened from the L1 left after the solve
    "d2": dict(NSD_SLICE_DEPTH=2),
    "dauto": dict(NSD_SLICE_DEPTH="auto"),
    # send order (conv2d_3x3_nslice_order.hpp NSD_ORDER): far = both RISCs farthest owner first, rfar = reader only,
    # wfar = writer only; at the committed slice depth (d1) or deepened (auto)
    "far": dict(NSD_DEFINES=("NSD_ORDER=1",)),
    "rfar": dict(NSD_DEFINES=("NSD_ORDER=2",)),
    "dauto_far": dict(NSD_SLICE_DEPTH="auto", NSD_DEFINES=("NSD_ORDER=1",)),
    "dauto_rfar": dict(NSD_SLICE_DEPTH="auto", NSD_DEFINES=("NSD_ORDER=2",)),
    "dauto_wfar": dict(NSD_SLICE_DEPTH="auto", NSD_DEFINES=("NSD_ORDER=3",)),
    # ring reduce-scatter: each owner's partial walks +1 column per step on NoC0 and is accumulated (packer L1 acc)
    # by every core on the way; the owner adds its own slice last and only the bias add remains
    "ring": dict(NSD_DEFINES=("NSD_RING",)),
    # one hop per whole slice instead of per subblock row group
    "ring_slice": dict(NSD_DEFINES=("NSD_RING",), NSD_RING_GROUP="slice"),
    # one write in flight (each group's write lands before the next is issued)
    "ring_if1": dict(NSD_DEFINES=("NSD_RING",), NSD_RING_INFLIGHT=1),
    "ring_slice_if1": dict(NSD_DEFINES=("NSD_RING",), NSD_RING_GROUP="slice", NSD_RING_INFLIGHT=1),
    "ring_if3": dict(NSD_DEFINES=("NSD_RING",), NSD_RING_INFLIGHT=3),
    # the token is freed as soon as the next step's partial landed, before this step's own send is fenced
    # diagnostics (wrong output by construction): no packer gate / no L1 accumulation
    "ring_nogate": dict(_diag=True, NSD_DEFINES=("NSD_RING", "NSD_RING_NOGATE")),
    "ring_noacc": dict(_diag=True, NSD_DEFINES=("NSD_RING", "NSD_RING_NOACC")),
    # pipelined sends: issue the next slice before fencing the previous one (per-RISC trid fences)
    "pipe": dict(NSD_SLICE_DEPTH="auto", NSD_DEFINES=("NSD_PIPE",)),
    "pipe2": dict(NSD_SLICE_DEPTH=2, NSD_DEFINES=("NSD_PIPE",)),
    # the last remote slice (d = k/2, even k) split over both NoCs
    "split": dict(NSD_DEFINES=("NSD_SPLIT_LAST",)),
    "dauto_split": dict(NSD_SLICE_DEPTH="auto", NSD_DEFINES=("NSD_SPLIT_LAST",)),
    "pipe_split": dict(NSD_SLICE_DEPTH="auto", NSD_DEFINES=("NSD_PIPE", "NSD_SPLIT_LAST")),
    # progressive combine: slots in step order, summed as they land
    "prog": dict(NSD_DEFINES=("NSD_PROG_COMBINE",)),
    "dauto_prog": dict(NSD_SLICE_DEPTH="auto", NSD_DEFINES=("NSD_PROG_COMBINE",)),
    "pipe_prog": dict(NSD_SLICE_DEPTH="auto", NSD_DEFINES=("NSD_PIPE", "NSD_PROG_COMBINE")),
    "all": dict(NSD_SLICE_DEPTH="auto", NSD_DEFINES=("NSD_PIPE", "NSD_SPLIT_LAST", "NSD_PROG_COMBINE")),
}

LOFI, HIFI4 = ttnn.MathFidelity.LoFi, ttnn.MathFidelity.HiFi4

# tests/ttnn/unit_tests/operations/conv2d_3x3/test_conv2d_3x3_perf.py BLOCK_SHAPES (the BLOCK guard harness).
# (layer, ((H, W, C_in), C_out, padding, dilation), shard_shape, (grid_x, grid_y), bias + ReLU, fidelity, stride)
BLOCK_SHAPES = [
    ("res4_0_conv2", ((64, 128, 256), 256, 1, 1), [1024, 32], (8, 8), True, HIFI4, 2),
    ("res4_conv2", ((32, 64, 256), 256, 1, 1), [256, 32], (8, 8), True, HIFI4, 1),
    ("res4_conv2_lofi", ((32, 64, 256), 256, 1, 1), [256, 32], (8, 8), True, LOFI, 1),
    ("res5_0_conv2", ((32, 64, 512), 512, 2, 2), [224, 64], (8, 10), True, HIFI4, 1),
    ("res5_1_conv2", ((32, 64, 512), 512, 4, 4), [224, 64], (8, 10), True, HIFI4, 1),
    ("res5_2_conv2", ((32, 64, 512), 512, 8, 8), [224, 64], (8, 10), True, HIFI4, 1),
    ("aspp_convs1", ((32, 64, 2048), 256, 6, 6), [256, 192], (11, 8), True, HIFI4, 1),
    ("aspp_convs2", ((32, 64, 2048), 256, 12, 12), [256, 192], (11, 8), True, HIFI4, 1),
    ("aspp_convs3", ((32, 64, 1024), 256, 18, 18), [256, 96], (11, 8), False, HIFI4, 1),
    # Not in the harness: several M-blocks per core (the non-shared N-slice walk, slice depth 2).
    ("res4_s1_lofi", ((64, 128, 256), 256, 1, 1), [1024, 32], (8, 8), True, LOFI, 1),
    # Not in the harness: C_out not a multiple of the slice width (skipped owners, short last slice), no bias.
    ("odd_cout", ((32, 64, 256), 200, 1, 1), [256, 32], (8, 8), False, LOFI, 1),
]


def _focus_cases():
    from eval.golden_tests.conv2d_3x3.feature_spec import LOOSE_CASES

    out = {}
    for c in LOOSE_CASES:
        if "[perf]" in c["extras"]["layer"]:
            out[c["extras"]["layer"].split(" ")[0].replace(".", "_")] = dict(kind="focus", spec=c)
    return out


def _cases():
    out = _focus_cases()
    for layer, ((H, W, c_in), c_out, padding, dilation), shard, grid_xy, bias_relu, fid, stride in BLOCK_SHAPES:
        out["block:" + layer] = dict(
            kind="block",
            shape=((H, W, c_in), c_out, padding, stride, dilation),
            shard=shard,
            grid_xy=grid_xy,
            bias_relu=bias_relu,
            fidelity=fid,
        )
    return out


CASES = _cases()


def _selected_cases():
    v = os.environ.get("NSD_CASES", "res4_0_conv2")
    out = []
    for x in v.split(","):
        if x == "focus":
            out += [k for k, c in CASES.items() if c["kind"] == "focus"]
        elif x == "block":
            out += [k for k, c in CASES.items() if c["kind"] == "block"]
        elif x == "all":
            out += list(CASES)
        elif x:
            out.append(x)
    return out


def _selected_variants():
    v = os.environ.get("NSD_VARIANTS")
    return [x for x in v.split(",") if x] if v else ["baseline"]


LAST = {}


def apply_variant(variant, monkeypatch):
    knobs = dict(KNOB_DEFAULTS)
    cfg = dict(VARIANTS[variant])
    ablate = cfg.pop("_ablate", "")
    diag = cfg.pop("_diag", False)
    knobs.update(cfg)
    for k, v in knobs.items():
        monkeypatch.setattr(pd, k, v, raising=False)
    if ablate:
        monkeypatch.setenv("CONV2D_3X3_ABLATE", ablate)
    else:
        monkeypatch.delenv("CONV2D_3X3_ABLATE", raising=False)
    pd.select_blocking.cache_clear()
    real = pd.create_program_descriptor

    def capture(*args, geometry, blocking, **kwargs):
        LAST.clear()
        LAST.update(
            k_cols=geometry.k_cols,
            ns=geometry.out_n_tiles,
            bm=blocking.block_m_tiles,
            bk=blocking.block_k_tiles,
            nkb=blocking.num_k_blocks,
            nslice=bool(blocking.nslice),
            shared=bool(blocking.shared_slab_bytes),
            mblocks=blocking.max_m_blocks_per_core,
            slice_depth=getattr(blocking, "slice_depth", None),
            foot=blocking.footprint_bytes,
        )
        return real(*args, geometry=geometry, blocking=blocking, **kwargs)

    monkeypatch.setattr(op, "create_program_descriptor", capture)
    return bool(ablate) or diag


def _block_mem(shard, grid_xy):
    gx, gy = grid_xy
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))]),
            shard,
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )


def run_case(case, device, seed=0):
    """Returns (tt output as float torch, torch reference, pcc threshold)."""
    import torch

    torch.manual_seed(seed)
    if case["kind"] == "focus":
        from eval.golden_tests.conv2d_3x3 import helpers

        spec = case["spec"]
        (H, W, c_in), c_out, padding, stride, dilation = spec["inputs"]
        extras = spec["extras"]
        dtype, weight_dtype, layout = spec["dtype"], spec["weight_dtype"], spec["layout"]
        memory_config = helpers.build_input_memory_config(
            (1, 1, H * W, c_in), spec["memory_layout"], layout=layout, dtype=dtype, device=device, extras=extras
        )
        if memory_config.shard_spec is None or list(memory_config.shard_spec.shape) != list(extras["shard_shape"]):
            pytest.skip("pinned shard geometry does not fit this device's grid")
        torch_weight = helpers.make_weight(c_out, c_in, dtype=torch.bfloat16)
        has_bias = spec["bias_mode"] == "with_bias"
        activation = None if spec["activation"] == "none" else spec["activation"]
        torch_bias = helpers.make_bias(c_out, dtype=torch.bfloat16) if has_bias else None
        torch_input = helpers.make_activation(H, W, c_in, dtype=torch.bfloat16)
        config = helpers.make_compute_kernel_config(
            spec["fp32_dest_acc_en"], extras.get("math_fidelity", ttnn.MathFidelity.HiFi4)
        )
        pcc_min = extras.get("tolerance", (0.99, None))[0]
    else:
        (H, W, c_in), c_out, padding, stride, dilation = case["shape"]
        dtype, weight_dtype, layout = ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.TILE_LAYOUT
        memory_config = _block_mem(case["shard"], case["grid_xy"])
        torch_weight = torch.randn(c_out, c_in, 3, 3).to(torch.bfloat16)
        torch_bias = torch.randn(c_out).to(torch.bfloat16) if case["bias_relu"] else None
        torch_input = torch.randn(1, 1, H * W, c_in).to(torch.bfloat16)
        activation = "relu" if case["bias_relu"] else None
        config = ttnn.ComputeConfigDescriptor(
            math_fidelity=case["fidelity"], fp32_dest_acc_en=True, math_approx_mode=False
        )
        pcc_min = 0.99
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
    tt_weight = op.prepare_conv2d_weights(torch_weight, **kwargs)
    tt_bias = op.prepare_conv2d_bias(torch_bias, **kwargs) if torch_bias is not None else None
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
        compute_kernel_config=config,
    )
    x = torch_input.float().reshape(1, H, W, c_in).permute(0, 3, 1, 2)
    ref = torch.nn.functional.conv2d(
        x,
        torch_weight.float(),
        None if torch_bias is None else torch_bias.float(),
        stride=stride,
        padding=padding,
        dilation=dilation,
    )
    if activation == "relu":
        ref = torch.relu(ref)
    ref = ref.permute(0, 2, 3, 1).reshape(1, 1, -1, c_out)
    return ttnn.to_torch(tt_output).float().reshape(ref.shape), ref, pcc_min


def _pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()))


def _log(rec):
    log = HERE / "runs" / os.environ.get("NSD_LOG", "default.jsonl")
    with open(log, "a") as f:
        f.write(json.dumps(rec) + "\n")


PERF = [(c, v) for c in _selected_cases() for v in _selected_variants()]


@pytest.mark.parametrize("case_id,variant", PERF, ids=[f"{c}-{v}" for c, v in PERF])
def test_nsd_perf(case_id, variant, device, monkeypatch):
    """One dispatch of one variant, PCC-gated against torch (except diagnostic ablations)."""
    ablated = apply_variant(variant, monkeypatch)
    out, ref, pcc_min = run_case(CASES[case_id], device)
    pcc = _pcc(out, ref)
    sha = hashlib.sha256(out.numpy().tobytes()).hexdigest()[:16]
    _log(dict(case=case_id, variant=variant, pcc=pcc, pcc_threshold=pcc_min, out_sha=sha, blocking=dict(LAST)))
    if os.environ.get("NSD_FLUSH") == "1":
        ttnn.ReadDeviceProfiler(device)
    if not ablated:
        assert pcc >= pcc_min, f"{case_id}/{variant}: pcc {pcc} < {pcc_min}"


PRECISION = [c for c in _selected_cases()]


@pytest.mark.parametrize("case_id", PRECISION)
def test_nsd_precision(case_id, device, monkeypatch):
    """Same inputs through `baseline` and NSD_CANDIDATE (default `ring`): bit-identity, how far the two differ, and
    each one's error against the fp32 torch reference. Not for --profile (two dispatches per test)."""
    import torch

    cand = os.environ.get("NSD_CANDIDATE", "ring")
    outs = {}
    for v in ("baseline", cand):
        apply_variant(v, monkeypatch)
        outs[v], ref, pcc_min = run_case(CASES[case_id], device, seed=1)
    a, b = outs["baseline"], outs[cand]
    diff = (a - b).abs()
    nz = diff > 0
    # relative size of a differing element's change vs its magnitude (a bf8_b step is 2^-7 relative to the block max)
    rel = (diff[nz] / a.abs()[nz].clamp_min(1e-30)) if nz.any() else torch.zeros(1)
    rec = dict(
        kind="precision",
        case=case_id,
        variant=cand,
        identical=bool(torch.equal(a, b)),
        differing=int(nz.sum()),
        numel=int(diff.numel()),
        max_diff=float(diff.max()),
        median_rel_change=float(rel.median()),
        pcc_base=_pcc(a, ref),
        pcc_cand=_pcc(b, ref),
        mae_base=float((a - ref).abs().mean()),
        mae_cand=float((b - ref).abs().mean()),
        maxerr_base=float((a - ref).abs().max()),
        maxerr_cand=float((b - ref).abs().max()),
        ref_absmax=float(ref.abs().max()),
    )
    _log(rec)
    print(json.dumps(rec))
    assert rec["pcc_cand"] >= pcc_min
