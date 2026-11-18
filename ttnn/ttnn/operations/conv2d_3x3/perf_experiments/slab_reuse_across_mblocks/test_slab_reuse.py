# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Isolated bake-off: untilize each input tile-row once per core instead of once per M-block (perf experiment).

Runs a private copy of conv2d_3x3 (op_copy/), switched by the module knob CORE_SLAB (sr_load.set_variant):
  base  the op's current TILE slab: per M-block the reader plans the block's three kernel-row bands
        (tile_slab_plan), streams their input tile-rows, compute untilizes slab_tile_rows of them into cb_halo_slab
        and the reader pops the slab after the block's gather. Rows shared by consecutive M-blocks are re-read and
        re-untilized.
  core  the core-wide slab: the same plan applied once to the core's whole M range (the core as one block), filled
        and untilized once before the first M-block and popped after the last; every M-block gathers at the core's
        slab_x_base, waiting only for the slab rows of its own bands. select_blocking takes it when it is fewer
        tile-rows than the per-block slabs together and fits L1, else falls back to per-block.

Precision contract is each case's own (fp32 DEST, its fidelity, bf8_b/bf16): every variant runs it unchanged. The two
variants untilize the same input tiles into slab rows the gather reads at the same pixel, so their outputs must be
bit-identical (test_sr_identical, torch.equal).

One dispatch per test_sr_perf, for `run_safe_pytest.sh --profile`. Each test appends a JSON line to results.jsonl.
  SR_CASES=focus SR_VARIANTS=base,core SR_TAG=t1 scripts/run_safe_pytest.sh --profile --run-all \\
      <this file> -k test_sr_perf --import-mode=prepend
Never profile this copy next to the real op in one process (zone-id hash collisions abort the profile).
"""

import json
import os
import sys
from pathlib import Path

import pytest
import ttnn

sys.path.insert(0, str(Path(__file__).parent))
import sr_load  # noqa: E402

opc, pd, op_mod = sr_load.load()

RESULTS = Path(__file__).parent / "results.jsonl"
VARIANTS = ["base", "core", "core_first"]

HS = ttnn.TensorMemoryLayout.HEIGHT_SHARDED


def _grid(num_cores, grid_x=11):
    full_rows, rem = divmod(num_cores, grid_x)
    ranges = []
    if full_rows:
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_x - 1, full_rows - 1)))
    if rem:
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, full_rows), ttnn.CoreCoord(rem - 1, full_rows)))
    return ttnn.CoreRangeSet(ranges)


def _height(shard, num_cores):
    return ttnn.MemoryConfig(
        HS, ttnn.BufferType.L1, ttnn.ShardSpec(_grid(num_cores), shard, ttnn.ShardOrientation.ROW_MAJOR)
    )


# The HEIGHT perf harness (test_conv2d_3x3_perf.py HEIGHT_SHAPES) in its config: bf8_b weights, bias + ReLU, the
# op's default compute config.
HEIGHT = [
    ("stem_conv1", ((512, 1024, 8), 64, 1, 2), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, [4767, 8], 110),
    ("stem_conv3", ((256, 512, 64), 128, 1, 1), ttnn.bfloat8_b, ttnn.TILE_LAYOUT, [1216, 64], 110),
    ("res2_conv2", ((128, 256, 64), 64, 1, 1), ttnn.bfloat8_b, ttnn.TILE_LAYOUT, [320, 64], 103),
    ("res3_0_conv2", ((128, 256, 128), 128, 1, 2), ttnn.bfloat8_b, ttnn.TILE_LAYOUT, [384, 128], 86),
    ("res3_conv2", ((64, 128, 128), 128, 1, 1), ttnn.bfloat8_b, ttnn.TILE_LAYOUT, [96, 128], 86),
    ("sem_res2_fuse_conv0", ((128, 256, 288), 256, 1, 1), ttnn.bfloat8_b, ttnn.TILE_LAYOUT, [320, 288], 103),
    ("stem_conv2", ((256, 512, 64), 64, 1, 1), ttnn.bfloat8_b, ttnn.TILE_LAYOUT, [1216, 64], 110),
    ("sem_res3_fuse_conv0", ((64, 128, 320), 256, 1, 1), ttnn.bfloat8_b, ttnn.TILE_LAYOUT, [96, 320], 86),
    ("inst_res3_fuse_conv0", ((64, 128, 320), 128, 1, 1), ttnn.bfloat8_b, ttnn.TILE_LAYOUT, [96, 320], 86),
    ("sem_res3_fuse_conv1", ((64, 128, 256), 256, 1, 1), ttnn.bfloat8_b, ttnn.TILE_LAYOUT, [96, 256], 86),
    ("inst_res2_fuse_conv0", ((128, 256, 160), 128, 1, 1), ttnn.bfloat8_b, ttnn.TILE_LAYOUT, [320, 160], 103),
    ("sem_res2_fuse_conv1", ((128, 256, 256), 256, 1, 1), ttnn.bfloat8_b, ttnn.TILE_LAYOUT, [320, 256], 103),
    ("inst_res2_fuse_conv1", ((128, 256, 128), 128, 1, 1), ttnn.bfloat8_b, ttnn.TILE_LAYOUT, [320, 128], 103),
    ("center_head1", ((128, 256, 128), 32, 1, 1), ttnn.bfloat8_b, ttnn.TILE_LAYOUT, [320, 128], 103),
]

# The interleaved perf harness (test_conv2d_3x3_perf.py PERF_SHAPES x LAYOUTS): bf16 weights, bias, no ReLU.
INTERLEAVED_SHAPES = [
    ((32, 64, 128), 128, 1),
    ((16, 16, 64), 128, 1),
    ((8, 16, 512), 256, 1),
    ((8, 16, 2048), 256, 1),
    ((64, 128, 128), 128, 1),
    ((128, 256, 64), 64, 1),
]
INTERLEAVED_LAYOUTS = [
    ("rm_bf16", ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
    ("tile_bf16", ttnn.bfloat16, ttnn.TILE_LAYOUT),
    ("tile_bf8b", ttnn.bfloat8_b, ttnn.TILE_LAYOUT),
]


# The BLOCK perf harness (test_conv2d_3x3_perf.py BLOCK_SHAPES): K split, TILE bf8_b x bf8_b, fp32 DEST.
BLOCK = [
    ("res4_0_conv2", ((64, 128, 256), 256, 1, 1), [1024, 32], (8, 8), True, ttnn.MathFidelity.HiFi4, 2),
    ("res4_conv2", ((32, 64, 256), 256, 1, 1), [256, 32], (8, 8), True, ttnn.MathFidelity.HiFi4, 1),
    ("res4_conv2_lofi", ((32, 64, 256), 256, 1, 1), [256, 32], (8, 8), True, ttnn.MathFidelity.LoFi, 1),
    ("res5_0_conv2", ((32, 64, 512), 512, 2, 2), [224, 64], (8, 10), True, ttnn.MathFidelity.HiFi4, 1),
    ("res5_1_conv2", ((32, 64, 512), 512, 4, 4), [224, 64], (8, 10), True, ttnn.MathFidelity.HiFi4, 1),
    ("res5_2_conv2", ((32, 64, 512), 512, 8, 8), [224, 64], (8, 10), True, ttnn.MathFidelity.HiFi4, 1),
    ("aspp_convs1", ((32, 64, 2048), 256, 6, 6), [256, 192], (11, 8), True, ttnn.MathFidelity.HiFi4, 1),
    ("aspp_convs2", ((32, 64, 2048), 256, 12, 12), [256, 192], (11, 8), True, ttnn.MathFidelity.HiFi4, 1),
    ("aspp_convs3", ((32, 64, 1024), 256, 18, 18), [256, 96], (11, 8), False, ttnn.MathFidelity.HiFi4, 1),
]

# Interleaved TILE shapes outside the harness, picked to land on more than one M-block per core.
INTERLEAVED_EXTRA = [
    ((128, 256, 256), 256, 1),
    ((128, 256, 512), 256, 1),
    ((256, 512, 128), 128, 1),
    ((64, 128, 1024), 512, 1),
]


def _block_mem(shard, gx, gy):
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))]),
            shard,
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )


def _cases():
    from eval.golden_tests.conv2d_3x3.feature_spec import LOOSE_CASES

    out = {}
    (focus,) = [
        c
        for c in LOOSE_CASES
        if c["extras"]["layer"].startswith("semantic res2.fuse_conv.1") and "[perf]" in c["extras"]["layer"]
    ]
    (H, W, c_in), c_out, padding, stride, dilation = focus["inputs"]
    out["focus"] = dict(
        shape=((H, W, c_in), c_out, padding, stride, dilation),
        dtype=focus["dtype"],
        layout=focus["layout"],
        weight_dtype=focus["weight_dtype"],
        mem=lambda: _height(list(focus["extras"]["shard_shape"]), 103),
        bias=focus["bias_mode"] == "with_bias",
        relu=focus["activation"] == "relu",
        fidelity=focus["extras"].get("math_fidelity"),
        fp32_dest=focus["fp32_dest_acc_en"],
        pcc=focus["extras"]["tolerance"][0],
        golden=True,
    )
    for layer, ((H, W, c_in), c_out, padding, stride), dtype, layout, shard, cores in HEIGHT:
        out[layer] = dict(
            shape=((H, W, c_in), c_out, padding, stride, 1),
            dtype=dtype,
            layout=layout,
            weight_dtype=ttnn.bfloat8_b,
            mem=(lambda shard=shard, cores=cores: _height(shard, cores)),
            bias=True,
            relu=True,
            fidelity=None,
            fp32_dest=None,
            pcc=0.99,
            golden=False,
        )
    for (H, W, c_in), c_out, padding in INTERLEAVED_SHAPES:
        for lid, dtype, layout in INTERLEAVED_LAYOUTS:
            out[f"dram_H{H}_W{W}_Ci{c_in}_Co{c_out}_{lid}"] = dict(
                shape=((H, W, c_in), c_out, padding, 1, 1),
                dtype=dtype,
                layout=layout,
                weight_dtype=ttnn.bfloat16,
                mem=lambda: ttnn.DRAM_MEMORY_CONFIG,
                bias=True,
                relu=False,
                fidelity=None,
                fp32_dest=None,
                pcc=0.995 if dtype == ttnn.bfloat16 else 0.99,
                golden=False,
            )
    for (H, W, c_in), c_out, padding in INTERLEAVED_EXTRA:
        for lid, dtype, layout in INTERLEAVED_LAYOUTS[1:]:
            out[f"dramx_H{H}_W{W}_Ci{c_in}_Co{c_out}_{lid}"] = dict(
                shape=((H, W, c_in), c_out, padding, 1, 1),
                dtype=dtype,
                layout=layout,
                weight_dtype=ttnn.bfloat8_b,
                mem=lambda: ttnn.DRAM_MEMORY_CONFIG,
                bias=True,
                relu=True,
                fidelity=None,
                fp32_dest=None,
                pcc=0.995 if dtype == ttnn.bfloat16 else 0.99,
                golden=False,
            )
    for layer, ((H, W, c_in), c_out, padding, dilation), shard, (gx, gy), bias_relu, fidelity, stride in BLOCK:
        out[f"block_{layer}"] = dict(
            shape=((H, W, c_in), c_out, padding, stride, dilation),
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            weight_dtype=ttnn.bfloat8_b,
            mem=(lambda shard=shard, gx=gx, gy=gy: _block_mem(shard, gx, gy)),
            bias=bias_relu,
            relu=bias_relu,
            fidelity=fidelity,
            fp32_dest=True,
            pcc=0.99,
            golden=False,
        )
    return out


CASES = _cases()


def _selected(env, default):
    v = os.environ.get(env)
    return [x for x in v.split(",") if x] if v else default


def run_case(case, device, seed=0, dispatch=True):
    import torch

    torch.manual_seed(seed)
    (H, W, c_in), c_out, padding, stride, dilation = case["shape"]
    memory_config = case["mem"]()
    kwargs = dict(
        stride=stride,
        padding=padding,
        dilation=dilation,
        input_dtype=case["dtype"],
        input_layout=case["layout"],
        weight_dtype=case["weight_dtype"],
        input_memory_config=memory_config,
        device=device,
    )
    if case["golden"]:
        from eval.golden_tests.conv2d_3x3 import helpers

        torch_weight = helpers.make_weight(c_out, c_in, dtype=torch.bfloat16)
        torch_bias = helpers.make_bias(c_out, dtype=torch.bfloat16) if case["bias"] else None
        torch_input = helpers.make_activation(H, W, c_in, dtype=torch.bfloat16)
    else:
        torch_weight = torch.randn(c_out, c_in, 3, 3).to(torch.bfloat16)
        torch_bias = torch.randn(c_out).to(torch.bfloat16) if case["bias"] else None
        torch_input = torch.randn(1, 1, H * W, c_in).to(torch.bfloat16)
    tt_weight = opc.prepare_conv2d_weights(torch_weight, **kwargs)
    tt_bias = opc.prepare_conv2d_bias(torch_bias, **kwargs) if case["bias"] else None
    tt_input = ttnn.from_torch(
        torch_input, dtype=case["dtype"], layout=case["layout"], device=device, memory_config=memory_config
    )
    config = None
    if case["fidelity"] is not None:
        config = ttnn.ComputeConfigDescriptor(
            math_fidelity=case["fidelity"], fp32_dest_acc_en=case["fp32_dest"], math_approx_mode=False
        )
    tt_output = opc.conv2d_3x3(
        tt_input,
        tt_weight,
        input_height=H,
        input_width=W,
        bias=tt_bias,
        padding=padding,
        stride=stride,
        dilation=dilation,
        activation="relu" if case["relu"] else None,
        compute_kernel_config=config,
    )
    if not dispatch:
        return None, None
    if case["golden"]:
        ref = helpers.pytorch_conv2d_3x3(
            torch_input,
            torch_weight,
            input_height=H,
            input_width=W,
            bias=torch_bias,
            padding=padding,
            stride=stride,
            dilation=dilation,
            activation="relu" if case["relu"] else None,
        ).float()
        return ttnn.to_torch(tt_output).float().reshape(ref.shape), ref
    x = torch_input.float().reshape(1, H, W, c_in).permute(0, 3, 1, 2)
    ref = torch.nn.functional.conv2d(
        x,
        torch_weight.float(),
        None if torch_bias is None else torch_bias.float(),
        stride=stride,
        padding=padding,
        dilation=dilation,
    )
    if case["relu"]:
        ref = torch.relu(ref)
    ref = ref.permute(0, 2, 3, 1).reshape(1, 1, -1, c_out)
    return ttnn.to_torch(tt_output).float().reshape(ref.shape), ref


def _pcc(a, b):
    import torch

    a, b = a.flatten().double(), b.flatten().double()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()))


class _Spy:
    """Records the blocking select_blocking returns (conv2d_3x3 imported it by name)."""

    def __init__(self):
        self.blocking = None

    def __call__(self, g, grid, budget):
        self.blocking = pd.select_blocking(g, grid, budget)
        self.geometry, self.budget = g, budget
        return self.blocking


def _blocking_summary(spy):
    b, g = spy.blocking, spy.geometry
    return dict(
        bm=b.block_m_tiles,
        bk=b.block_k_tiles,
        core_m_tiles=b.core_m_tiles,
        nb=b.max_m_blocks_per_core,
        slab_rows=b.slab_tile_rows,
        ub=b.untilize_batch_rows,
        im2col_depth=b.im2col_rm_depth_rows,
        core_slab=b.core_slab,
        footprint_kb=round(b.footprint_bytes / 1024, 1),
        budget_kb=round(spy.budget / 1024, 1),
        c_in_tiles=g.c_in_tiles,
        tile=g.tile_input,
    )


ALL = list(CASES)


@pytest.mark.parametrize("case_id", _selected("SR_CASES", ALL))
def test_sr_blocking(case_id, device, monkeypatch):
    """Host only (generic_op stubbed): the blocking each variant selects, appended to results.jsonl."""
    case = CASES[case_id]
    spy = _Spy()
    monkeypatch.setattr(op_mod, "select_blocking", spy)
    monkeypatch.setattr(op_mod.ttnn, "generic_op", lambda io, desc: io[-1])
    rec = dict(kind="blocking", case=case_id)
    for v in VARIANTS:
        sr_load.set_variant(pd, v)
        run_case(case, device, dispatch=False)
        rec[v] = _blocking_summary(spy)
    sr_load.set_variant(pd, "core")
    print(json.dumps(rec))
    with open(RESULTS, "a") as f:
        f.write(json.dumps(rec) + "\n")


@pytest.mark.parametrize("variant", _selected("SR_VARIANTS", VARIANTS))
@pytest.mark.parametrize("case_id", _selected("SR_CASES", ALL))
def test_sr_perf(case_id, variant, device, monkeypatch):
    """One dispatch of one variant, PCC-gated against torch."""
    case = CASES[case_id]
    spy = _Spy()
    monkeypatch.setattr(op_mod, "select_blocking", spy)
    sr_load.set_variant(pd, variant)
    out, ref = run_case(case, device)
    pcc = _pcc(out, ref)
    with open(RESULTS, "a") as f:
        f.write(
            json.dumps(
                dict(
                    kind="perf",
                    case=case_id,
                    variant=variant,
                    pcc=pcc,
                    tag=os.environ.get("SR_TAG", ""),
                    **_blocking_summary(spy),
                )
            )
            + "\n"
        )
    assert pcc >= case["pcc"], f"{case_id}/{variant}: pcc {pcc} < {case['pcc']}"


@pytest.mark.parametrize("variant", [v for v in _selected("SR_VARIANTS", VARIANTS) if v != "base"])
@pytest.mark.parametrize("case_id", _selected("SR_CASES", ALL))
def test_sr_identical(case_id, variant, device, monkeypatch):
    """Same inputs through base and a core-slab variant: the outputs must be bit-identical (torch.equal), and
    PCC-gated. Skipped where select_blocking does not take the core slab (the same program both ways)."""
    import torch

    case = CASES[case_id]
    spy = _Spy()
    monkeypatch.setattr(op_mod, "select_blocking", spy)
    sr_load.set_variant(pd, variant)
    pd.select_blocking.cache_clear()
    spy(*_geometry_args(case, device, monkeypatch))
    if not spy.blocking.core_slab:
        pytest.skip(f"{case_id}/{variant}: select_blocking does not take the core slab (same program both ways)")
    outs = {}
    for v in ("base", variant):
        sr_load.set_variant(pd, v)
        outs[v], ref = run_case(case, device, seed=1)
        assert spy.blocking.core_slab == (v != "base")
    same = bool(torch.equal(outs["base"], outs[variant]))
    pcc = _pcc(outs[variant], ref)
    with open(RESULTS, "a") as f:
        f.write(
            json.dumps(
                dict(kind="identical", case=case_id, variant=variant, identical=same, pcc=pcc, **_blocking_summary(spy))
            )
            + "\n"
        )
    assert (
        same
    ), f"{case_id}/{variant}: output differs from base (max |diff| {(outs['base'] - outs[variant]).abs().max()})"
    assert pcc >= case["pcc"]


def _geometry_args(case, device, monkeypatch):
    """(geometry, grid, budget) select_blocking sees for `case`, from one host-only call (generic_op stubbed)."""
    seen = {}

    def record(g, grid, budget):
        seen["args"] = (g, grid, budget)
        return pd.select_blocking(g, grid, budget)

    with monkeypatch.context() as m:
        m.setattr(op_mod, "select_blocking", record)
        m.setattr(op_mod.ttnn, "generic_op", lambda io, desc: io[-1])
        run_case(case, device, dispatch=False)
    return seen["args"]
