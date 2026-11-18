# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The graduation candidate (op_grad/: the committed op + graduate.patch, no experiment knobs) against the op copy's
baseline walk (op_copy/ with its knobs at default), torch.equal on the same inputs: the focus cases, the BLOCK perf
shapes and the BLOCK unit geometries. Not for --profile (two copies of the op's kernels in one process)."""

import importlib

import pytest

_BASE = "ttnn.operations.conv2d_3x3.perf_experiments.nslice_stream_first_slice"
perf = importlib.import_module(_BASE + ".test_nslice_stream")
unit = importlib.import_module(_BASE + ".test_nslice_stream_unit")
grad_op = importlib.import_module(_BASE + ".op_grad.conv2d_3x3")
grad_pd = importlib.import_module(_BASE + ".op_grad.conv2d_3x3_program_descriptor")

PERF_CASES = perf.FOCUS_DEFAULT + [f"block:{k}" for k in perf.BLOCK_SHAPES]


def _stream_of(module_op, monkeypatch, info):
    sel = module_op.select_blocking

    def recording(g, grid, l1_budget):
        b = sel(g, grid, l1_budget)
        info["stream"] = grad_pd._nslice_stream_args(b, b.block_m_tiles, g.out_n_tiles) if b.nslice else None
        return b

    monkeypatch.setattr(module_op, "select_blocking", recording)


@pytest.mark.parametrize("case_id", PERF_CASES)
def test_grad_equal_perf_shapes(case_id, device, monkeypatch):
    import torch
    import ttnn

    perf.apply_variant("baseline", monkeypatch)

    def run():
        if case_id.startswith("block:"):
            return perf._run_block(case_id.split(":", 1)[1], device)
        return perf._run_focus(perf.FOCUS[case_id], device)

    base, ref, _ = run()
    base = ttnn.to_torch(base).float()
    info = {}
    monkeypatch.setattr(perf, "op", grad_op)
    _stream_of(grad_op, monkeypatch, info)
    cand, _, _ = run()
    cand = ttnn.to_torch(cand).float()
    print(f"{case_id}: stream {info.get('stream')}")
    assert torch.equal(cand, base), f"{case_id}: max diff {(cand - base).abs().max()}"


@pytest.mark.parametrize("shape,lid,br,multi", unit.CASES)
def test_grad_equal_unit(shape, lid, br, multi, device, monkeypatch):
    import torch

    for k, v in unit.KNOBS.items():
        monkeypatch.setattr(unit.pd, k, v)
    unit.pd.select_blocking.cache_clear()
    if multi:
        monkeypatch.setattr(unit.op, "select_blocking", unit._one_tile_row_blocks(unit.pd.select_blocking))
    base, ref = unit._run_once(shape, lid, br, device, seed=42)
    if multi:
        monkeypatch.setattr(grad_op, "select_blocking", unit._one_tile_row_blocks(grad_pd.select_blocking))
    info = {}
    _stream_of(grad_op, monkeypatch, info)
    monkeypatch.setattr(unit, "op", grad_op)
    monkeypatch.setattr(unit, "pd", grad_pd)
    cand, _ = unit._run_once(shape, lid, br, device, seed=42)
    print(f"stream {info.get('stream')}")
    assert torch.equal(cand, base), f"max diff {(cand - base).abs().max()}"
