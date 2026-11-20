# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""ttnn.bringup.high_bw_all_gather gathered_dim_start (deepseek_v41_flash_d_p PB.6L): each rank moves its local rows
[start, start + gathered_dim_size / ranks) to the same rows of its fixed output slot (stock: only the prefix).

Whole mesh (the caller's: FABRIC_2D, cluster_axis None = every chip, and the SP axis), ROW_MAJOR uint8 records of
320 B (the model's 4-bit history records) and bf16 rows of 1024 B, random per-chip rows. Every chip's output against
the host: the range of every rank equal to its input rows, every other row untouched (the output pre-filled with a
pattern); several ranges in a row compose the whole prefix (the model's pieces); starts not on a bank boundary
(odd starts: lanes keep their banks); the default (no start) equal to the stock op's prefix byte for byte; on the
gather sub-device's column (sub_core_grids) as the model calls it; refusals (no gathered_dim_size, a range past the
local rows). Smallest shapes first. Runs on the resident runner's mesh (BATCH_SHARED: no device open of its own).

    python -m models.demos.common.bringup.resident submit --spec $BRINGUP_SPEC --cmd "scripts/run_safe_pytest.sh \\
        --run-all --no-precompile ttnn/ttnn/bringup/high_bw_all_gather_ttnn/tests/unit/test_range_gather.py"
"""

import pytest
import torch

from models.demos.common.bringup.testing.harness import mesh_parametrize

BATCH_SHARED = True
BATCH_LAYERS = "executed"


def _sems(mesh, axis):
    """The caller-owned (ready, data_valid) pair of one gather geometry (one pair an axis, never shared: DS41-216):
    the model's own pairs (zero between calls), so the runner's held model and this test share them."""
    from models.demos.deepseek_v41_flash_d_p.tt import ccl

    return ccl._sems(mesh, axis)


def _host_out(mesh, t, rows, width):
    import ttnn

    return [ttnn.to_torch(d).reshape(rows, width) for d in ttnn.get_device_tensors(t)]


def _case(mesh, axis, R, width, ranges, dtype, sub=None, seed=0):
    """Gather the ranges [(a, b)] in order into one output pre-filled with a pattern; check every chip."""
    import ttnn

    sp, tp = tuple(mesh.shape)
    n = sp * tp if axis is None else (sp if axis == 0 else tp)
    g = torch.Generator().manual_seed(seed)
    sems = _sems(mesh, axis)
    tdt = torch.uint8 if dtype == ttnn.uint8 else torch.bfloat16
    if tdt == torch.uint8:
        src = torch.randint(0, 256, (sp * tp, R, width), dtype=torch.uint8, generator=g)
        fill = torch.full((n * R, width), 0xA5, dtype=torch.uint8)
    else:
        src = torch.randn(sp * tp, R, width, generator=g).to(torch.bfloat16)
        fill = torch.full((n * R, width), -7.0, dtype=torch.bfloat16)
    x = ttnn.from_torch(
        src.reshape(1, 1, sp * tp * R, width),
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=2),
    )
    out = ttnn.from_torch(
        fill.reshape(1, 1, n * R, width),
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    for a, b in ranges:
        ttnn.bringup.high_bw_all_gather(
            x,
            2,
            out,
            cluster_axis=axis,
            gathered_dim_size=n * (b - a),
            gathered_dim_start=a if a > 0 else None,
            ready_semaphore=sems[0],
            data_valid_semaphore=sems[1],
            **(sub or {}),
        )
    ttnn.synchronize_device(mesh)
    bad = []
    for q, got in enumerate(_host_out(mesh, out, n * R, width)):
        r, c = divmod(q, tp)
        if axis is None:
            ranks = list(range(sp * tp))
        elif axis == 0:
            ranks = [rr * tp + c for rr in range(sp)]
        else:
            ranks = [r * tp + cc for cc in range(tp)]
        want = fill.clone()
        for slot, chip in enumerate(ranks):
            for a, b in ranges:
                want[slot * R + a : slot * R + b] = src[chip, a:b]
        if not torch.equal(got.view(torch.uint8) if tdt == torch.uint8 else got, want):
            diff = (got != want).any(dim=-1).nonzero().flatten()
            bad.append((q, int(diff.numel()), [int(v) for v in diff[:4]]))
    ttnn.deallocate(x)
    ttnn.deallocate(out)
    return bad


@mesh_parametrize(timeout_s=900)
def test_range_gather(mesh_device):
    if mesh_device is None:
        pytest.skip("device test")
    import ttnn

    mesh = mesh_device
    sp, tp = tuple(mesh.shape)
    if sp * tp < 2:
        pytest.skip("a gather needs two chips")
    sems = _sems(mesh, None)
    cases = [
        # (axis, R, width, ranges, dtype): smallest first
        (None, 64, 320, [(32, 64)], ttnn.uint8),
        (None, 64, 320, [(0, 32), (32, 64)], ttnn.uint8),
        (None, 256, 320, [(7, 101)], ttnn.uint8),  # odd start / end: off the bank boundaries
        (None, 1024, 320, [(0, 300), (300, 640), (640, 1000)], ttnn.uint8),  # pieces composing a prefix
        (None, 128, 512, [(40, 120)], ttnn.bfloat16),
        (0, 256, 320, [(96, 200)], ttnn.uint8),  # the SP axis
        (None, 4096, 320, [(1024, 4000)], ttnn.uint8),  # many chunks a lane
    ]
    fails = []
    for k, (axis, R, width, ranges, dtype) in enumerate(cases):
        bad = _case(mesh, axis, R, width, ranges, dtype, seed=k)
        print(f"[range_gather] axis {axis} R {R} width {width} ranges {ranges} {dtype}: bad chips {bad[:4]}")
        if bad:
            fails.append((k, bad[:4]))
    # the default (no start) against the stock op: the same prefix, byte for byte
    g = torch.Generator().manual_seed(99)
    n, R = sp * tp, 512
    src = torch.randint(0, 256, (1, 1, n * R, 320), dtype=torch.uint8, generator=g)
    x = ttnn.from_torch(
        src,
        dtype=ttnn.uint8,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=2),
    )
    outs = []
    for op in (ttnn.experimental.high_bw_all_gather, ttnn.bringup.high_bw_all_gather):
        o = ttnn.from_torch(
            torch.zeros(1, 1, n * R, 320, dtype=torch.uint8),
            dtype=ttnn.uint8,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        op(x, 2, o, cluster_axis=None, gathered_dim_size=n * 200, ready_semaphore=sems[0], data_valid_semaphore=sems[1])
        outs.append(o)
    ttnn.synchronize_device(mesh)
    same = all(
        torch.equal(a, b) for a, b in zip(_host_out(mesh, outs[0], n * R, 320), _host_out(mesh, outs[1], n * R, 320))
    )
    print(f"[range_gather] default (no start) == stock prefix: {same}")
    if not same:
        fails.append(("default", "differs from stock"))
    for t in outs + [x]:
        ttnn.deallocate(t)
    # on the gather column (the model's sub-device, where the grid has one beside the matmul grid)
    try:
        from models.demos.deepseek_v41_flash_d_p.tt import cores
        from models.demos.deepseek_v41_flash_d_p.tt.gather_hide import Split

        cores.init_caps(mesh)
        split = Split.of(mesh)
    except Exception as e:  # noqa: BLE001
        split = None
        print(f"[range_gather] no gather sub-device: {e}")
    if split is not None:
        mesh.load_sub_device_manager(split.mgr)
        try:
            bad = _case(mesh, None, 2048, 320, [(0, 700), (700, 1500), (1500, 2048)], ttnn.uint8, split.sub, 7)
        finally:
            mesh.clear_loaded_sub_device_manager()
        print(f"[range_gather] on the gather column: bad chips {bad[:4]}")
        if bad:
            fails.append(("column", bad[:4]))
    # refusals
    x = ttnn.from_torch(
        torch.zeros(1, 1, n * 64, 320, dtype=torch.uint8),
        dtype=ttnn.uint8,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=2),
    )
    o = ttnn.from_torch(
        torch.zeros(1, 1, n * 64, 320, dtype=torch.uint8),
        dtype=ttnn.uint8,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    refused = []
    for kw, text in (
        (dict(gathered_dim_start=32), "gathered_dim_start needs gathered_dim_size"),
        (dict(gathered_dim_start=40, gathered_dim_size=n * 32), "exceeds the local length"),
    ):
        try:
            ttnn.bringup.high_bw_all_gather(
                x, 2, o, cluster_axis=None, ready_semaphore=sems[0], data_valid_semaphore=sems[1], **kw
            )
            refused.append((kw, "not refused"))
        except RuntimeError as e:
            if text not in str(e):
                refused.append((kw, str(e)[:200]))
    ttnn.synchronize_device(mesh)
    ttnn.deallocate(x)
    ttnn.deallocate(o)
    print(f"[range_gather] refusals wrong: {refused}")
    assert not fails and not refused, (fails, refused)
