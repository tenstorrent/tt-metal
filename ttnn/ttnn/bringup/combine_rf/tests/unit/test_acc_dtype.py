# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""sparse_combine_rf acc_dtype (deepseek_v41_flash_d_p P.9): the home tokens' running sums in bf16 instead of fp32.

- test_accumulator_bytes_bf16 (host): the sums take half the L1 (W x 2 B a token); DeepSeek-V4.1's 1024 home tokens per
  chip (W 5120, 9 accumulators) fit a 1356 KB arena in one call with bf16 sums, not with fp32.
- test_refusal_bf16 (mesh): an arena too small for the bf16 sums is refused with the largest S and the smallest A that
  fit, and those numbers are right: S = s_fit builds, s_fit + A does not; the same arena's fp32 limit is about half.
- test_acc_precision (mesh): the standalone combine (rows in DRAM, every row summed onto its home token, whole 2x2
  mesh) with fp32 vs bf16 sums on the same rows: each vs the float64 sum of the bf16 rows, and bf16 vs fp32 (rel
  error, bias, worst row).
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import fabric2d_device_params
from ttnn.bringup.combine_rf.sparse_combine_rf import accumulator_bytes, sparse_combine_rf_prepare

W = 5120  # DeepSeek-V4.1 hidden (one 10240 B bf16 row a fabric packet)
PAYLOAD = 10880  # row + 32 B trailer, as the model opens its mesh (deepseek_v41 tt/weighted_combine.py)
PARAMS = pytest.mark.parametrize(
    "mesh_device, device_params",
    [pytest.param((2, 2), fabric2d_device_params(fabric_payload_size=PAYLOAD, l1_small_size=24576), id="2x2")],
    indirect=["mesh_device", "device_params"],
)


def test_accumulator_bytes_bf16():
    cap = 1356 * 1024  # the model's ARENA_MAX on the 2x2 p300c grid
    for S in (64, 512, 1024):
        f = accumulator_bytes(2, 9, W, S, out_dtype=ttnn.bfloat16)
        b = accumulator_bytes(2, 9, W, S, out_dtype=ttnn.bfloat16, acc_dtype=ttnn.bfloat16)
        maxt = -(-S // 9)
        assert f - b == maxt * (W // 1024) * 2048, (S, f, b)  # (NT = W / 1024 tiles a row, 2048 B saved a tile)
    assert accumulator_bytes(2, 9, W, 1024, depth=4, local_slots=8, acc_dtype=ttnn.bfloat16) <= cap
    assert accumulator_bytes(2, 9, W, 1024, depth=1, acc_dtype=ttnn.float32) > cap
    largest = max(S for S in range(9, 2048, 9) if accumulator_bytes(2, 9, W, S, acc_dtype=ttnn.bfloat16) <= cap)
    largest32 = max(S for S in range(9, 2048, 9) if accumulator_bytes(2, 9, W, S, acc_dtype=ttnn.float32) <= cap)
    logger.info(f"ACC_DTYPE largest S per call at 9 accumulators, 1356 KB: bf16 {largest}, fp32 {largest32}")
    print(f"ACC_DTYPE largest S per call at 9 accumulators, 1356 KB: bf16 {largest}, fp32 {largest32}")
    assert largest >= 1024 > largest32


def _arena(mesh, kb):
    grid = mesh.compute_with_storage_grid_size()
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
    rows = kb * 1024 // 64
    return ttnn.allocate_tensor_on_device(
        ttnn.Shape([grid.x * grid.y * rows, 16]),
        ttnn.uint32,
        ttnn.ROW_MAJOR_LAYOUT,
        mesh,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(crs, [rows, 16], ttnn.ShardOrientation.ROW_MAJOR),
        ),
    )


def _prep(mesh, S, arena, acc_dtype):
    R, C = tuple(mesh.shape)
    return sparse_combine_rf_prepare(
        mesh,
        S,
        W,
        {(r, c): [] for r in range(R) for c in range(C)},
        num_links=2,
        n_acc=9,
        depth=4,
        arena=arena,
        writers=1,
        l1=dict(writers=[(0, 0)], words=0),
        piece_max=2 * W,
        axis=0,
        out_dtype=ttnn.bfloat16,
        acc_dtype=acc_dtype,
    )


def _s_fit(err):
    msg = str(err)
    return int(msg.split("Call with <= ")[1].split(" ")[0]), int(msg.split("or with >= ")[1].split(" ")[0])


@pytest.mark.timeout(600)
@PARAMS
def test_refusal_bf16(mesh_device, device_params, expect_error):
    arena = _arena(mesh_device, 1156)
    with expect_error(ValueError, "home tokens over 9 accumulators") as e:
        _prep(mesh_device, 1024, arena, ttnn.bfloat16)
    s_fit, a_min = _s_fit(e.value)
    print(f"ACC_REFUSE bf16: {e.value}")
    assert "bf16 tiles" in str(e.value)
    assert 9 < s_fit < 1024 and s_fit % 9 == 0, s_fit
    _prep(mesh_device, s_fit, arena, ttnn.bfloat16)  # (the stated limit builds)
    with expect_error(ValueError, "home tokens over"):
        _prep(mesh_device, s_fit + 9, arena, ttnn.bfloat16)  # (one more token an accumulator does not)
    assert a_min is not None and a_min > 9
    with expect_error(ValueError, "home tokens over") as e32:
        _prep(mesh_device, 1024, arena, ttnn.float32)
    s32, _ = _s_fit(e32.value)
    print(f"ACC_REFUSE fp32 (same arena): {e32.value}")
    assert "fp32 tiles" in str(e32.value)
    assert s32 <= s_fit // 2 + 9, (s32, s_fit)
    ttnn.deallocate(arena)


def _rows(R, C, S, K, g):
    """Per home chip and token, K rows from random source chips (the whole mesh): rows_by_chip and the y values."""
    chips = [(r, c) for r in range(R) for c in range(C)]
    rows = {ch: [] for ch in chips}
    vals = {ch: [] for ch in chips}
    for h in chips:
        for t in range(S):
            for _ in range(K):
                src = chips[int(torch.randint(0, len(chips), (1,), generator=g))]
                rows[src].append((len(rows[src]), h, t, 0))
                # rows of a token span 2^-6 .. 2^6 in scale (the combine's weighted expert rows differ by the weights)
                v = torch.randn(W, generator=g) * 2.0 ** float(torch.randint(-6, 7, (1,), generator=g))
                vals[src].append(v.to(torch.bfloat16))
    return rows, vals


@pytest.mark.timeout(900)
@PARAMS
@pytest.mark.parametrize("S", [72, 288])
def test_acc_precision(mesh_device, device_params, S):
    R, C = tuple(mesh_device.shape)
    K = 6
    g = torch.Generator().manual_seed(11)
    rows, vals = _rows(R, C, S, K, g)
    n_y = max(len(v) for v in vals.values())
    y_host = torch.zeros(R, C, n_y, W, dtype=torch.bfloat16)
    ref = torch.zeros(R, C, S, W, dtype=torch.float64)
    for (r, c), lst in rows.items():
        for (yrow, h, t, _), v in zip(lst, vals[(r, c)]):
            y_host[r, c, yrow] = v
            ref[h[0], h[1], t] += v.double()
    shard = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(R, C), dims=(0, 1))
    y = ttnn.from_torch(
        y_host,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=shard,
    )
    outs = {}
    for name, acc in (("fp32", ttnn.float32), ("bf16", ttnn.bfloat16)):
        out = ttnn.from_torch(
            torch.zeros(R, C, S, W),
            dtype=ttnn.float32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=shard,
        )
        run = sparse_combine_rf_prepare(
            mesh_device,
            S,
            W,
            rows,
            num_links=2,
            n_acc=9,
            depth=4,
            piece_max=2 * W,
            out_dtype=ttnn.float32,
            acc_unpack_fp32=True,
            acc_dtype=acc,
        )
        got = []
        for _ in range(2):
            run(y, out)
            got.append(
                torch.stack([ttnn.to_torch(d).reshape(S, W).double() for d in ttnn.get_device_tensors(out)]).reshape(
                    R, C, S, W
                )
            )
        outs[name] = got[0]
        nd = int((got[0] != got[1]).sum())
        # (arrival order: rings of different sources interleave per call; the fixed order is the deterministic
        # option, which needs the L1 hand-off: checked in the model's tt/perf_experts.py::test_combine_acc)
        print(f"ACC_PRECISION S={S} acc={name}: two calls (arrival order) differ in {nd} of {got[0].numel()} values")
        ttnn.deallocate(out)

    def stats(a, b):
        d = a - b
        rel = (d.norm() / b.norm()).item()
        bias = ((d * b).sum() / (b * b).sum()).item()
        row = (d.reshape(-1, W).norm(dim=1) / b.reshape(-1, W).norm(dim=1)).max().item()
        return rel, bias, row

    res = {
        "fp32 vs float64": stats(outs["fp32"], ref),
        "bf16 vs float64": stats(outs["bf16"], ref),
        "bf16 vs fp32": stats(outs["bf16"], outs["fp32"]),
        "float64 sum rounded once to bf16 vs float64": stats(ref.to(torch.bfloat16).double(), ref),
    }
    for k, (rel, bias, row) in res.items():
        print(f"ACC_PRECISION S={S} {k}: rel {rel:.3e} bias {bias:+.3e} worst_row {row:.3e}")
    assert res["fp32 vs float64"][0] < 1e-6  # (fp32 sums of bf16 rows, the final copy at fp32: ~exact)
    # bf16: one rounding per added row (K = 6 rows a token): within ~K x the single bf16 rounding (2^-9 rel), unbiased
    one = res["float64 sum rounded once to bf16 vs float64"][0]
    rel, bias, row = res["bf16 vs float64"]
    assert one < rel < 2 * K * one, (rel, one)
    assert abs(bias) < 0.25 * one, f"bf16 sums biased: {bias:+.3e} (one rounding rel {one:.3e})"
    assert row < 2**-5, row
    ttnn.deallocate(y)
