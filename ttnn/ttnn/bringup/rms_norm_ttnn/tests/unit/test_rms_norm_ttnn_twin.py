# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""ttnn.bringup.rms_norm's split-program options of deepseek_v41_flash_d_p P.21 (CHANGELOG: pack_precise, bfp8_twin).
On the whole box (FABRIC_2D), random per-chip inputs, DeepSeek-V4.1's 8x4 per-chip shapes, the Galaxy's 11 x 10 grid:

  pack_precise=True, dtype=bfloat8_b: the output equals ttnn.typecast(the same norm's float32 output, bfloat8_b) on
      every chip, bit for bit (one rounding from the fp32 DEST, as typecast packs); negative control: without the
      option the default bfp8 pack differs (it rounds twice, DS41-137), so the option is not a no-op.
  bfp8_twin=t (bf16 output): t equals ttnn.typecast(the bf16 output, bfloat8_b) bit for bit; the bf16 output is the
      one the call without the option writes, bit for bit; a second call with other inputs (a program-cache hit)
      rewrites t; negative control: t was filled with a marker first.
  refusals: a twin with a non-bf16 output, of another shape or dtype, with stats_out.
"""

import pytest
import torch

import ttnn

D = ttnn.DRAM_MEMORY_CONFIG


@pytest.fixture(scope="module")
def mesh():
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_2D)
    m = ttnn.open_mesh_device(l1_small_size=24576)
    yield m
    ttnn.close_mesh_device(m)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def _grid(mesh):
    g = mesh.compute_with_storage_grid_size()
    return ttnn.CoreCoord(min(g.x, 11), min(g.y, 10))  # (agent rule 18: at most 110 cores an op on a Galaxy)


def _dev(mesh, t, dtype):
    # (built on the host, then copied: from_torch with a device converts on the whole grid, DS41-177)
    h = ttnn.from_torch(
        t.contiguous(), dtype=dtype, layout=ttnn.TILE_LAYOUT, mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0)
    )
    return ttnn.to_device(h, mesh, memory_config=D)


def _typecast(mesh, t, dtype):
    """ttnn.typecast on the capped grid (the model's call: tt/cores.py / the MoE head's sub_core_grids)."""
    g = _grid(mesh)
    cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(g.x - 1, g.y - 1))})
    return ttnn.typecast(t, dtype, memory_config=D, sub_core_grids=cores)


def _chips(t):
    return [ttnn.to_torch(c) for c in ttnn.get_device_tensors(t)]


def _ckc(mesh):
    return ttnn.init_device_compute_kernel_config(
        mesh.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True
    )


def _bits(x):
    return x.view(torch.int16) if x.dtype == torch.bfloat16 else x.float().view(torch.int32)


def _same(a, b):
    for x, y in zip(_chips(a), _chips(b)):
        assert torch.equal(_bits(x), _bits(y)), (x.flatten()[:8], y.flatten()[:8])


def _differs(a, b) -> bool:
    return any(not torch.equal(_bits(x), _bits(y)) for x, y in zip(_chips(a), _chips(b)))


def _marker(mesh, rows, width, dtype, fill=1.0e4):
    return _dev(mesh, torch.full([mesh.get_num_devices(), 1, rows, width], fill), dtype)


def _inputs(mesh, rows, width, in_dtype, wwidth, seed):
    nd = mesh.get_num_devices()
    g = torch.Generator().manual_seed(seed)
    w = _dev(mesh, (torch.rand(1, 1, 1, wwidth, generator=g) + 0.5).expand(nd, -1, -1, -1), ttnn.float32)
    x = _dev(mesh, torch.randn(nd, 1, rows, width, generator=g) * 3.0, in_dtype)
    return x, w


# (rows, input width, input dtype, window): attn_norm on the chip's whole rows (LOCAL mode), q_norm on the fused
# projection's wq_a window (with the softmax scale folded into its gamma: any positive gamma does here)
PRECISE = [
    pytest.param(160, 4096, ttnn.bfloat16, None, id="attn_norm_rows"),
    pytest.param(160, 1792, ttnn.bfloat16, (0, 1280), id="q_norm_window"),
    pytest.param(160, 1792, ttnn.float32, (0, 1280), id="q_norm_window_fp32_in"),
]


@pytest.mark.parametrize("rows, width, in_dtype, window", PRECISE)
def test_pack_precise(mesh, rows, width, in_dtype, window):
    c0, c1 = window or (0, width)
    x, w = _inputs(mesh, rows, width, in_dtype, c1 - c0, seed=11)
    kw = dict(epsilon=1e-6, weight=w, compute_kernel_config=_ckc(mesh), memory_config=D, max_grid=_grid(mesh))
    if window is not None:
        kw["input_window"] = window
    y32 = ttnn.bringup.rms_norm(x, dtype=ttnn.float32, **kw)
    want = _typecast(mesh, y32, ttnn.bfloat8_b)
    got = ttnn.bringup.rms_norm(x, dtype=ttnn.bfloat8_b, pack_precise=True, **kw)
    _same(got, want)
    default = ttnn.bringup.rms_norm(x, dtype=ttnn.bfloat8_b, **kw)
    assert _differs(default, want), "the default bfp8 pack equals the precise one: pack_precise not exercised"


# (rows, input width, input dtype, window, split): ffn_norm's column split (partial_stats over 4 chips' columns, the
# MoE block's x, NORM_INTO_MOE), q_norm's window in the indexer layers (the indexer's q_resid, NORM_GEMM_DTYPE)
TWIN = [
    pytest.param(640, 1280, ttnn.bfloat16, None, True, id="ffn_norm_split"),
    pytest.param(160, 1792, ttnn.bfloat16, (0, 1280), False, id="q_norm_window"),
    pytest.param(160, 1792, ttnn.float32, (0, 1280), False, id="q_norm_window_fp32_in"),
]


@pytest.mark.parametrize("rows, width, in_dtype, window, split", TWIN)
def test_bfp8_twin(mesh, rows, width, in_dtype, window, split):
    c0, c1 = window or (0, width)
    ckc = _ckc(mesh)
    twin = _marker(mesh, rows, c1 - c0, ttnn.bfloat8_b)
    for seed in (5, 6):  # (the second call: a program-cache hit with other inputs)
        x, w = _inputs(mesh, rows, width, in_dtype, c1 - c0, seed=seed)
        kw = dict(epsilon=1e-6, weight=w, compute_kernel_config=ckc, memory_config=D, max_grid=_grid(mesh))
        kw["dtype"] = ttnn.bfloat16
        if window is not None:
            kw["input_window"] = window
        if split:
            part = ttnn.bringup.rms_norm(
                x, stats_out=True, compute_kernel_config=ckc, memory_config=D, max_grid=_grid(mesh)
            )
            kw.update(partial_stats=part, stats_width=width * 4)  # (the other chips' partials: a scaled statistic)
        plain = ttnn.bringup.rms_norm(x, **kw)
        y = ttnn.bringup.rms_norm(x, bfp8_twin=twin, **kw)
        _same(y, plain)
        _same(twin, _typecast(mesh, plain, ttnn.bfloat8_b))


def test_twin_refusals(mesh, expect_error):
    rows, width = 160, 1280
    x, w = _inputs(mesh, rows, width, ttnn.bfloat16, width, seed=3)
    kw = dict(epsilon=1e-6, weight=w, compute_kernel_config=_ckc(mesh), memory_config=D, max_grid=_grid(mesh))
    twin = _marker(mesh, rows, width, ttnn.bfloat8_b)
    with expect_error(RuntimeError, "bfp8_twin needs a bfloat16 output"):
        ttnn.bringup.rms_norm(x, dtype=ttnn.float32, bfp8_twin=twin, **kw)
    with expect_error(RuntimeError, "bfp8_twin shape"):
        ttnn.bringup.rms_norm(x, dtype=ttnn.bfloat16, bfp8_twin=_marker(mesh, rows, 2 * width, ttnn.bfloat8_b), **kw)
    with expect_error(RuntimeError, "bfp8_twin must be bfloat8_b"):
        ttnn.bringup.rms_norm(x, dtype=ttnn.bfloat16, bfp8_twin=_marker(mesh, rows, width, ttnn.bfloat16), **kw)
    with expect_error(RuntimeError, "bfp8_twin takes no stats_out"):
        ttnn.bringup.rms_norm(
            x, stats_out=True, compute_kernel_config=_ckc(mesh), memory_config=D, bfp8_twin=twin, max_grid=_grid(mesh)
        )
