# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""ttnn.bringup.mhc_pre_xing(..., hc_output=, y_output=) and mhc_pre_xing_pack(..., output_tensor=)
(deepseek_v41_flash_d_p P.16): the op writes the caller's buffers instead of allocating its outputs. On the whole box
(FABRIC_2D), random per-chip inputs, DeepSeek-V4.1's call (deepseek=True, collapse_pre, fp32 and bf16 streams):
  * hc and y are byte-identical on every chip to the same call without the options, and live in the given buffers;
  * either buffer alone works (the other output allocated);
  * the buffers reused by a second call with other inputs hold the second result (a program-cache hit), and their
    marker contents are overwritten (negative control);
  * pack: the given buffer holds the packed row, byte-identical;
  * refusals: a buffer of another spec, an input as the output, a y buffer without streams."""

import importlib.util
from pathlib import Path

import pytest
import torch

import ttnn

_TESTS = Path(__file__).resolve().parent.parent


def _load(name):
    spec = importlib.util.spec_from_file_location(f"bringup_mhc_pre_ttnn_unit_{name}", _TESTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ref = _load("reference")
N, NG = 4, 24
SCALE = [0.06, 0.012, 0.06]


@pytest.fixture(scope="module")
def mesh():
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_2D)
    m = ttnn.open_mesh_device(l1_small_size=24576)
    yield m
    ttnn.close_mesh_device(m)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def _dev(t, mesh, dtype=ttnn.float32):
    return ttnn.from_torch(
        t.float().contiguous(),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
    )


def _bits(t):
    return [
        c.view(torch.int16) if c.dtype == torch.bfloat16 else c.view(torch.int32)
        for c in map(ttnn.to_torch, ttnn.get_device_tensors(t))
    ]


def _same(a, b):
    for x, y in zip(_bits(a), _bits(b)):
        assert torch.equal(x, y)


def _inputs(nd, T, C, hidden, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(nd, 1, T, N * C, generator=g)
    full = torch.randn(nd, 1, T, N * hidden, generator=g)
    fn = torch.randn(NG, N * hidden, generator=g) * (N * hidden) ** -0.5
    row = torch.zeros(nd, 1, T, 32)
    row[..., :NG] = full @ fn.T
    row[..., NG] = full.square().sum(-1)
    base = (torch.randn(NG, generator=g) * 2).tolist()
    row2 = torch.zeros(nd, 1, T, 32)
    row2[..., :NG] = torch.randn(nd, 1, T, N * hidden, generator=g) @ fn.T
    row2[..., NG] = (N * hidden) * (1 + torch.rand(nd, 1, T, generator=g))
    prev = ref.mhc_pre_xing_coefficients(row2, SCALE, base, float(N * hidden), N, 1e-6, 1e-6, 20, (-30, 30), True)
    return x, row, base, prev.float()


def _call(row_d, x_d, base, hidden, **kw):
    return ttnn.bringup.mhc_pre_xing(
        row_d,
        x_d,
        scale=SCALE,
        base=base,
        norm_width=float(N * hidden),
        n=N,
        norm_eps=1e-6,
        hc_eps=1e-6,
        sinkhorn_iters=20,
        deepseek=True,
        **kw,
    )


# (T, C, hidden, stream dtype): DeepSeek-V4.1 on 8x4 (640 rows a chip, H/TP = 1280, bf16 streams) and fp32 streams
@pytest.mark.parametrize(
    "T, C, hidden, sdt",
    [(640, 1280, 5120, ttnn.bfloat16), (64, 256, 512, ttnn.float32)],
    ids=["ds41_8x4_bf16", "small_fp32"],
)
def test_output_buffers(mesh, expect_error, T, C, hidden, sdt):
    nd = mesh.get_num_devices()
    hco = _dev(torch.full((nd, 1, T, NG), 1.0e4), mesh)
    yo = _dev(torch.full((nd, 1, T, C), 1.0e4), mesh, sdt)
    for seed in (1, 2):  # the second call reuses both buffers (a cache hit)
        x, row, base, prev = _inputs(nd, T, C, hidden, seed=seed + T)
        row_d, x_d, prev_d = _dev(row, mesh), _dev(x, mesh, sdt), _dev(prev, mesh)
        hc, y = _call(row_d, x_d, base, hidden, collapse_pre=prev_d)
        hc2, y2 = _call(row_d, x_d, base, hidden, collapse_pre=prev_d, hc_output=hco, y_output=yo)
        assert hc2.buffer_address() == hco.buffer_address() and y2.buffer_address() == yo.buffer_address()
        _same(hco, hc)
        _same(yo, y)
    # one buffer alone
    _, y3 = _call(row_d, x_d, base, hidden, collapse_pre=prev_d, y_output=yo)
    _same(y3, y)
    hc4, _ = _call(row_d, x_d, base, hidden, collapse_pre=prev_d, hc_output=hco)
    _same(hc4, hc)
    # refusals
    with expect_error(RuntimeError, "is not the op's"):
        _call(row_d, x_d, base, hidden, collapse_pre=prev_d, hc_output=yo)
    with expect_error(RuntimeError, "a given output is an input"):
        _call(row_d, x_d, base, hidden, collapse_pre=prev_d, hc_output=prev_d)
    with expect_error(RuntimeError, "computes no y"):
        _call(row_d, None, base, hidden, y_output=yo)


def test_pack_output(mesh, expect_error):
    nd, T, C = mesh.get_num_devices(), 640, 1280
    g = torch.Generator().manual_seed(11)
    mix = torch.zeros(nd, 1, T, 32)
    mix[..., :NG] = torch.randn(nd, 1, T, NG, generator=g)
    mix_d, x_d = _dev(mix, mesh), _dev(torch.randn(nd, 1, T, N * C, generator=g), mesh)
    want = ttnn.bringup.mhc_pre_xing_pack(mix_d, x_d, n=N)
    out = _dev(torch.full((nd, 1, T, 32), 1.0e4), mesh)
    got = ttnn.bringup.mhc_pre_xing_pack(mix_d, x_d, n=N, output_tensor=out)
    assert got.buffer_address() == out.buffer_address()
    _same(out, want)
    with expect_error(RuntimeError, "a given output is an input"):
        ttnn.bringup.mhc_pre_xing_pack(mix_d, x_d, n=N, output_tensor=mix_d)
