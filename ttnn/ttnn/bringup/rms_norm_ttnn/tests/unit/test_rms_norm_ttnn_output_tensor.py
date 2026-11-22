# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""ttnn.bringup.rms_norm(..., output_tensor=) (deepseek_v41_flash_d_p P.16): the op writes the caller's buffer instead
of allocating its output, in both programs (the default one and the split one: input_window / dtype / stats_out /
partial_stats). On the whole box (FABRIC_2D), random per-chip inputs, DeepSeek-V4.1's 8x4 shapes:
  * the result is byte-identical on every chip to the same call without the option, and lives in the given buffer;
  * the buffer reused by a second call with other inputs holds the second result (a program-cache hit);
  * negative control: the buffer's old contents are overwritten (it was filled with a marker first);
  * refusals: a buffer of another spec, the input itself as the output."""

import pytest
import torch

import ttnn


@pytest.fixture(scope="module")
def mesh():
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_2D)
    m = ttnn.open_mesh_device(l1_small_size=24576)
    yield m
    ttnn.close_mesh_device(m)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def _dev(mesh, t, dtype):
    return ttnn.from_torch(
        t.contiguous(),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
    )


def _chips(t):
    return [ttnn.to_torch(c) for c in ttnn.get_device_tensors(t)]


def _ckc(mesh):
    return ttnn.init_device_compute_kernel_config(
        mesh.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True
    )


def _same(a, b):
    for x, y in zip(_chips(a), _chips(b)):
        assert torch.equal(
            x.view(torch.int16) if x.dtype == torch.bfloat16 else x.view(torch.int32),
            y.view(torch.int16) if y.dtype == torch.bfloat16 else y.view(torch.int32),
        )


def _buffer(mesh, like_shape, dtype, fill=1.0e4):
    """A load-time-style output: [1, 1, R, W] dtype TILE DRAM, filled with a marker (the negative control)."""
    nd = mesh.get_num_devices()
    return _dev(mesh, torch.full([nd, 1, *like_shape[-2:]], fill), dtype)


# (rows, input width, input dtype, window, out dtype): attn_norm (default program), q_norm / kv_norm (window of the fused
# projection, split program), a column-split norm's output (partial_stats) and its stats_out
CASES = [
    pytest.param(160, 5120, ttnn.bfloat16, None, None, id="attn_norm_default"),
    pytest.param(160, 1792, ttnn.float32, (0, 1280), ttnn.bfloat16, id="q_norm_window"),
    pytest.param(160, 1792, ttnn.float32, (1280, 1792), ttnn.bfloat16, id="kv_norm_window"),
]


@pytest.mark.parametrize("rows, width, dtype, window, out_dtype", CASES)
def test_output_tensor_local(mesh, expect_error, rows, width, dtype, window, out_dtype):
    nd = mesh.get_num_devices()
    g = torch.Generator().manual_seed(rows + width)
    c0, c1 = window or (0, width)
    w = _dev(mesh, (torch.rand(nd, 1, 1, c1 - c0, generator=g) + 0.5)[:1].expand(nd, -1, -1, -1), ttnn.float32)
    kw = dict(epsilon=1e-6, weight=w, compute_kernel_config=_ckc(mesh), memory_config=ttnn.DRAM_MEMORY_CONFIG)
    if window is not None:
        kw["input_window"] = window
    if out_dtype is not None:
        kw["dtype"] = out_dtype
    odt = out_dtype or dtype
    xs = [_dev(mesh, torch.randn(nd, 1, rows, width, generator=g) * 3, dtype) for _ in range(2)]
    out = _buffer(mesh, [rows, c1 - c0], odt)
    for x in xs:  # the second call reuses the buffer (a cache hit)
        want = ttnn.bringup.rms_norm(x, **kw)
        got = ttnn.bringup.rms_norm(x, output_tensor=out, **kw)
        assert got.buffer_address() == out.buffer_address()
        _same(got, want)
        _same(out, want)
    bad = _buffer(mesh, [rows, c1 - c0 + 32], odt)
    with expect_error(RuntimeError, "is not the op's output spec"):
        ttnn.bringup.rms_norm(xs[0], output_tensor=bad, **kw)


def test_output_tensor_split(mesh, expect_error):
    """A column-split norm: stats_out into a given buffer, then partial_stats= into another (each byte-identical)."""
    nd, rows, width = mesh.get_num_devices(), 640, 1280
    g = torch.Generator().manual_seed(7)
    ckc = _ckc(mesh)
    w = _dev(mesh, (torch.rand(1, 1, 1, width, generator=g) + 0.5).expand(nd, -1, -1, -1), ttnn.float32)
    x = _dev(mesh, torch.randn(nd, 1, rows, width, generator=g), ttnn.bfloat16)
    part = ttnn.bringup.rms_norm(x, stats_out=True, compute_kernel_config=ckc, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    so = _buffer(mesh, list(part.shape), ttnn.float32)
    got = ttnn.bringup.rms_norm(
        x, stats_out=True, compute_kernel_config=ckc, memory_config=ttnn.DRAM_MEMORY_CONFIG, output_tensor=so
    )
    assert got.buffer_address() == so.buffer_address()
    _same(so, part)
    kw = dict(epsilon=1e-6, weight=w, compute_kernel_config=ckc, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    want = ttnn.bringup.rms_norm(x, partial_stats=part, stats_width=width * 4, **kw)
    out = _buffer(mesh, [rows, width], ttnn.bfloat16)
    got = ttnn.bringup.rms_norm(x, partial_stats=so, stats_width=width * 4, output_tensor=out, **kw)
    assert got.buffer_address() == out.buffer_address()
    _same(out, want)
    with expect_error(RuntimeError, "must not be the input"):
        ttnn.bringup.rms_norm(x, partial_stats=so, stats_width=width * 4, output_tensor=x, **kw)
