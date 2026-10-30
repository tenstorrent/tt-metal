# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""ttnn.bringup.rotary_embedding_indexed(uniform=True): ONE program for the whole mesh instead of one per coordinate
(the fork's one change, DeepSeek-V4.1 P.5). Valid for per-chip pre-sliced cos / sin read from row 0
(kv_actual_global == 0, no seq_subshard_axis): every chip gets different input rows and different cos / sin rows, and
the output must equal, byte for byte on every chip, the stock op's per-coordinate programs and a torch reference of the
Meta-order rotation. The refusals (kv_actual_global != 0, seq_subshard_axis) are checked on a cache miss and on a hit.
The host time a call (uniform vs stock, warm) is printed, not gated."""

import time

import pytest
import torch

import ttnn

R = 64  # rotary width (DeepSeek-V4.1)


def _mesh_cases():
    # The case runs on the box's own mesh (a smaller mesh on a bigger box fails the 2D fabric handshake)
    return [pytest.param(m, id=f"{m[0]}x{m[1]}") for m in ((8, 4), (2, 4), (2, 2))]


DEVICE_PARAMS = [{"fabric_config": ttnn.FabricConfig.FABRIC_2D, "require_exact_physical_num_devices": True}]

# DeepSeek-V4.1's rope_stock calls on 8x4 (rows a chip, heads, input width, prefix width): attention q with the nope
# prepended, the attention output's inverse rope, window kv, compressor ckv / k (320 rows), indexer q.
CASES = [
    pytest.param(160, 64, 64, 448, False, id="attn_q_prefix"),
    pytest.param(160, 64, 512, 0, True, id="attn_o_inverse"),
    pytest.param(160, 1, 512, 0, False, id="kv"),
    pytest.param(320, 1, 512, 0, False, id="compressor_ckv"),
    pytest.param(320, 1, 128, 0, False, id="compressor_k"),
    pytest.param(160, 32, 64, 64, False, id="indexer_q_prefix"),
]


def _trans(inverse: bool) -> torch.Tensor:
    t = torch.zeros(1, 1, 32, 32)
    t[..., torch.arange(0, 32, 2), torch.arange(1, 32, 2)] = 1.0
    t[..., torch.arange(1, 32, 2), torch.arange(0, 32, 2)] = -1.0
    return t.transpose(-1, -2).contiguous() if inverse else t


def _ref(x, cos, sin, inverse):
    """Meta-order rotation of the last R columns: pairs (e, o) -> (e c - o s, o c + e s) (inverse: -s)."""
    s = -sin if inverse else sin
    tail = x[..., -R:]
    e, o = tail[..., 0::2], tail[..., 1::2]
    c2, s2 = cos[..., 0::2], s[..., 0::2]
    re, ro = e * c2 - o * s2, o * c2 + e * s2
    out = x.clone()
    out[..., -R:] = torch.stack([re, ro], dim=-1).flatten(-2)
    return out


def _put(mesh, t, shard: bool):
    """t [n_chips, ...] chip-linear (row-major over the mesh) when shard, else one tensor replicated."""
    if shard:
        rows, cols = tuple(mesh.shape)
        t = t.reshape(rows, cols, *t.shape[1:])
        t = t.reshape(rows, cols * t.shape[2], *t.shape[3:])  # [rows, cols B, ...]
        mapper = ttnn.ShardTensor2dMesh(mesh, mesh_shape=(rows, cols), dims=(0, 1))
    else:
        mapper = ttnn.ReplicateTensorToMesh(mesh)
    return ttnn.from_torch(
        t.contiguous(),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mapper,
    )


def _chips(mesh, t) -> list[torch.Tensor]:
    return [ttnn.to_torch(c) for c in ttnn.get_device_tensors(t)]


def _inputs(mesh, rows, heads, width, prefix_w, inverse):
    n = mesh.get_num_devices()
    g = torch.Generator().manual_seed(1234 + rows + heads + width + prefix_w)
    x = torch.randn(n, heads, rows, width, generator=g).bfloat16().float()
    ang = torch.rand(n, 1, rows, R // 2, generator=g) * 6.28
    cos = torch.cos(ang).repeat_interleave(2, dim=-1).bfloat16().float()
    sin = torch.sin(ang).repeat_interleave(2, dim=-1).bfloat16().float()
    pre = torch.randn(n, heads, rows, prefix_w, generator=g).bfloat16().float() if prefix_w else None
    dev = dict(
        x=_put(mesh, x, True),
        cos=_put(mesh, cos, True),
        sin=_put(mesh, sin, True),
        trans=_put(mesh, _trans(inverse), False),
        pre=_put(mesh, pre, True) if prefix_w else None,
    )
    return (x, cos, sin, pre), dev


def _call(op, d, uniform=None, kv=0, **kw):
    extra = {} if uniform is None else {"uniform": uniform}
    return op(
        d["x"],
        d["cos"],
        d["sin"],
        d["trans"],
        kv,
        0,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        rotary_dim=R,
        rotary_offset=d["x"].shape[-1] - R,
        concat_prefix=d["pre"],
        **extra,
        **kw,
    )


STOCK = ttnn.experimental.deepseek_prefill.rotary_embedding_indexed


@pytest.mark.parametrize("mesh_device", _mesh_cases(), indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("rows, heads, width, prefix_w, inverse", CASES)
def test_uniform_matches_stock(mesh_device, rows, heads, width, prefix_w, inverse):
    mesh = mesh_device
    (x, cos, sin, pre), d = _inputs(mesh, rows, heads, width, prefix_w, inverse)
    stock = _chips(mesh, _call(STOCK, d))
    for rep in range(2):  # a miss, then a hit (addresses patched on the one program)
        got = _chips(mesh, _call(ttnn.bringup.rotary_embedding_indexed, d, uniform=True))
        for i, (a, b) in enumerate(zip(got, stock)):
            assert torch.equal(a, b), f"chip {i} rep {rep}: uniform differs from the stock op"
    off = _chips(mesh, _call(ttnn.bringup.rotary_embedding_indexed, d))  # option off: the stock program
    for i, (a, b) in enumerate(zip(off, stock)):
        assert torch.equal(a, b), f"chip {i}: uniform=False differs from the stock op"
    for i, a in enumerate(got):
        want = _ref(x[i : i + 1], cos[i : i + 1], sin[i : i + 1], inverse)
        if pre is not None:
            want = torch.cat([pre[i : i + 1], want], dim=-1)
        tail = slice(-R, None)
        err = (a[..., tail] - want[..., tail]).abs().max().item()
        assert err <= 0.04 * want[..., tail].abs().max().item(), f"chip {i}: tail max |diff| {err}"
        assert torch.equal(a[..., : a.shape[-1] - R], want[..., : want.shape[-1] - R]), f"chip {i}: copied columns"


@pytest.mark.parametrize("mesh_device", _mesh_cases(), indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_uniform_refusals(mesh_device, expect_error):
    mesh = mesh_device
    _, d = _inputs(mesh, 160, 1, 512, 0, False)
    with expect_error(RuntimeError, "uniform=True requires kv_actual_global == 0"):
        _call(ttnn.bringup.rotary_embedding_indexed, d, uniform=True, kv=32)
    ttnn.deallocate(_call(ttnn.bringup.rotary_embedding_indexed, d, uniform=True))  # cached now
    with expect_error(RuntimeError, "uniform=True requires kv_actual_global == 0"):  # the hit path checks too
        _call(ttnn.bringup.rotary_embedding_indexed, d, uniform=True, kv=64)
    with expect_error(RuntimeError, "uniform=True does not support seq_subshard_axis"):
        _call(ttnn.bringup.rotary_embedding_indexed, d, uniform=True, seq_subshard_axis=1)


@pytest.mark.parametrize("mesh_device", _mesh_cases(), indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_uniform_host_time(mesh_device):
    """Warm host time a call (dispatch only, then one synchronize), printed: uniform vs the stock op."""
    mesh = mesh_device
    _, d = _inputs(mesh, 160, 64, 64, 448, False)
    res = {}
    for name, op, kw in (
        ("stock", STOCK, {}),
        ("uniform", ttnn.bringup.rotary_embedding_indexed, {"uniform": True}),
        ("stock", STOCK, {}),
        ("uniform", ttnn.bringup.rotary_embedding_indexed, {"uniform": True}),
    ):
        outs = [_call(op, d, **kw) for _ in range(5)]
        ttnn.synchronize_device(mesh)
        for o in outs:
            ttnn.deallocate(o)
        n = 50
        t0 = time.perf_counter()
        outs = [_call(op, d, **kw) for _ in range(n)]
        t1 = time.perf_counter()
        ttnn.synchronize_device(mesh)
        for o in outs:
            ttnn.deallocate(o)
        res[name] = (t1 - t0) / n * 1e6
    print(
        f"\nrotary_embedding_indexed host us a call on {tuple(mesh.shape)}: stock {res['stock']:.1f}, uniform {res['uniform']:.1f}"
    )
