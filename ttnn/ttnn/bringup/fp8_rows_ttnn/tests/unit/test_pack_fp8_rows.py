# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""ttnn.bringup.pack_fp8_rows (deepseek_v41_flash_d_p P.39): per_token_cast_to_fp8's e4m3 rows [.., M, 128] + fp32
scales [.., M, 1] -> Blaze decode's 192-byte indexer key cache rows [e4m3 | fp32 scale | 60 zero bytes].

Whole box (FABRIC_2D), random per-chip keys (the index keys' range: normed, a few large channels), bf16 TILE, through
the stock ttnn.experimental.deepseek_prefill.per_token_cast_to_fp8(round_scale_to_power_of_two=False): every byte
equal to the op's outputs (their raw bytes) packed on the host; the RM-input cast (the GLM writer tt-blaze's test takes
as its oracle) gives the same rows; against tt-blaze's torch recipe (_quantize_scaled_keys of
tests/blaze/micro_ops/dsa/test_dsa_indexer_direct_k_cache_update.py @ 8223e7f9963e) the scales are byte-equal and a
value differs by at most one e4m3 step (the device lands one step higher at ~1 % of values and keeps e4m3 subnormals
the recipe flushes: measured, not a failure; tt-blaze compares its writer with the cast, not the recipe); all-zero
and all -0 rows (zeroed pad rows) written all zero; a negative control (one changed scale byte is seen); the prepared
program on fresh buffers (a cache hit), a caller's output buffer, determinism; refusals."""

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


def _grid(mesh):
    g = mesh.compute_with_storage_grid_size()
    return ttnn.CoreCoord(min(g.x, 11), g.y)  # (a Galaxy's 110-core cap, agent rule 18)


def _keys(nd, m, seed):
    g = torch.Generator().manual_seed(seed)
    k = torch.randn(nd, 1, m, 128, generator=g)
    k[..., :, 3] *= 9.0  # a large channel
    k[..., 1, :] *= 1e-6  # a tiny row (values flush below 2^-6 x scale)
    k[..., 2, :] = 0.0  # a pad row
    k[..., 5, :] = -0.0  # a pad row of -0 (a masked negative key)
    k[..., 7, :64] = 0.0  # half a row zero
    return k.to(torch.bfloat16)


def _dev(t, mesh):
    return ttnn.from_torch(
        t.contiguous(),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
    )


def _bytes(t, nd, m):
    """uint8 [nd, m, 192] of a device uint8 tensor."""
    return torch.cat([ttnn.to_torch(x).reshape(1, m, 192) for x in ttnn.get_device_tensors(t)])[:nd]


def _raw(t):
    """The bytes of a mesh tensor's every chip, chip-linear (r cols + c), from its host buffer (to_dtype / to_torch of
    FP8_E4M3 flush e4m3 subnormals): [chips, bytes] uint8."""
    host = t.cpu()
    buf = host.host_buffer()
    rows, cols = tuple(t.device().shape)
    out = []
    for r in range(rows):
        for c in range(cols):
            shard = buf.get_shard(ttnn.MeshCoordinate(r, c))
            out.append(torch.from_dlpack(shard).view(torch.uint8).flatten().clone())
    return torch.stack(out)


def _op_rows(e4m3, sc, nd, m):
    """The cast's own outputs packed on the host from their raw bytes: [nd, m, 192]."""
    v = _raw(e4m3)[:, : m * 128].reshape(nd, m, 128)
    s = _raw(sc)[:, : m * 4].reshape(nd, m, 4)
    raw = torch.zeros(nd, m, 192, dtype=torch.uint8)
    raw[..., :128] = v
    raw[..., 128:132] = s
    zero = ((v & 0x7F) == 0).all(dim=-1)
    raw[zero] = 0
    return raw


def _recipe(keys):
    """tt-blaze _quantize_scaled_keys (verbatim) + all-zero rows for all-zero keys."""
    values = keys.float().contiguous()
    scales = values.abs().amax(dim=-1, keepdim=True).clamp(1e-4, 3e38) * (1 / 448)
    inv = scales.reciprocal().contiguous()
    inv = (inv.view(torch.int32) & -8192).view(torch.float32)
    normalized = (values * inv).contiguous()
    truncated = (normalized.view(torch.int32) & ~((1 << 20) - 1)).view(torch.float32)
    truncated[normalized.abs() < 2**-6] = 0
    raw = torch.zeros((*values.shape[:-1], 192), dtype=torch.uint8)
    raw[..., :128] = truncated.to(torch.float8_e4m3fn).view(torch.uint8)
    raw[..., 128:132] = scales.contiguous().view(torch.uint8)
    raw[(values == 0).all(dim=-1)] = 0
    return raw


def _run(mesh, keys, out=None, max_grid=None):
    x = _dev(keys, mesh)
    e4m3, sc = ttnn.experimental.deepseek_prefill.per_token_cast_to_fp8(x, round_scale_to_power_of_two=False)
    rows = ttnn.bringup.pack_fp8_rows(e4m3, sc, output_tensor=out, max_grid=max_grid or _grid(mesh))
    return x, e4m3, sc, rows


@pytest.mark.parametrize("m", [320, 640, 32, 96])
def test_pack_fp8_rows(mesh, m):
    nd = mesh.get_num_devices()
    keys = _keys(nd, m, seed=m)
    x, e4m3, sc, rows = _run(mesh, keys)
    got = _bytes(rows, nd, m)
    op = _op_rows(e4m3, sc, nd, m)
    bad = (got != op).any(dim=-1)
    assert (
        not bad.any()
    ), f"{int(bad.sum())} rows differ from the cast's outputs, first {torch.nonzero(bad)[:3].tolist()}"
    want = _recipe(keys.reshape(nd, m, 128))
    assert torch.equal(got[..., 128:132], want[..., 128:132]), "scales differ from Blaze's recipe"
    gv = got[..., :128].contiguous().view(torch.float8_e4m3fn).double()
    wv = want[..., :128].contiguous().view(torch.float8_e4m3fn).double()
    step = torch.where(
        wv.abs() >= 2**-6, 2.0 ** (torch.floor(torch.log2(wv.abs().clamp_min(2**-6))) - 3), 2.0**-6
    )
    far = (gv - wv).abs() > step
    diff = int((gv != wv).sum())
    print(
        f"[pack_fp8_rows m {m}] values differing from Blaze's recipe: {diff} of {gv.numel()}, beyond one step: {int(far.sum())}"
    )
    assert not far.any(), f"{int(far.sum())} values beyond one e4m3 step of Blaze's recipe"
    # the RM-input cast (tt-blaze's GLM oracle) writes the same rows (rows <= 96: the RM path takes a core a row)
    if m <= 96:
        xr = ttnn.from_torch(
            keys.contiguous(),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
        )
        er, sr = ttnn.experimental.deepseek_prefill.per_token_cast_to_fp8(xr, round_scale_to_power_of_two=False)
        rr = ttnn.bringup.pack_fp8_rows(er, sr, max_grid=_grid(mesh))
        assert torch.equal(_bytes(rr, nd, m), got), "the RM-input cast's rows differ"
        for t in (xr, er, sr, rr):
            ttnn.deallocate(t)
    assert (got[:, 2] == 0).all() and (got[:, 5] == 0).all(), "pad rows not all zero"
    assert (got[..., 132:] == 0).all(), "padding bytes not zero"
    # negative control: the comparison sees one changed scale byte
    bad_rows = want.clone()
    bad_rows[0, 0, 129] ^= 1
    assert not torch.equal(got, bad_rows)
    # prepared program on fresh buffers (a cache hit) and a caller's output buffer: the same bytes, deterministic
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, m, 192]), ttnn.uint8, ttnn.ROW_MAJOR_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG
    )
    x2, e2, s2, rows2 = _run(mesh, keys, out=out)
    assert rows2 is out
    assert torch.equal(_bytes(rows2, nd, m), got), "a second call (prepared, caller's buffer) differs"
    for t in (x, e4m3, sc, rows, x2, e2, s2, rows2):
        ttnn.deallocate(t)


def test_pack_fp8_rows_host_time(mesh):
    """OV-23: the prepared dispatch's host time a call (320 rows a chip, 8x4: one chunk's ratio-2 slab), printed."""
    import time

    nd = mesh.get_num_devices()
    keys = _keys(nd, 320, seed=3)
    x = _dev(keys, mesh)
    e4m3, sc = ttnn.experimental.deepseek_prefill.per_token_cast_to_fp8(x, round_scale_to_power_of_two=False)
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, 320, 192]), ttnn.uint8, ttnn.ROW_MAJOR_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG
    )
    ttnn.bringup.pack_fp8_rows(e4m3, sc, output_tensor=out, max_grid=_grid(mesh))
    ttnn.synchronize_device(mesh)
    n = 50
    t0 = time.perf_counter()
    for _ in range(n):
        ttnn.bringup.pack_fp8_rows(e4m3, sc, output_tensor=out, max_grid=_grid(mesh))
    t1 = time.perf_counter()
    ttnn.synchronize_device(mesh)
    print(f"[pack_fp8_rows host] {(t1 - t0) / n * 1e6:.1f} us a call (prepared, {nd} chips, 320 rows a chip)")
    for t in (x, e4m3, sc, out):
        ttnn.deallocate(t)


def test_pack_fp8_rows_refusals(mesh, expect_error):
    nd = mesh.get_num_devices()
    keys = _keys(nd, 32, seed=1)
    x = _dev(keys, mesh)
    e4m3, sc = ttnn.experimental.deepseek_prefill.per_token_cast_to_fp8(x, round_scale_to_power_of_two=False)
    with expect_error(AssertionError, "e4m3 must be FP8_E4M3"):
        ttnn.bringup.pack_fp8_rows(sc, sc)
    bad = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, 32, 128]), ttnn.uint8, ttnn.ROW_MAJOR_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG
    )
    with expect_error(AssertionError, "output_tensor must be uint8"):
        ttnn.bringup.pack_fp8_rows(e4m3, sc, output_tensor=bad)
    for t in (x, e4m3, sc, bad):
        ttnn.deallocate(t)
