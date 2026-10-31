# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""ttnn.bringup.rotary_embedding_indexed chain options (deepseek_v41_flash_d_p P.22, ROPE_CHAINS): input_heads,
output_layout, output_dtype, group_heads and a ROW_MAJOR input fold the layout / dtype moves around the RoPE into the
op. On the box's mesh (FABRIC_2D), DeepSeek-V4.1's three chains, every chip with its own inputs and cos / sin rows; the
fused call must equal the unfused chain of the fork's / stock ops bit for bit on every chip, on a miss and a hit:
  * attention q: nlp_create_q_heads_split -> rope (nope prepended) -> to_layout(ROW_MAJOR)
      == rope(wq_b's [1, 1, S, H 512] output, input_heads=H, output_layout=ROW_MAJOR);
  * attention output: to_layout(TILE) -> inverse rope -> nlp_concat_heads of 8-head groups -> typecast(bfloat8_b)
      == rope(sparse_sdpa's ROW_MAJOR [1, H, S, 512] output, output_dtype=bfloat8_b, group_heads=8);
  * indexer q: nlp_create_q_heads_split -> rope (nope prepended) -> bfp_round(bfloat4_b)
      == rope(wq_b's output, input_heads=H, output_dtype=bfloat4_b).
Plus the plain TILE in / out case (== the fork's normal program), the output_tensor option, the refusals, and the
warm host time a call (printed, not gated)."""

import importlib.util
import time
from pathlib import Path

import pytest
import torch

import ttnn

_spec = importlib.util.spec_from_file_location(
    "bringup_rotary_unit_test_uniform_chain", Path(__file__).resolve().parent / "test_uniform.py"
)
U = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(U)

R = U.R
DRAM = ttnn.DRAM_MEMORY_CONFIG


def _grid():
    return ttnn.CoreCoord(11, 10)  # the Galaxy cap (MAX_WORKER_CORES 110) the model passes


def _tables(mesh, rows, inverse, seed):
    n = mesh.get_num_devices()
    g = torch.Generator().manual_seed(seed)
    ang = torch.rand(n, 1, rows, R // 2, generator=g) * 6.28
    cos = torch.cos(ang).repeat_interleave(2, dim=-1)
    sin = torch.sin(ang).repeat_interleave(2, dim=-1)
    return U._put(mesh, cos, True), U._put(mesh, sin, True), U._put(mesh, U._trans(inverse), False)


def _rope(x, cos, sin, trans, offset, **kw):
    return ttnn.bringup.rotary_embedding_indexed(
        x,
        cos,
        sin,
        trans,
        0,
        0,
        memory_config=DRAM,
        rotary_dim=R,
        rotary_offset=offset,
        uniform=True,
        max_grid=_grid(),
        **kw,
    )


def _equal(mesh, a, b, what):
    for i, (x, y) in enumerate(zip(U._chips(mesh, a), U._chips(mesh, b))):
        assert x.shape == y.shape, f"{what}: chip {i} shape {tuple(x.shape)} vs {tuple(y.shape)}"
        assert torch.equal(x, y), f"{what}: chip {i} differs, max |diff| {(x.float() - y.float()).abs().max().item()}"


def _flat_q(mesh, rows, heads, width, seed):
    n = mesh.get_num_devices()
    g = torch.Generator().manual_seed(seed)
    return U._put(mesh, torch.randn(n, 1, rows, heads * width, generator=g) * 3, True)


CASES_Q = [
    pytest.param(160, 64, 512, id="attn_q"),
    pytest.param(160, 32, 128, id="indexer_q_shape"),
]


@pytest.mark.parametrize("mesh_device", U._mesh_cases(), indirect=True)
@pytest.mark.parametrize("device_params", U.DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("rows, heads, width", CASES_Q)
def test_chain_q_rm(mesh_device, rows, heads, width):
    mesh = mesh_device
    cos, sin, trans = _tables(mesh, rows, False, 7 + heads)
    for rep in range(2):  # a miss, then a hit with other inputs
        qb = _flat_q(mesh, rows, heads, width, 11 + rep)
        qn, qr = ttnn.bringup.nlp_create_q_heads_split(
            qb, num_heads=heads, split_head_dim=width - R, memory_config=DRAM, max_grid=_grid()
        )
        ref = ttnn.to_layout(_rope(qr, cos, sin, trans, 0, concat_prefix=qn), ttnn.ROW_MAJOR_LAYOUT, memory_config=DRAM)
        got = _rope(qb, cos, sin, trans, width - R, input_heads=heads, output_layout=ttnn.ROW_MAJOR_LAYOUT)
        assert got.layout == ttnn.ROW_MAJOR_LAYOUT and list(got.shape) == [1, heads, rows, width], got.shape
        _equal(mesh, got, ref, f"q chain rep {rep}")


@pytest.mark.parametrize("mesh_device", U._mesh_cases(), indirect=True)
@pytest.mark.parametrize("device_params", U.DEVICE_PARAMS, indirect=True)
def test_chain_indexer_q_bfp4(mesh_device):
    mesh = mesh_device
    rows, heads, width = 160, 32, 128
    cos, sin, trans = _tables(mesh, rows, False, 21)
    for rep in range(2):
        qb = _flat_q(mesh, rows, heads, width, 31 + rep)
        qn, qr = ttnn.bringup.nlp_create_q_heads_split(
            qb, num_heads=heads, split_head_dim=width - R, memory_config=DRAM, max_grid=_grid()
        )
        ref = ttnn.bringup.bfp_round(_rope(qr, cos, sin, trans, 0, concat_prefix=qn), ttnn.bfloat4_b)
        got = _rope(qb, cos, sin, trans, width - R, input_heads=heads, output_dtype=ttnn.bfloat4_b)
        assert got.dtype == ttnn.bfloat4_b and list(got.shape) == [1, heads, rows, width], (got.dtype, got.shape)
        _equal(mesh, got, ref, f"indexer q chain rep {rep}")


@pytest.mark.parametrize("mesh_device", U._mesh_cases(), indirect=True)
@pytest.mark.parametrize("device_params", U.DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("group", [8, 1], ids=["grouped8", "plain"])
def test_chain_out_bfp8(mesh_device, group):
    mesh = mesh_device
    rows, heads, width = 160, 64, 512
    n = mesh.get_num_devices()
    cos, sin, trans = _tables(mesh, rows, True, 41)
    for rep in range(2):
        g = torch.Generator().manual_seed(51 + rep)
        o_rm = ttnn.to_layout(
            U._put(mesh, torch.randn(n, heads, rows, width, generator=g), True),
            ttnn.ROW_MAJOR_LAYOUT,
            memory_config=DRAM,
        )
        ot = ttnn.to_layout(o_rm, ttnn.TILE_LAYOUT, memory_config=DRAM)
        oq = _rope(ot, cos, sin, trans, width - R)
        if group > 1:
            og = ttnn.bringup.nlp_concat_heads(
                ttnn.reshape(oq, [heads // group, group, rows, width]), memory_config=DRAM, max_grid=_grid()
            )
            og = ttnn.reshape(og, [1, heads // group, rows, group * width])
        else:
            og = oq
        ref = ttnn.typecast(og, ttnn.bfloat8_b, memory_config=DRAM)
        got = _rope(o_rm, cos, sin, trans, width - R, output_dtype=ttnn.bfloat8_b, group_heads=group)
        assert got.dtype == ttnn.bfloat8_b and list(got.shape) == list(ref.shape), (got.dtype, got.shape, ref.shape)
        _equal(mesh, got, ref, f"out chain rep {rep}")


@pytest.mark.parametrize("mesh_device", U._mesh_cases(), indirect=True)
@pytest.mark.parametrize("device_params", U.DEVICE_PARAMS, indirect=True)
def test_chain_tile_bf16_and_output_tensor(mesh_device):
    """TILE in, TILE bf16 out through the chain program (group_heads 2 selects it) vs the normal program + concat;
    then the output_tensor option on the chain (same bytes, the caller's buffer, a hit)."""
    mesh = mesh_device
    rows, heads, width = 160, 4, 512
    n = mesh.get_num_devices()
    cos, sin, trans = _tables(mesh, rows, False, 61)
    x = U._put(mesh, torch.randn(n, heads, rows, width), True)
    oq = _rope(x, cos, sin, trans, width - R)
    ref = ttnn.reshape(
        ttnn.bringup.nlp_concat_heads(ttnn.reshape(oq, [2, 2, rows, width]), memory_config=DRAM),
        [1, 2, rows, 2 * width],
    )
    got = _rope(x, cos, sin, trans, width - R, group_heads=2)
    _equal(mesh, got, ref, "tile bf16 grouped")
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 2, rows, 2 * width]), ttnn.bfloat16, ttnn.TILE_LAYOUT, mesh, DRAM
    )
    for _ in range(2):
        r = _rope(x, cos, sin, trans, width - R, group_heads=2, output_tensor=out)
        assert r.buffer_address() == out.buffer_address()
        _equal(mesh, out, ref, "output_tensor")


@pytest.mark.parametrize("mesh_device", U._mesh_cases(), indirect=True)
@pytest.mark.parametrize("device_params", U.DEVICE_PARAMS, indirect=True)
def test_chain_refusals(mesh_device, expect_error):
    mesh = mesh_device
    rows = 160
    cos, sin, trans = _tables(mesh, rows, False, 71)
    qb = _flat_q(mesh, rows, 4, 512, 72)
    with expect_error(RuntimeError, "needs kv_actual_global 0, no seq_subshard_axis and no concat_prefix"):
        ttnn.bringup.rotary_embedding_indexed(
            qb, cos, sin, trans, 32, 0, rotary_dim=R, rotary_offset=448, input_heads=4, uniform=True
        )
    with expect_error(RuntimeError, "is not input_heads 3 x a tile-aligned head width"):
        _rope(qb, cos, sin, trans, 448, input_heads=3)
    with expect_error(RuntimeError, "a ROW_MAJOR output is bf16"):
        _rope(qb, cos, sin, trans, 448, input_heads=4, output_layout=ttnn.ROW_MAJOR_LAYOUT, group_heads=2)
    bad = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 4, rows, 512]), ttnn.bfloat16, ttnn.TILE_LAYOUT, mesh, DRAM)
    with expect_error(RuntimeError, "is not the op's output spec"):
        _rope(qb, cos, sin, trans, 448, input_heads=4, output_layout=ttnn.ROW_MAJOR_LAYOUT, output_tensor=bad)


@pytest.mark.parametrize("mesh_device", U._mesh_cases(), indirect=True)
@pytest.mark.parametrize("device_params", U.DEVICE_PARAMS, indirect=True)
def test_chain_host_time(mesh_device):
    """Warm host us a call (dispatch only), printed: the fused q chain vs its three unfused ops."""
    mesh = mesh_device
    rows, heads, width = 160, 64, 512
    cos, sin, trans = _tables(mesh, rows, False, 81)
    qb = _flat_q(mesh, rows, heads, width, 82)

    def unfused():
        qn, qr = ttnn.bringup.nlp_create_q_heads_split(
            qb, num_heads=heads, split_head_dim=width - R, memory_config=DRAM, max_grid=_grid()
        )
        q = _rope(qr, cos, sin, trans, 0, concat_prefix=qn)
        for t in (qn, qr):
            ttnn.deallocate(t)
        r = ttnn.to_layout(q, ttnn.ROW_MAJOR_LAYOUT, memory_config=DRAM)
        ttnn.deallocate(q)
        return r

    def fused():
        return _rope(qb, cos, sin, trans, width - R, input_heads=heads, output_layout=ttnn.ROW_MAJOR_LAYOUT)

    res = {}
    for name, fn in (("unfused", unfused), ("fused", fused)) * 2:
        for _ in range(3):
            ttnn.deallocate(fn())
        ttnn.synchronize_device(mesh)
        n = 30
        t0 = time.perf_counter()
        for _ in range(n):
            ttnn.deallocate(fn())
        t1 = time.perf_counter()
        ttnn.synchronize_device(mesh)
        res[name] = (t1 - t0) / n * 1e6
    print(
        f"\nq chain host us a call on {tuple(mesh.shape)}: unfused (split + rope + to_layout) {res['unfused']:.1f}, "
        f"fused {res['fused']:.1f}"
    )
