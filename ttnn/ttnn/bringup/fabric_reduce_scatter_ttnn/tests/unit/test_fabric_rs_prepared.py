# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""fabric_reduce_scatter(prepared=True) (deepseek_v41_flash_d_p P.1): each cached descriptor is prepared once
(ttnn.bringup.prepare_program) and dispatched with ttnn.bringup.run_prepared instead of ttnn.generic_op.

Checks (2x2 / the box's mesh via BRINGUP_TEST_MESH, FABRIC_2D), for bf16 / fp32 exact, rows / columns, both mesh axes:
  * bit-identical to the same call through generic_op (program_hash alone);
  * the prepared entries are cache entries of their own, each holding a PreparedProgram;
  * another input in fresh buffers (a second descriptor of the same structure, so the shared cached program has its
    address slots patched) gives the right sum, and the first buffers again after it (patched back) too;
  * deterministic; prepared without program_hash is refused."""

import sys

import pytest
import torch

import ttnn
from ttnn.bringup._testing.mesh import MESH_PARAMS

ROWS, COLS = 256, 512


def _mod():
    return sys.modules["ttnn.bringup.fabric_reduce_scatter_ttnn.fabric_reduce_scatter"]


def _upload(mesh_device, parts, dtype):
    rows, cols = parts.shape[0], parts.shape[1]
    t = ttnn.from_torch(
        parts,
        device=mesh_device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(rows, cols), dims=(0, 1)),
    )
    return ttnn.reshape(t, (1, 1, parts.shape[2], parts.shape[3]))


def _per_chip(t):
    return [ttnn.to_torch(d).float() for d in ttnn.get_device_tensors(t)]


@MESH_PARAMS
@pytest.mark.parametrize("exact", [False, True], ids=["bf16", "fp32_exact"])
@pytest.mark.parametrize("dim", [2, 3], ids=["rows", "cols"])
@pytest.mark.parametrize("cluster_axis", [0, 1])
def test_prepared(mesh_device, device_params, cluster_axis, dim, exact):
    rows, cols = tuple(mesh_device.shape)
    g = (rows, cols)[cluster_axis]
    if g < 2:
        pytest.skip("one chip on this axis")
    dtype = ttnn.float32 if exact else ttnn.bfloat16
    gen = torch.Generator().manual_seed(11 + cluster_axis + 2 * dim + 4 * int(exact))
    parts = torch.randn(rows, cols, ROWS, COLS * g // 2, generator=gen)
    if not exact:
        parts = parts.bfloat16().float()
    x = _upload(mesh_device, parts, dtype)
    kw = dict(cluster_axis=cluster_axis, num_links=2, dim=dim, exact_fp32_add=exact, cache_descriptors=True)
    out_g = ttnn.bringup.fabric_reduce_scatter(x, program_hash=True, **kw)
    ref = _per_chip(out_g)
    n0 = len(_mod()._DESC_CACHE)
    out_p = ttnn.bringup.fabric_reduce_scatter(x, program_hash=True, prepared=True, **kw)
    first = _per_chip(out_p)
    new = [k for k in list(_mod()._DESC_CACHE)[n0:] if "prepared" in k]
    assert new, "prepared: no cache entry of its own"
    for k in new:
        hit = _mod()._DESC_CACHE[k]
        assert len(hit) == 3 and hit[2].n_io == 3, "a prepared entry without its PreparedProgram"
    # the same output buffer again (the descriptor's own buffers: nothing to patch)
    again = _per_chip(ttnn.bringup.fabric_reduce_scatter(x, program_hash=True, prepared=True, output=out_p, **kw))
    for d in range(rows * cols):
        assert torch.equal(first[d], ref[d]), f"dev {d}: prepared differs from generic_op"
        assert torch.equal(first[d], again[d]), f"dev {d}: not deterministic"
    # another input in fresh buffers: a second descriptor sharing the cached program (its address slots patched)
    keep = [
        ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, 64, 64]), dtype, ttnn.TILE_LAYOUT, mesh_device)
        for _ in range(3)
    ]
    x2 = _upload(mesh_device, 2 * parts, dtype)
    got2 = _per_chip(ttnn.bringup.fabric_reduce_scatter(x2, program_hash=True, prepared=True, **kw))
    # and the first buffers once more (patched back)
    back = _per_chip(ttnn.bringup.fabric_reduce_scatter(x, program_hash=True, prepared=True, output=out_p, **kw))
    for d in range(rows * cols):
        assert torch.equal(got2[d], 2 * ref[d]), f"dev {d}: fresh buffers on the prepared path give another sum"
        assert torch.equal(back[d], ref[d]), f"dev {d}: the first buffers after the second give another sum"
    print(f"axis {cluster_axis} dim {dim} exact {exact}: bit-identical to generic_op, {len(new)} prepared entries")
    for t in keep:
        ttnn.deallocate(t)


@MESH_PARAMS
def test_prepared_needs_program_hash(mesh_device, device_params, expect_error):
    rows, cols = tuple(mesh_device.shape)
    x = _upload(mesh_device, torch.zeros(rows, cols, 64, 64 * cols), ttnn.bfloat16)
    with expect_error(AssertionError, "prepared needs program_hash"):
        ttnn.bringup.fabric_reduce_scatter(x, cluster_axis=1, num_links=1, cache_descriptors=True, prepared=True)
