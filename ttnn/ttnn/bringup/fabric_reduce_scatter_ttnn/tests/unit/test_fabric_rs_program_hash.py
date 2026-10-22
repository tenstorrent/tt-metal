# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""fabric_reduce_scatter(program_hash=True) (deepseek_v41_flash_d_p P.19): the cached descriptors carry a
custom_program_hash (generic_op skips hashing the kernel sources), the plan lookup is one dict hit.

Checks (2x2 / the box's mesh, FABRIC_2D), for bf16 / fp32 exact, rows / columns, both mesh axes:
  * bit-identical to the same call without program_hash (and to cache_descriptors alone);
  * fresh buffers (other addresses, another input) on the hashed path give the right sum: the program reused across
    addresses has its runtime arguments patched (2 x the input -> 2 x the output, exactly);
  * the hashed descriptors are separate cache entries from the default path's (each carries a custom hash);
  * deterministic; program_hash without cache_descriptors is refused."""

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
def test_program_hash(mesh_device, device_params, cluster_axis, dim, exact):
    rows, cols = tuple(mesh_device.shape)
    g = (rows, cols)[cluster_axis]
    if g < 2:
        pytest.skip("one chip on this axis")
    dtype = ttnn.float32 if exact else ttnn.bfloat16
    gen = torch.Generator().manual_seed(7 + cluster_axis + 2 * dim + 4 * int(exact))
    parts = torch.randn(rows, cols, ROWS, COLS, generator=gen)
    if not exact:
        parts = parts.bfloat16().float()
    x = _upload(mesh_device, parts, dtype)
    kw = dict(cluster_axis=cluster_axis, num_links=2, dim=dim, exact_fp32_add=exact)
    plain = _per_chip(ttnn.bringup.fabric_reduce_scatter(x, **kw))
    cached = _per_chip(ttnn.bringup.fabric_reduce_scatter(x, cache_descriptors=True, **kw))
    n0 = len(_mod()._DESC_CACHE)
    outs = [_per_chip(ttnn.bringup.fabric_reduce_scatter(x, cache_descriptors=True, program_hash=True, **kw))]
    outs.append(_per_chip(ttnn.bringup.fabric_reduce_scatter(x, cache_descriptors=True, program_hash=True, **kw)))
    new = [k for k in list(_mod()._DESC_CACHE)[n0:] if "hashed" in k]
    assert new, "program_hash: no descriptor of its own in the cache"
    for k in new:
        desc = _mod()._DESC_CACHE[k][1]
        assert all(p.custom_program_hash is not None for _, p in desc), "a program without a custom hash"
    for d in range(rows * cols):
        assert torch.equal(outs[0][d], plain[d]), f"dev {d}: program_hash differs from the default path"
        assert torch.equal(outs[0][d], cached[d]), f"dev {d}: program_hash differs from cache_descriptors"
        assert torch.equal(outs[0][d], outs[1][d]), f"dev {d}: not deterministic"
    # another input in fresh buffers on the hashed path: the cached program with patched addresses
    keep = [
        ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, 64, 64]), dtype, ttnn.TILE_LAYOUT, mesh_device)
        for _ in range(3)
    ]
    x2 = _upload(mesh_device, 2 * parts, dtype)
    got2 = _per_chip(ttnn.bringup.fabric_reduce_scatter(x2, cache_descriptors=True, program_hash=True, **kw))
    for d in range(rows * cols):
        assert torch.equal(got2[d], 2 * outs[0][d]), f"dev {d}: fresh buffers on the hashed path give another sum"
    print(f"axis {cluster_axis} dim {dim} exact {exact}: bit-identical, {len(new)} hashed descriptors")
    for t in keep:
        ttnn.deallocate(t)


@MESH_PARAMS
def test_program_hash_needs_cache(mesh_device, device_params, expect_error):
    rows, cols = tuple(mesh_device.shape)
    x = _upload(mesh_device, torch.zeros(rows, cols, 64, 64), ttnn.bfloat16)
    with expect_error(AssertionError, "program_hash needs cache_descriptors"):
        ttnn.bringup.fabric_reduce_scatter(x, cluster_axis=1, num_links=1, program_hash=True)
