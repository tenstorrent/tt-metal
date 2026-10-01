# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""ttnn.bringup.ring_mla with kv_actual_isl=0 on a cache exactly one chunk long (Q.seq == K.seq per device).

A model that always passes kv_actual_isl=start (chunked prefill over a latent cache) runs its first and only chunk
when the sequence is one chunk long: the cache then has the chunk's length, the input is not chunk-shaped, and the
op used to TT_FATAL ("KV-pad rotation ... requires chunked-prefill input"). kv_actual_isl=0 has no prefix to rotate
past, so the fork drops it there and runs the full-prefill causal path, as its metadata path already does
(kv_pad_from_metadata needs is_chunked). Covered on a 4x2 mesh, FABRIC_2D, ring over axis 0 (Xing4.0's MLA: 16
heads, K 576, V 512, q32 / k256), fp32 and bf16 DEST:
- accuracy vs a float32 torch causal reference;
- bit-identical to the same call without kv_actual_isl (the path it now takes);
- the source op still refuses it (only the previously refused input changes).
"""

import pytest
import torch

import ttnn

H, K, V = 16, 576, 512
CHUNK = 2048
SCALE = 0.14468
_CCL = {}


def _ckc(fp32):
    return ttnn.init_device_compute_kernel_config(
        ttnn.Arch.BLACKHOLE,
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=fp32,
        packer_l1_acc=False,
    )


def _ccl(mesh):
    if id(mesh) not in _CCL:
        g = mesh.compute_with_storage_grid_size()
        cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(g.x - 1, g.y - 1))})
        _CCL[id(mesh)] = (
            [ttnn.create_global_semaphore(mesh, cores, 0) for _ in range(2)],
            (g.x - 1, 0),
            (g.x - 1, g.y),
        )
    return _CCL[id(mesh)]


def _inputs(q_scale, seed):
    g = torch.Generator().manual_seed(seed)
    q = (torch.randn(1, H, CHUNK, K, generator=g) * q_scale).to(torch.bfloat16)
    kv = torch.randn(CHUNK, K, generator=g).to(torch.bfloat16)
    return q, kv


def _golden(q, kv):
    k = kv.float()
    sc = torch.einsum("hsd,td->hst", q[0].float(), k) * SCALE
    sc = sc.masked_fill(torch.ones(CHUNK, CHUNK, dtype=torch.bool).triu(1)[None], float("-inf"))
    return torch.einsum("hst,tv->hsv", sc.softmax(-1), k[:, :V])[None]


def _run(mesh, op, q, kv, fp32, **extra):
    """Cache = the chunk (block-cyclic with one period = the SP-contiguous split of the natural order)."""
    sp, cols = mesh.shape[0], mesh.shape[1]
    sems, offset, grid = _ccl(mesh)
    dram = ttnn.DRAM_MEMORY_CONFIG
    shard = ttnn.ShardTensor2dMesh(mesh, mesh_shape=tuple(mesh.shape), dims=(2, None))
    tq = ttnn.from_torch(
        q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=dram, mesh_mapper=shard
    )
    tkv = ttnn.from_torch(
        kv.reshape(1, 1, CHUNK, K),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=dram,
        mesh_mapper=shard,
    )
    buf = ttnn.from_torch(
        torch.zeros(1, 1, CHUNK, K),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=dram,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    pc = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=grid, q_chunk_size=32, k_chunk_size=256, exp_approx_mode=False
    )
    out, stats = op(
        tq,
        tkv,
        persistent_output_buffer_kv=buf,
        head_dim_v=V,
        logical_n=CHUNK,
        program_config=pc,
        scale=SCALE,
        compute_kernel_config=_ckc(fp32),
        dim=2,
        multi_device_global_semaphore=sems,
        num_links=2,
        cluster_axis=0,
        mesh_device=mesh,
        topology=ttnn.Topology.Linear,
        ccl_core_grid_offset=offset,
        use_column_major_ccl=True,
        is_balanced=False,
        kv_cache_batch_idx=0,
        **extra,
    )
    devs = ttnn.get_device_tensors(out)
    res = [torch.cat([ttnn.to_torch(devs[r * cols + c]).float() for r in range(sp)], dim=2) for c in range(cols)]
    for t in (tq, tkv, buf, out, stats):
        ttnn.deallocate(t)
    return res


def _err(got, want):
    g, w = got.reshape(-1, V), want.reshape(-1, V)
    return ((g - w).norm() / w.norm()).item(), ((g - w).norm(dim=-1) / w.norm(dim=-1)).max().item()


MESH = pytest.mark.parametrize("mesh_device", [(4, 2)], indirect=True)
FABRIC = pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_2D, "l1_small_size": 24576}], indirect=True
)


@MESH
@FABRIC
@pytest.mark.parametrize("fp32", [True, False], ids=["fp32dest", "bf16dest"])
def test_ring_mla_single_chunk_isl0(mesh_device, fp32):
    q, kv = _inputs(0.25, seed=11)
    got = _run(mesh_device, ttnn.bringup.ring_mla, q, kv, fp32, kv_actual_isl=0)
    ref = _run(mesh_device, ttnn.bringup.ring_mla, q, kv, fp32)
    assert torch.equal(got[0], got[1]), "the two mesh columns disagree"
    assert torch.equal(got[0], ref[0]), "kv_actual_isl=0 differs from the call without it"
    rel, row = _err(got[0], _golden(q, kv))
    print(f"single chunk, kv_actual_isl=0, fp32 dest {fp32}: rel {rel:.5f} worst row {row:.5f}")
    assert rel <= 0.03 and row <= 0.06, f"rel {rel:.5f} / worst row {row:.5f}"


@MESH
@FABRIC
def test_source_refuses_single_chunk_isl0(mesh_device, expect_error):
    q, kv = _inputs(0.25, seed=12)
    with expect_error(Exception, "requires chunked-prefill input"):
        _run(mesh_device, ttnn.transformer.ring_mla, q, kv, False, kv_actual_isl=0)
