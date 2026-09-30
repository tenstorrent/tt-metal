# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""ttnn.bringup.ring_mla at fp32_dest_acc_en=True: latent-V ring attention on the streaming path with fp32 DEST.

The source op runs latent V (ring_mla: V = the first head_dim_v columns of the one KV tensor) and kv_actual_isl only
on its streaming compute path, which it takes only at bf16 DEST; fp32 DEST is refused ("Latent-V ring attention is
implemented only for streaming compute"). At bf16 DEST the QK^T scores accumulate over the 576-wide head in 16-bit
DEST, which on sharp softmaxes (scores ~ 90) moves single rows by several percent. The fork sends latent V at fp32
DEST to the streaming path (fp32 accumulation, the path's bf16 intermediate CBs as sparse_sdpa at fp32 DEST).
Covered here, on a 4x2 mesh with FABRIC_2D, the ring over axis 0 (Xing4.0's chunked MLA: 16 heads, K 576, V 512,
a block-cyclic latent cache with period = the chunk, q32 / k256):
- accuracy vs a float32 torch reference on the same bf16 inputs, chunk 0 and a chunk after a prefix, plain and sharp
  scores: rel L2 and worst row, each tighter than the source op at bf16 DEST;
- option off (bf16 DEST): bit-identical to ttnn.transformer.ring_mla;
- the source op still refuses fp32 DEST (the combination the fork turns on).
"""

import os

import pytest
import torch

import ttnn

H, K, V = 16, 576, 512
CHUNK, MAX_SEQ = 2048, 4096
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


def _block_cyclic_rows(sp):
    """Natural position of each row of the SP-contiguous split of the block-cyclic cache (row r's shard first)."""
    cl, ls = CHUNK // sp, MAX_SEQ // sp
    c = torch.arange(sp).repeat_interleave(ls)
    lr = torch.arange(ls).repeat(sp)
    return (lr // cl) * CHUNK + c * cl + lr % cl


def _inputs(start, q_scale, seed):
    g = torch.Generator().manual_seed(seed)
    end = start + CHUNK
    q = (torch.randn(1, H, CHUNK, K, generator=g) * q_scale).to(torch.bfloat16)  # queries at [start, end)
    kv = torch.zeros(MAX_SEQ, K)
    kv[:end] = torch.randn(end, K, generator=g)
    return q, kv.to(torch.bfloat16)


def _golden(q, kv, start):
    """float32 causal attention on the bf16 inputs: [1, H, CHUNK, V]."""
    end = start + CHUNK
    k = kv[:end].float()
    sc = torch.einsum("hsd,td->hst", q[0].float(), k) * SCALE
    sc = sc.masked_fill(torch.arange(end)[None, None] > torch.arange(start, end)[None, :, None], float("-inf"))
    return torch.einsum("hst,tv->hsv", sc.softmax(-1), k[:, :V])[None]


def _run(mesh, op, q, kv, start, fp32):
    sp = mesh.shape[0]
    sems, offset, grid = _ccl(mesh)
    dram = ttnn.DRAM_MEMORY_CONFIG
    shard = ttnn.ShardTensor2dMesh(mesh, mesh_shape=tuple(mesh.shape), dims=(2, None))
    tq = ttnn.from_torch(
        q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=dram, mesh_mapper=shard
    )
    cache = kv[_block_cyclic_rows(sp)].reshape(1, 1, MAX_SEQ, K)
    tkv = ttnn.from_torch(
        cache, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=dram, mesh_mapper=shard
    )
    buf = ttnn.from_torch(
        torch.zeros(1, 1, MAX_SEQ, K),
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
        logical_n=start + CHUNK,
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
        kv_actual_isl=start,
    )
    devs = ttnn.get_device_tensors(out)
    cols = mesh.shape[1]
    res = [torch.cat([ttnn.to_torch(devs[r * cols + c]).float() for r in range(sp)], dim=2) for c in range(cols)]
    for t in (tq, tkv, buf, out, stats):
        ttnn.deallocate(t)
    return res  # one [1, H, CHUNK, V] per mesh column (both hold the same heads here)


def _err(got, want):
    g, w = got.reshape(-1, V), want.reshape(-1, V)
    return ((g - w).norm() / w.norm()).item(), ((g - w).norm(dim=-1) / w.norm(dim=-1)).max().item()


MESH = pytest.mark.parametrize("mesh_device", [(4, 2)], indirect=True)
FABRIC = pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_2D, "l1_small_size": 24576}], indirect=True
)
# q_scale 0.25: score std ~ 0.9 (a spread softmax); 3.0: score std ~ 10, max ~ 45 (sharp, as Xing's attention).
# Limits ~1.4x the fork's measured error (spread: rel 0.018 / row 0.031, both chunks; sharp: rel 0.019 / row 0.157;
# the bf16 P and running state bound it); the source at bf16 DEST measured rel 0.023 / row 0.065 (spread) and
# rel 0.091 / row 0.70 (sharp).
CASES = pytest.mark.parametrize(
    "start, q_scale, rel_max, row_max",
    [(0, 0.25, 0.025, 0.045), (2048, 0.25, 0.025, 0.045), (0, 3.0, 0.028, 0.22), (2048, 3.0, 0.028, 0.22)],
    ids=["chunk0", "prefix2048", "chunk0-sharp", "prefix2048-sharp"],
)


@MESH
@FABRIC
@CASES
def test_ring_mla_fp32_dest_accuracy(mesh_device, start, q_scale, rel_max, row_max):
    q, kv = _inputs(start, q_scale, seed=start + int(q_scale * 10))
    want = _golden(q, kv, start)
    fork = _run(mesh_device, ttnn.bringup.ring_mla, q, kv, start, fp32=True)
    fork = [f * float(os.environ.get("RING_MLA_TEST_CORRUPT", "1")) for f in fork]  # hand check: 1.02 must fail
    src = _run(mesh_device, ttnn.transformer.ring_mla, q, kv, start, fp32=False)[0]
    assert torch.equal(fork[0], fork[1]), "the two mesh columns disagree"
    rf, wf = _err(fork[0], want)
    rs, ws = _err(src, want)
    print(f"fp32 dest (fork): rel {rf:.5f} worst row {wf:.5f}; bf16 dest (source): rel {rs:.5f} worst row {ws:.5f}")
    assert rf <= rel_max and wf <= row_max, f"fork fp32 dest rel {rf:.5f} / worst row {wf:.5f}"
    assert rf < rs and wf < ws, "fp32 dest is not more accurate than the source's bf16 dest"


@MESH
@FABRIC
@pytest.mark.parametrize("start", [0, 2048], ids=["chunk0", "prefix2048"])
def test_ring_mla_bf16_dest_matches_source(mesh_device, start):
    q, kv = _inputs(start, 3.0, seed=7 + start)
    fork = _run(mesh_device, ttnn.bringup.ring_mla, q, kv, start, fp32=False)[0]
    src = _run(mesh_device, ttnn.transformer.ring_mla, q, kv, start, fp32=False)[0]
    assert torch.equal(fork, src), f"bf16 dest differs from the source op, max diff {(fork - src).abs().max().item()}"


@MESH
@FABRIC
def test_source_refuses_fp32_dest(mesh_device, expect_error):
    q, kv = _inputs(0, 0.25, seed=3)
    with expect_error(Exception, "Latent-V ring attention|kv_actual_isl requires"):
        _run(mesh_device, ttnn.transformer.ring_mla, q, kv, 0, fp32=True)
