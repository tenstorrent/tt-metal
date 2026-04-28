#!/usr/bin/env python3
"""3B: Validate fused TQ SDPA on a mesh device.

Each device has its own copy of the paged cache (ReplicateTensorToMesh) and
independently scatters/reads only its local KV head(s). For T3K with 8 KV
heads / 8 devices, n_local_kv_heads=1 per device.

This test populates the cache via paged_update_cache (per-step write) on
every device, then calls fused_sdpa_decode at cur_pos=41. Compares each
device's output to a per-device CPU reference. Pass = cos > 0.99 on every
device.
"""

import os
import sys

import torch
import ttnn

sys.path.insert(0, "/localdev/mtairum/tt-metal")
from models.tt_transformers.tt.common import PagedAttentionConfig
from turbo_quant.quantizer import TurboQuantMSE
from turbo_quant.ttnn_integration import TTNNTurboQuantCache


def reference_sdpa_masked(q, k, v, scale, cur_pos):
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    mask = torch.full_like(scores, float("-inf"))
    mask[..., : cur_pos + 1] = 0
    scores = scores + mask
    attn = torch.softmax(scores, dim=-1)
    return torch.matmul(attn, v)


def main():
    num_devices = int(os.environ.get("TT_NUM_DEVICES", 8))
    if num_devices > 1:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)

    mesh_shape = ttnn.MeshShape(1, num_devices)
    mesh = ttnn.open_mesh_device(mesh_shape)
    print(f"\nOpened mesh: {mesh.get_num_devices()} devices ({mesh_shape})")

    head_dim = 128
    n_kv_global = 8
    n_q_global = 32
    bits = 3
    scale = head_dim**-0.5
    seq_len = 2048
    # Need valid_k_chunks ≥ K so every worker has at least one chunk (otherwise
    # matmul_reduce in the post-loop hangs on an empty alias_prev_sum CB).
    # For K=14: need valid_k_chunks=14, i.e. cur_pos in [13*128, 14*128-1] = [1664, 1791].
    cur_pos = 1791
    block_size = 32
    max_blocks = 1024

    # Per-device head counts (mirror Llama-3.1-8B sharding on T3K)
    nkh_local = n_kv_global // num_devices
    nqh_local = n_q_global // num_devices
    if nkh_local == 0:
        # If num_devices > n_kv_global, KV heads are replicated across some devices
        # (not the typical Llama T3K config — abort for clarity)
        raise RuntimeError(f"num_devices={num_devices} > n_kv_heads={n_kv_global}, layout undefined")
    print(f"  Per-device: nqh_local={nqh_local}, nkh_local={nkh_local}, head_dim={head_dim}")

    paged_cfg = PagedAttentionConfig(block_size=block_size, max_num_blocks=max_blocks)
    tq = TTNNTurboQuantCache(
        mesh,
        num_layers=1,
        num_kv_heads=nkh_local,
        head_dim=head_dim,
        max_seq_len=seq_len,
        bits=bits,
        memory_efficient=True,
        paged_config=paged_cfg,
        max_batch_size=1,
    )

    # Global K/V/Q (rank-4 with full head dim). Sharded along dim 1 (head dim)
    # so each device gets [1, nkh_local, *, dh] / [1, nqh_local, *, dh].
    torch.manual_seed(42)
    k_global = torch.randn(1, n_kv_global, seq_len, head_dim)
    v_global = torch.randn(1, n_kv_global, seq_len, head_dim)
    q_global = torch.randn(1, n_q_global, 1, head_dim)
    # Zero positions > cur_pos (matches e2e cache state at decode step 41)
    k_global[:, :, cur_pos + 1 :, :] = 0
    v_global[:, :, cur_pos + 1 :, :] = 0

    # Page table: identity mapping, replicated across devices
    blocks_per_seq = (seq_len + block_size - 1) // block_size
    page_table_torch = torch.arange(blocks_per_seq, dtype=torch.int32).reshape(1, -1)
    page_table_dev = ttnn.from_torch(
        page_table_torch,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )

    # Sharding helpers: KV/Q head dim is dim 1 of [B, H, S, D].
    head_shard = ttnn.ShardTensor2dMesh(mesh, dims=(None, 1), mesh_shape=(1, num_devices)) if num_devices > 1 else None
    replicate = ttnn.ReplicateTensorToMesh(mesh) if num_devices > 1 else None

    # Populate cache one step at a time, mirroring eval_e2e's per-step write.
    for step in range(cur_pos + 1):
        k_step = k_global[:, :, step : step + 1, :]  # [1, n_kv_global, 1, dh]
        v_step = v_global[:, :, step : step + 1, :]
        k_dev = ttnn.from_torch(
            k_step,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            mesh_mapper=head_shard,
        )
        v_dev = ttnn.from_torch(
            v_step,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            mesh_mapper=head_shard,
        )

        cur_pos_dev = ttnn.from_torch(
            torch.tensor([step], dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            mesh_mapper=replicate,
        )

        tq.update_cache(k_dev, v_dev, layer_idx=0, current_pos=cur_pos_dev, page_table=page_table_dev)
        ttnn.deallocate(k_dev)
        ttnn.deallocate(v_dev)
        ttnn.deallocate(cur_pos_dev)

    # Run fused SDPA at cur_pos=41
    q_dev = ttnn.from_torch(
        q_global,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        mesh_mapper=head_shard,
    )
    cur_pos_dev = ttnn.from_torch(
        torch.tensor([cur_pos], dtype=torch.int32),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh,
        mesh_mapper=replicate,
    )

    K = int(os.environ.get("TQ_NUM_CORES_PER_HEAD", "1"))
    print(f"  Running with num_cores_per_head={K}")
    out = tq.fused_sdpa_decode(
        q_dev,
        layer_idx=0,
        current_pos=cur_pos_dev,
        scale=scale,
        page_table=page_table_dev,
        num_cores_per_head=K,
    )

    # Output shape per device: [1, nqh_local, 1, dh]. Concat along head dim → [1, n_q_global, 1, dh].
    if num_devices > 1:
        out_cpu = ttnn.to_torch(
            out,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh, dims=(0, 1), mesh_shape=(1, num_devices)),
        ).float()
    else:
        out_cpu = ttnn.to_torch(out).float()
    # out_cpu shape: [1, n_q_global, 1, dh]

    quantizer = TurboQuantMSE(head_dim=head_dim, bits=bits, seed=42, dtype=torch.float32)
    # CPU reference: quantize the global K/V and run masked SDPA over all heads.
    k_idx, k_norms = quantizer.quantize(k_global)
    v_idx, v_norms = quantizer.quantize(v_global)
    k_dq = quantizer.codebook.dequantize(k_idx.long()) * k_norms
    v_dq = quantizer.codebook.dequantize(v_idx.long()) * v_norms
    heads_per_kv = n_q_global // n_kv_global
    k_exp = k_dq.repeat_interleave(heads_per_kv, dim=1)
    v_exp = v_dq.repeat_interleave(heads_per_kv, dim=1)
    ref = reference_sdpa_masked(q_global, k_exp, v_exp, scale, cur_pos)
    # ref shape: [1, n_q_global, 1, dh]

    print(f"\nPer-device cosine vs masked reference (cur_pos={cur_pos}):")
    all_pass = True
    for d in range(num_devices):
        h0 = d * nqh_local
        h1 = h0 + nqh_local
        out_d = out_cpu[0, h0:h1]  # [nqh_local, 1, dh]
        ref_d = ref[0, h0:h1]
        cos = torch.nn.functional.cosine_similarity(
            out_d.flatten().unsqueeze(0),
            ref_d.flatten().unsqueeze(0),
        ).item()
        out_max = out_d.abs().max().item()
        ref_max = ref_d.abs().max().item()
        ratio = out_max / max(ref_max, 1e-9)
        status = "PASS" if cos > 0.99 else "FAIL"
        print(
            f"  device {d} (heads {h0}-{h1-1}): cos={cos:.4f}  TQ_max={out_max:.4f}  "
            f"ref_max={ref_max:.4f}  ratio={ratio:.2f}  [{status}]"
        )
        all_pass = all_pass and (cos > 0.99)

    ttnn.deallocate(out)
    ttnn.deallocate(q_dev)
    ttnn.deallocate(cur_pos_dev)
    ttnn.deallocate(page_table_dev)
    tq.deallocate()
    ttnn.close_mesh_device(mesh)

    print(f"\nResult: {'PASS' if all_pass else 'FAIL'} ({num_devices} devices)")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
