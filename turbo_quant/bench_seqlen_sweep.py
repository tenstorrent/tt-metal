#!/usr/bin/env python3
"""Seqlen sweep: TQ Full Dequant fused SDPA vs baseline BFP8 paged SDPA.

Measures the per-decode-step latency of just the SDPA op (the part that
scales with seqlen) at power-of-2 cur_pos values, on either a single
device (TT_NUM_DEVICES=1, default) or a MeshDevice (TT_NUM_DEVICES=8 = T3K).

Reports:
  - Fused TQ SDPA ms/call
  - Baseline BFP8 paged SDPA ms/call
  - Speedup ratio
  - Total KV cache memory (32 layers, all devices) for each format

KV cache size assumes Llama-3.1-8B (32 layers, n_kv_heads=8, head_dim=128).

Usage:
    TT_NUM_DEVICES=1 python turbo_quant/bench_seqlen_sweep.py        # N150
    TT_NUM_DEVICES=8 python turbo_quant/bench_seqlen_sweep.py        # T3K
"""

import gc
import os
import sys
import time

import torch
import ttnn

sys.path.insert(0, "/localdev/mtairum/tt-metal")
from models.tt_transformers.tt.common import PagedAttentionConfig
from turbo_quant.quantizer import TurboQuantMSE
from turbo_quant.ttnn_integration import TTNNTurboQuantCache


N_LAYERS = 32
N_KV_GLOBAL = 8
N_Q_GLOBAL = 32
HEAD_DIM = 128
BLOCK_SIZE = 32
BITS = 3
SCALE = HEAD_DIM**-0.5


def kv_cache_bytes(seq_len, n_kv=N_KV_GLOBAL, head_dim=HEAD_DIM, bytes_per_elem=1.0, n_layers=N_LAYERS):
    """Total KV cache bytes across all layers (K + V, all heads)."""
    seq_padded = ((seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE
    return 2 * n_layers * n_kv * seq_padded * head_dim * bytes_per_elem


def fmt_bytes(n):
    if n >= 1 << 30:
        return f"{n / (1 << 30):.2f} GB"
    if n >= 1 << 20:
        return f"{n / (1 << 20):.1f} MB"
    return f"{n / 1024:.1f} KB"


def shard_helpers(mesh, num_devices):
    """Return (head_shard, replicate) mesh mappers, or (None, None) for single device."""
    if num_devices > 1:
        head_shard = ttnn.ShardTensor2dMesh(mesh, dims=(None, 1), mesh_shape=(1, num_devices))
        replicate = ttnn.ReplicateTensorToMesh(mesh)
    else:
        head_shard = None
        replicate = None
    return head_shard, replicate


def bench_fused_tq(mesh, num_devices, seq_len, warmup=2, iters=5):
    """Per-call latency of fused TQ SDPA decode at cur_pos = seq_len - 1.

    Returns (elapsed_ms, cache_bytes_per_device_per_layer).
    """
    nkh_local = N_KV_GLOBAL // max(num_devices, 1)
    head_shard, replicate = shard_helpers(mesh, num_devices)

    seq_padded = ((seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE
    blocks_needed = seq_padded // BLOCK_SIZE
    paged_cfg = PagedAttentionConfig(block_size=BLOCK_SIZE, max_num_blocks=blocks_needed)

    tq = TTNNTurboQuantCache(
        mesh,
        num_layers=1,
        num_kv_heads=nkh_local,
        head_dim=HEAD_DIM,
        max_seq_len=seq_padded,
        bits=BITS,
        memory_efficient=True,
        paged_config=paged_cfg,
        max_batch_size=1,
    )

    # Compute on-device cache size from the TTNN tensors' padded shapes.
    DTYPE_BYTES_PER_ELEM = {
        ttnn.bfloat4_b: 0.5,
        ttnn.bfloat8_b: 1.0,
        ttnn.bfloat16: 2.0,
        ttnn.float32: 4.0,
        ttnn.uint32: 4.0,
        ttnn.int32: 4.0,
    }

    def _tensor_bytes_per_device(t):
        n = 1
        for d in t.padded_shape:
            n *= int(d)
        return int(n * DTYPE_BYTES_PER_ELEM.get(t.dtype, 2.0))

    idx_bytes = _tensor_bytes_per_device(tq.k_indices_dev[0]) + _tensor_bytes_per_device(tq.v_indices_dev[0])
    norm_bytes = _tensor_bytes_per_device(tq.k_norms_dev[0]) + _tensor_bytes_per_device(tq.v_norms_dev[0])
    cache_bytes_per_layer_per_dev = idx_bytes + norm_bytes

    torch.manual_seed(42)
    k_global = torch.randn(1, N_KV_GLOBAL, seq_padded, HEAD_DIM)
    v_global = torch.randn(1, N_KV_GLOBAL, seq_padded, HEAD_DIM)
    q_global = torch.randn(1, N_Q_GLOBAL, 1, HEAD_DIM)

    # Build the page table covering all blocks (identity).
    page_table_torch = torch.arange(blocks_needed, dtype=torch.int32).reshape(1, -1)
    page_table_dev = ttnn.from_torch(
        page_table_torch,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh,
        mesh_mapper=replicate,
    )

    # Populate the entire cache in one shot at position seq_padded-1 by writing a
    # full-seq stack (we write per-position but in a tight loop). For benchmarking
    # latency at cur_pos=seq_len-1 the cache content does not matter; just need it
    # populated so the kernel reads real data. Single bulk write via a placeholder.
    # Use a per-step write loop only at small seqs; otherwise just zero-init is OK
    # (kernel still iterates the right number of chunks based on cur_pos).
    # NOTE: cache is zero-initialized by TTNNTurboQuantCache, so we can skip writes
    # entirely for benchmarking — the kernel still iterates the same chunks, only
    # the data values differ.

    cur_pos = seq_len - 1
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
    for _ in range(warmup):
        out = tq.fused_sdpa_decode(
            q_dev,
            layer_idx=0,
            current_pos=cur_pos_dev,
            scale=SCALE,
            page_table=page_table_dev,
            num_cores_per_head=K,
        )
        ttnn.deallocate(out)

    ttnn.synchronize_device(mesh)
    t0 = time.perf_counter()
    for _ in range(iters):
        out = tq.fused_sdpa_decode(
            q_dev,
            layer_idx=0,
            current_pos=cur_pos_dev,
            scale=SCALE,
            page_table=page_table_dev,
            num_cores_per_head=K,
        )
        ttnn.deallocate(out)
    ttnn.synchronize_device(mesh)
    elapsed_ms = (time.perf_counter() - t0) / iters * 1000

    ttnn.deallocate(q_dev)
    ttnn.deallocate(cur_pos_dev)
    ttnn.deallocate(page_table_dev)
    tq.deallocate()
    return elapsed_ms, cache_bytes_per_layer_per_dev, idx_bytes, norm_bytes


def _bench_paged_sdpa(mesh, num_devices, seq_len, dtype, warmup, iters):
    """Per-call latency of paged standard SDPA decode at cur_pos = seq_len - 1.

    `dtype` selects the K/V storage format on device (ttnn.bfloat8_b or
    ttnn.bfloat4_b). The math is the same — we feed BF16 random K/V, ttnn
    handles the BFP packing on transfer.

    Returns (elapsed_ms, cache_bytes_per_device_per_layer).
    """
    nkh_local = N_KV_GLOBAL // max(num_devices, 1)
    head_shard, replicate = shard_helpers(mesh, num_devices)

    seq_padded = ((seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE
    blocks_needed = seq_padded // BLOCK_SIZE
    max_blocks = blocks_needed

    cache_shape = (max_blocks, nkh_local, BLOCK_SIZE, HEAD_DIM)
    torch.manual_seed(123)
    k_data = torch.randn(cache_shape, dtype=torch.bfloat16)
    v_data = torch.randn(cache_shape, dtype=torch.bfloat16)
    keys = ttnn.from_torch(
        k_data,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=replicate,
    )
    values = ttnn.from_torch(
        v_data,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=replicate,
    )

    DTYPE_BYTES_PER_ELEM = {
        ttnn.bfloat4_b: 0.5,
        ttnn.bfloat8_b: 1.0,
        ttnn.bfloat16: 2.0,
    }

    def _tensor_bytes_per_device(t):
        n = 1
        for d in t.padded_shape:
            n *= int(d)
        return int(n * DTYPE_BYTES_PER_ELEM.get(t.dtype, 2.0))

    cache_bytes_per_layer_per_dev = _tensor_bytes_per_device(keys) + _tensor_bytes_per_device(values)

    torch.manual_seed(42)
    # Standard paged SDPA decode expects Q in [1, B, NQH, D] layout (head dim is dim 2).
    q_global = torch.randn(1, 1, N_Q_GLOBAL, HEAD_DIM)
    cur_pos = seq_len - 1

    page_table_torch = torch.arange(blocks_needed, dtype=torch.int32).reshape(1, -1)
    page_table_dev = ttnn.from_torch(
        page_table_torch,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh,
        mesh_mapper=replicate,
    )
    q_head_shard = (
        ttnn.ShardTensor2dMesh(mesh, dims=(None, 2), mesh_shape=(1, num_devices)) if num_devices > 1 else None
    )
    q_dev = ttnn.from_torch(
        q_global,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        mesh_mapper=q_head_shard,
    )
    cur_pos_dev = ttnn.from_torch(
        torch.tensor([cur_pos], dtype=torch.int32),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh,
        mesh_mapper=replicate,
    )

    for _ in range(warmup):
        out = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q_dev,
            keys,
            values,
            page_table_tensor=page_table_dev,
            cur_pos_tensor=cur_pos_dev,
            scale=SCALE,
        )
        ttnn.deallocate(out)

    ttnn.synchronize_device(mesh)
    t0 = time.perf_counter()
    for _ in range(iters):
        out = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q_dev,
            keys,
            values,
            page_table_tensor=page_table_dev,
            cur_pos_tensor=cur_pos_dev,
            scale=SCALE,
        )
        ttnn.deallocate(out)
    ttnn.synchronize_device(mesh)
    elapsed_ms = (time.perf_counter() - t0) / iters * 1000

    ttnn.deallocate(q_dev)
    ttnn.deallocate(cur_pos_dev)
    ttnn.deallocate(page_table_dev)
    ttnn.deallocate(keys)
    ttnn.deallocate(values)
    return elapsed_ms, cache_bytes_per_layer_per_dev


def bench_baseline_bfp8(mesh, num_devices, seq_len, warmup=2, iters=5):
    """Per-call latency of paged BFP8 SDPA decode."""
    return _bench_paged_sdpa(mesh, num_devices, seq_len, ttnn.bfloat8_b, warmup, iters)


def bench_rescaled_bfp4(mesh, num_devices, seq_len, warmup=2, iters=5):
    """Per-call latency of paged BFP4 SDPA decode (Track B storage format).

    With --tq-rescaled-bfp4 the layer_past is allocated as BFP4 and the SDPA
    decode kernel reads it natively. The TQ rotation is amortised at write
    time (update_cache), so per-call SDPA latency is the same as plain BFP4
    storage. This row tells us where Track B's per-call latency lands.
    """
    return _bench_paged_sdpa(mesh, num_devices, seq_len, ttnn.bfloat4_b, warmup, iters)


def main():
    num_devices = int(os.environ.get("TT_NUM_DEVICES", 1))
    if num_devices > 1:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)

    mesh_shape = ttnn.MeshShape(1, num_devices)
    mesh = ttnn.open_mesh_device(mesh_shape)
    print(
        f"\nDevice: {num_devices}-device mesh ({mesh_shape}) "
        f"[{'N150' if num_devices == 1 else 'T3K' if num_devices == 8 else f'{num_devices}-chip'}]"
    )
    print(f"Model dims: n_layers={N_LAYERS}, n_kv_heads={N_KV_GLOBAL}, n_q_heads={N_Q_GLOBAL}, head_dim={HEAD_DIM}")

    seq_lens = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]

    print(f"\n{'='*148}")
    print(
        f"{'seq_len':>8} | {'TQ FD ms':>10} | {'BFP4 ms':>10} | {'BFP8 ms':>10} | "
        f"{'TQ KV total':>14} | {'BFP4 total':>14} | {'BFP8 total':>14} | "
        f"{'TQ/BFP8':>8} | {'BFP4/BFP8':>10}"
    )
    print(
        f"{'':>8} | {'(per call)':>10} | {'(per call)':>10} | {'(per call)':>10} | "
        f"{'(32L, all)':>14} | {'(32L, all)':>14} | {'(32L, all)':>14} | "
        f"{'ratio':>8} | {'ratio':>10}"
    )
    print("-" * 148)

    for seq_len in seq_lens:
        if seq_len <= 4096:
            w, n = 2, 5
        elif seq_len <= 32768:
            w, n = 1, 3
        else:
            w, n = 1, 2

        # Fused TQ (Track A)
        tq_ms, tq_bytes_per_layer, tq_idx_per_layer, tq_norm_per_layer = float("nan"), 0, 0, 0
        try:
            tq_ms, tq_bytes_per_layer, tq_idx_per_layer, tq_norm_per_layer = bench_fused_tq(
                mesh, num_devices, seq_len, warmup=w, iters=n
            )
        except Exception as e:
            print(f"  [TQ FD seq={seq_len} failed: {type(e).__name__}: {str(e)[:80]}]")
        gc.collect()

        # Track B: paged BFP4 + standard SDPA
        bfp4_ms, bfp4_bytes_per_layer = float("nan"), 0
        try:
            bfp4_ms, bfp4_bytes_per_layer = bench_rescaled_bfp4(mesh, num_devices, seq_len, warmup=w, iters=n)
        except Exception as e:
            print(f"  [BFP4 seq={seq_len} failed: {type(e).__name__}: {str(e)[:80]}]")
        gc.collect()

        # Baseline BFP8
        bf_ms, bf_bytes_per_layer = float("nan"), 0
        try:
            bf_ms, bf_bytes_per_layer = bench_baseline_bfp8(mesh, num_devices, seq_len, warmup=w, iters=n)
        except Exception as e:
            print(f"  [BFP8 seq={seq_len} failed: {type(e).__name__}: {str(e)[:80]}]")
        gc.collect()

        tq_total = tq_bytes_per_layer * N_LAYERS * num_devices
        bfp4_total = bfp4_bytes_per_layer * N_LAYERS * num_devices
        bf_total = bf_bytes_per_layer * N_LAYERS * num_devices
        tq_ratio = tq_total / bf_total if bf_total > 0 else float("nan")
        bfp4_ratio = bfp4_total / bf_total if bf_total > 0 else float("nan")

        print(
            f"{seq_len:>8} | {tq_ms:>10.2f} | {bfp4_ms:>10.2f} | {bf_ms:>10.2f} | "
            f"{fmt_bytes(tq_total):>14} | {fmt_bytes(bfp4_total):>14} | {fmt_bytes(bf_total):>14} | "
            f"{tq_ratio:>7.2f}x | {bfp4_ratio:>9.2f}x"
        )

    print("=" * 148)
    ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
