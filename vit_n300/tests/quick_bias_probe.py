#!/usr/bin/env python3
"""
Quick probe: run bias matmuls for ~30s with DPRINT to gather contention data.
Designed to finish in <1 min. If it hangs, TT_METAL_OPERATION_TIMEOUT_SECONDS
will fire the "device timeout in fetch queue wait" error.

Usage:
  TT_METAL_OPERATION_TIMEOUT_SECONDS=10 \
  TT_METAL_DPRINT_CORES="(7,7)" \
  TT_METAL_DPRINT_RISCVS="BR+NC" \
  TT_METAL_DPRINT_FILE=vit_n300/logs/dprint_probe.log \
  python vit_n300/tests/quick_bias_probe.py
"""
import sys
import time
import torch
import ttnn

P = lambda *a: (print(*a, flush=True),)

CONFIGS = {
    "qkv": {
        "M": 8 * 224,
        "K": 768,
        "N": 3 * 768,
        "grid": (8, 8),
        "in0_block_w": 3,
        "out_subblock_h": 1,
        "out_subblock_w": 3,
        "per_core_M": 7,
        "per_core_N": 9,
        "fused_activation": None,
    },
    "self_output": {
        "M": 8 * 224,
        "K": 768,
        "N": 768,
        "grid": (8, 8),
        "in0_block_w": 3,
        "out_subblock_h": 1,
        "out_subblock_w": 3,
        "per_core_M": 7,
        "per_core_N": 3,
        "fused_activation": None,
    },
    "ff1": {
        "M": 8 * 224,
        "K": 768,
        "N": 4 * 768,
        "grid": (8, 8),
        "in0_block_w": 3,
        "out_subblock_h": 1,
        "out_subblock_w": 6,
        "per_core_M": 7,
        "per_core_N": 12,
        "fused_activation": (ttnn.UnaryOpType.GELU, True),
    },
    "ff2": {
        "M": 8 * 224,
        "K": 4 * 768,
        "N": 768,
        "grid": (8, 8),
        "in0_block_w": 12,
        "out_subblock_h": 1,
        "out_subblock_w": 3,
        "per_core_M": 7,
        "per_core_N": 3,
        "fused_activation": None,
    },
}

NUM_ITERS = 200  # 200 iters × 4 configs = 800 ops, ~20-30s
SYNC_EVERY = 25  # Sync every 25 iters so timeout can fire quickly after hang


def make_shard(grid, M, K):
    gx, gy = grid
    return ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))}),
        [(M // gy // 32) * 32, (K // gx // 32) * 32],
        ttnn.ShardOrientation.ROW_MAJOR,
    )


def run(device):
    torch.manual_seed(0)
    weights, biases = {}, {}
    for name, c in CONFIGS.items():
        weights[name] = ttnn.from_torch(
            torch.randn(1, 1, c["K"], c["N"], dtype=torch.bfloat16),
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        biases[name] = ttnn.from_torch(
            torch.randn(1, 1, 1, c["N"], dtype=torch.bfloat16),
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    P(f"[probe] Starting {NUM_ITERS} iters × {len(CONFIGS)} configs = {NUM_ITERS * len(CONFIGS)} bias-matmul ops")
    t0 = time.time()

    for i in range(NUM_ITERS):
        for name, c in CONFIGS.items():
            mem_in = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                ttnn.BufferType.L1,
                make_shard(c["grid"], c["M"], c["K"]),
            )
            tt_in = ttnn.from_torch(
                torch.randn(1, 1, c["M"], c["K"], dtype=torch.bfloat16),
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=mem_in,
            )
            prog = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=c["grid"],
                in0_block_w=c["in0_block_w"],
                out_subblock_h=c["out_subblock_h"],
                out_subblock_w=c["out_subblock_w"],
                per_core_M=c["per_core_M"],
                per_core_N=c["per_core_N"],
                transpose_mcast=False,
                fused_activation=c["fused_activation"],
            )
            out = ttnn.linear(
                tt_in,
                weights[name],
                bias=biases[name],
                program_config=prog,
                memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
                dtype=ttnn.bfloat8_b,
            )
            tt_in.deallocate()
            out.deallocate()

        if (i + 1) % SYNC_EVERY == 0:
            ttnn.synchronize_device(device)
            elapsed = time.time() - t0
            P(f"[probe] iter {i+1}/{NUM_ITERS}  ({elapsed:.1f}s)")

    ttnn.synchronize_device(device)
    elapsed = time.time() - t0
    P(f"[probe] DONE — {NUM_ITERS * len(CONFIGS)} ops in {elapsed:.1f}s — NO HANG")

    for w in weights.values():
        w.deallocate()
    for b in biases.values():
        b.deallocate()


if __name__ == "__main__":
    P("[probe] Opening device...")
    device = ttnn.open_device(device_id=0, l1_small_size=32768)
    try:
        run(device)
    except Exception as e:
        P(f"\n[probe] CAUGHT ERROR: {e}")
        P("[probe] ^^^ This is the hang we're looking for!")
        sys.exit(1)
    finally:
        try:
            ttnn.close_device(device)
        except Exception:
            P("[probe] Device close failed (expected after hang)")
    P("[probe] Clean exit.")
