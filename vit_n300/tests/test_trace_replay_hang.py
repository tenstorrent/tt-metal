# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
# Trace-replay + 2CQ stress test for ViT N300 ND hang reproduction.
# Run: TT_METAL_OPERATION_TIMEOUT_SECONDS=5 pytest vit_n300/tests/test_trace_replay_hang.py -v -s -x

import pytest
import time
import torch
from loguru import logger
import ttnn
from models.common.utility_functions import is_blackhole


def create_block_sharded_config(grid, M, K, N):
    grid_x, grid_y = grid if isinstance(grid, tuple) else (grid.x, grid.y)
    M_tiles = M // 32
    N_tiles = N // 32
    per_core_M = M_tiles // grid_y
    per_core_N = N_tiles // grid_x
    shard_shape = [per_core_M * 32, per_core_N * 32]
    shard_grid = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_x - 1, grid_y - 1)),
        }
    )
    shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(ttnn.types.TensorMemoryLayout.BLOCK_SHARDED, ttnn.types.BufferType.L1, shard_spec)


@pytest.mark.skipif(is_blackhole(), reason="Unsupported on BH")
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 32768, "num_command_queues": 2, "trace_region_size": 1753088}], indirect=True
)
@pytest.mark.parametrize("num_replays", [100000])
def test_trace_replay_matmul_hang(device, num_replays):
    """Single matmul trace-replay with 2CQ, matching ViT CI pattern."""
    torch.manual_seed(42)
    M, K, N = 8 * 224, 768, 768
    grid = (8, 8)
    prog = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=grid,
        in0_block_w=3,
        out_subblock_h=1,
        out_subblock_w=3,
        per_core_M=7,
        per_core_N=3,
        transpose_mcast=False,
        fused_activation=None,
    )
    in_mem = create_block_sharded_config(grid, M, K, N)
    w = ttnn.from_torch(
        torch.randn(1, 1, K, N, dtype=torch.bfloat16),
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    b = ttnn.from_torch(
        torch.randn(1, 1, 1, N, dtype=torch.bfloat16),
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    d_in = ttnn.from_torch(
        torch.randn(1, 1, M, K, dtype=torch.bfloat16),
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    # JIT
    logger.info("JIT run...")
    il = ttnn.to_memory_config(d_in, in_mem)
    out = ttnn.linear(
        il, w, bias=b, program_config=prog, memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG, dtype=ttnn.bfloat8_b
    )
    ttnn.synchronize_device(device)
    out.deallocate(force=True)
    il.deallocate(force=True)
    logger.info("JIT done.")
    # Trace capture (simple: just capture the matmul, replay with same input)
    logger.info("Capturing trace...")
    il = ttnn.to_memory_config(d_in, in_mem)
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    out = ttnn.linear(
        il, w, bias=b, program_config=prog, memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG, dtype=ttnn.bfloat8_b
    )
    ttnn.end_trace_capture(device, tid, cq_id=0)
    logger.info("Trace captured.")
    # Replay (blocking, same input each time - the key is exercising the L1_ACC race path)
    logger.info(f"Starting {num_replays} trace replays...")
    t0 = time.time()
    for i in range(num_replays):
        ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
        if (i + 1) % 500 == 0:
            ttnn.synchronize_device(device)
            logger.info(f"  Replay {i+1}/{num_replays} ({time.time()-t0:.1f}s)")
    ttnn.synchronize_device(device)
    logger.info(f"All {num_replays} replays in {time.time()-t0:.1f}s. No hang.")
    ttnn.release_trace(device, tid)
    w.deallocate()
    b.deallocate()
    d_in.deallocate()


@pytest.mark.skipif(is_blackhole(), reason="Unsupported on BH")
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 32768, "num_command_queues": 2, "trace_region_size": 1753088}], indirect=True
)
@pytest.mark.parametrize("num_replays", [2000])
def test_trace_replay_multi_matmul_hang(device, num_replays):
    """3 chained matmuls per replay (self_output->FF1->FF2), 3x race windows."""
    torch.manual_seed(42)
    M, K, N, N_ff = 8 * 224, 768, 768, 3072
    grid = (8, 8)
    so_p = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=grid,
        in0_block_w=3,
        out_subblock_h=1,
        out_subblock_w=3,
        per_core_M=7,
        per_core_N=3,
        transpose_mcast=False,
        fused_activation=None,
    )
    f1_p = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=grid,
        in0_block_w=3,
        out_subblock_h=1,
        out_subblock_w=6,
        per_core_M=7,
        per_core_N=12,
        transpose_mcast=False,
        fused_activation=(ttnn.UnaryOpType.GELU, True),
    )
    f2_p = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=grid,
        in0_block_w=12,
        out_subblock_h=1,
        out_subblock_w=3,
        per_core_M=7,
        per_core_N=3,
        transpose_mcast=False,
        fused_activation=None,
    )
    in_mem = create_block_sharded_config(grid, M, K, N)
    ws = ttnn.from_torch(
        torch.randn(1, 1, K, N, dtype=torch.bfloat16),
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    bs_ = ttnn.from_torch(
        torch.randn(1, 1, 1, N, dtype=torch.bfloat16),
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    w1 = ttnn.from_torch(
        torch.randn(1, 1, K, N_ff, dtype=torch.bfloat16),
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    b1 = ttnn.from_torch(
        torch.randn(1, 1, 1, N_ff, dtype=torch.bfloat16),
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    w2 = ttnn.from_torch(
        torch.randn(1, 1, N_ff, N, dtype=torch.bfloat16),
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    b2 = ttnn.from_torch(
        torch.randn(1, 1, 1, N, dtype=torch.bfloat16),
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    d_in = ttnn.from_torch(
        torch.randn(1, 1, M, K, dtype=torch.bfloat16),
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    # JIT
    logger.info("JIT (3 chained matmuls)...")
    il = ttnn.to_memory_config(d_in, in_mem)
    so = ttnn.linear(
        il, ws, bias=bs_, program_config=so_p, memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG, dtype=ttnn.bfloat8_b
    )
    f1 = ttnn.linear(
        so, w1, bias=b1, program_config=f1_p, memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG, dtype=ttnn.bfloat8_b
    )
    f2 = ttnn.linear(
        f1, w2, bias=b2, program_config=f2_p, memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG, dtype=ttnn.bfloat8_b
    )
    ttnn.synchronize_device(device)
    il.deallocate(force=True)
    so.deallocate(force=True)
    f1.deallocate(force=True)
    f2.deallocate(force=True)
    logger.info("JIT done.")
    # Trace capture
    logger.info("Capturing trace of 3 chained matmuls...")
    il = ttnn.to_memory_config(d_in, in_mem)
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    so = ttnn.linear(
        il, ws, bias=bs_, program_config=so_p, memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG, dtype=ttnn.bfloat8_b
    )
    f1 = ttnn.linear(
        so, w1, bias=b1, program_config=f1_p, memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG, dtype=ttnn.bfloat8_b
    )
    f2 = ttnn.linear(
        f1, w2, bias=b2, program_config=f2_p, memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG, dtype=ttnn.bfloat8_b
    )
    ttnn.end_trace_capture(device, tid, cq_id=0)
    logger.info("Trace captured.")
    # Replay
    logger.info(f"Starting {num_replays} replays ({num_replays*3} matmul ops)...")
    t0 = time.time()
    for i in range(num_replays):
        ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
        if (i + 1) % 500 == 0:
            ttnn.synchronize_device(device)
            logger.info(f"  Replay {i+1}/{num_replays} ({time.time()-t0:.1f}s)")
    ttnn.synchronize_device(device)
    logger.info(f"All {num_replays} replays ({num_replays*3} ops) in {time.time()-t0:.1f}s. No hang.")
    ttnn.release_trace(device, tid)
    for t in [ws, bs_, w1, b1, w2, b2, d_in]:
        t.deallocate()
