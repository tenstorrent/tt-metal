# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Stress test targeting the Tensix matmul deadlock that causes the ND vit-N300-func failure.
#
# Root cause from CI triage:
#   - Tensix cores deadlock in bmm_large_block_zm_fused_bias_activation
#     (cb_wait_front, noc_semaphore_wait, cb_reserve_back)
#   - Uses block-sharded L1 + MatmulMultiCoreReuseMultiCastProgramConfig on 8x8 grid
#   - Deadlock causes CQ0 dispatch to hang in process_go_signal_mcast_cmd
#   - Which cascades to CQ1 prefetcher stuck in process_stall -> fetch queue timeout
#
# This test isolates the matmul path: no ViT model, no 2CQ, just massive block-sharded
# matmul repetitions with the exact configs from the ViT model to maximize deadlock chance.
#
# The ViT encoder has 12 layers, each with 5 block-sharded matmuls (QKV, self_output,
# FF1, FF2, classifier). We run those same configs thousands of times.
#
# Run: pytest vit_n300/tests/test_matmul_deadlock_stress.py -v -s -x
# Or: ./vit_n300/scripts/stress_test_matmul.sh

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole


# ViT matmul configs extracted from ttnn_optimized_sharded_vit_wh.py (batch_size=8)
# These are the block-sharded matmuls that use bmm_large_block_zm with multicast
VIT_MATMUL_CONFIGS = {
    "qkv": {
        # QKV projection: [batch*seqL_padded, hidden] x [hidden, 3*hidden] on 8x8 grid
        "M": 8 * 224,  # batch_size * seqL_padded
        "K": 768,  # hidden_size
        "N": 3 * 768,  # 3 * hidden_size (Q, K, V)
        "grid": (8, 8),
        "in0_block_w": 3,
        "out_subblock_h": 1,
        "out_subblock_w": 3,
        "per_core_M": 7,
        "per_core_N": 9,
        "fused_activation": None,
    },
    "self_output": {
        # Self-output projection: [batch*seqL_padded, hidden] x [hidden, hidden] on 8x8 grid
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
        # FF1 with GELU: [batch*seqL_padded, hidden] x [hidden, 4*hidden] on 8x8 grid
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
        # FF2: [batch*seqL_padded, 4*hidden] x [4*hidden, hidden] on 8x8 grid
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

# Additional wider matmul configs to increase per-core tile counts and back-pressure
WIDE_MATMUL_CONFIGS = {
    "wide_6144": {
        # Extra-wide: more output tiles per core → more back-pressure on output CB
        "M": 8 * 224,
        "K": 768,
        "N": 6144,  # 192 tiles wide → 24 tiles/core on 8x8
        "grid": (8, 8),
        "in0_block_w": 3,
        "out_subblock_h": 1,
        "out_subblock_w": 8,  # larger subblock → more tiles reserved at once
        "per_core_M": 7,
        "per_core_N": 24,
        "fused_activation": None,
    },
    "wide_gelu": {
        # Wide with GELU activation (exercises the bias+activation deadlock path)
        "M": 8 * 224,
        "K": 768,
        "N": 4096,  # 128 tiles wide → 16 tiles/core on 8x8
        "grid": (8, 8),
        "in0_block_w": 3,
        "out_subblock_h": 1,
        "out_subblock_w": 8,
        "per_core_M": 7,
        "per_core_N": 16,
        "fused_activation": (ttnn.UnaryOpType.GELU, True),
    },
}


def create_block_sharded_config(grid, M, K, N):
    """Create block-sharded memory config matching ViT's layout."""
    grid_x, grid_y = grid
    shard_height = M // (grid_y * 32)  # tiles per core in M dimension
    shard_width_in = K // (grid_x * 32)  # tiles per core in K dimension
    shard_width_out = N // (grid_x * 32)  # tiles per core in N dimension

    shard_spec_in = ttnn.ShardSpec(
        ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, 0),
                    ttnn.CoreCoord(grid_x - 1, grid_y - 1),
                ),
            }
        ),
        [shard_height * 32, shard_width_in * 32],
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    input_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ttnn.BufferType.L1,
        shard_spec_in,
    )

    return input_mem_config


def run_single_matmul(device, name, cfg, weight_tensor):
    """Run one block-sharded matmul with the given config."""
    M, K, N = cfg["M"], cfg["K"], cfg["N"]
    grid = cfg["grid"]

    input_mem_config = create_block_sharded_config(grid, M, K, N)

    # Create input on host, move to device with block sharding
    torch_input = torch.randn(1, 1, M, K, dtype=torch.bfloat16)
    tt_input = ttnn.from_torch(
        torch_input, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=input_mem_config
    )

    program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=grid,
        in0_block_w=cfg["in0_block_w"],
        out_subblock_h=cfg["out_subblock_h"],
        out_subblock_w=cfg["out_subblock_w"],
        per_core_M=cfg["per_core_M"],
        per_core_N=cfg["per_core_N"],
        transpose_mcast=False,
        fused_activation=cfg["fused_activation"],
    )

    output = ttnn.matmul(
        tt_input,
        weight_tensor,
        program_config=program_config,
        memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
        dtype=ttnn.bfloat8_b,
    )

    tt_input.deallocate()
    output.deallocate()


def run_matmul_stress_loop(device, num_iterations, use_trace):
    """Run the ViT matmul configs repeatedly, optionally using trace."""
    # Pre-create weight tensors on device (DRAM, interleaved — like ViT parameters)
    weights = {}
    for name, cfg in VIT_MATMUL_CONFIGS.items():
        torch_weight = torch.randn(1, 1, cfg["K"], cfg["N"], dtype=torch.bfloat16)
        weights[name] = ttnn.from_torch(
            torch_weight,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    if use_trace:
        # Run once to JIT compile
        for name, cfg in VIT_MATMUL_CONFIGS.items():
            run_single_matmul(device, name, cfg, weights[name])
        ttnn.synchronize_device(device)

        # Capture trace of all 4 matmuls (mimics one ViT encoder layer)
        # Pre-allocate inputs for trace
        trace_inputs = {}
        for name, cfg in VIT_MATMUL_CONFIGS.items():
            M, K, N = cfg["M"], cfg["K"], cfg["N"]
            input_mem_config = create_block_sharded_config(cfg["grid"], M, K, N)
            torch_input = torch.randn(1, 1, M, K, dtype=torch.bfloat16)
            trace_inputs[name] = ttnn.from_torch(
                torch_input,
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=input_mem_config,
            )

        # Capture
        trace_id = ttnn.begin_trace_capture(device, cq_id=0)
        trace_outputs = {}
        for name, cfg in VIT_MATMUL_CONFIGS.items():
            program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=cfg["grid"],
                in0_block_w=cfg["in0_block_w"],
                out_subblock_h=cfg["out_subblock_h"],
                out_subblock_w=cfg["out_subblock_w"],
                per_core_M=cfg["per_core_M"],
                per_core_N=cfg["per_core_N"],
                transpose_mcast=False,
                fused_activation=cfg["fused_activation"],
            )
            trace_outputs[name] = ttnn.matmul(
                trace_inputs[name],
                weights[name],
                program_config=program_config,
                memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
                dtype=ttnn.bfloat8_b,
            )
        ttnn.end_trace_capture(device, trace_id, cq_id=0)

        # Replay trace many times
        logger.info(f"Replaying trace {num_iterations} times (4 block-sharded matmuls per replay)...")
        for i in range(num_iterations):
            ttnn.execute_trace(device, trace_id, cq_id=0, blocking=False)
            if (i + 1) % 1000 == 0:
                ttnn.synchronize_device(device)
                logger.info(f"  ...completed {i + 1}/{num_iterations} trace replays")

        ttnn.synchronize_device(device)
        ttnn.release_trace(device, trace_id)
    else:
        # No trace — run matmuls directly
        logger.info(f"Running {num_iterations} iterations of 4 block-sharded matmuls (no trace)...")
        for i in range(num_iterations):
            for name, cfg in VIT_MATMUL_CONFIGS.items():
                run_single_matmul(device, name, cfg, weights[name])
            if (i + 1) % 100 == 0:
                ttnn.synchronize_device(device)
                logger.info(f"  ...completed {i + 1}/{num_iterations} iterations")

        ttnn.synchronize_device(device)

    # Cleanup weights
    for w in weights.values():
        w.deallocate()


@pytest.mark.skipif(is_blackhole(), reason="Unsupported on BH")
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 32768, "trace_region_size": 1753088}],
    indirect=True,
)
@pytest.mark.parametrize("num_iterations", [10000])
@pytest.mark.parametrize("use_trace", [True, False], ids=["traced", "direct"])
def test_matmul_deadlock_stress(device, num_iterations, use_trace):
    """
    Stress test: spam block-sharded matmuls with multicast on 8x8 grid.
    Uses the exact ViT matmul configs that deadlocked in CI.
    Traced variant replays 10000× (= 40,000 matmul ops).
    Direct variant runs 10000× without trace (= 40,000 matmul ops).
    """
    torch.manual_seed(0)
    total_matmuls = num_iterations * len(VIT_MATMUL_CONFIGS)
    logger.info(
        f"Matmul deadlock stress: {num_iterations} iters × {len(VIT_MATMUL_CONFIGS)} matmuls "
        f"= {total_matmuls} block-sharded matmul ops ({'traced' if use_trace else 'direct'})"
    )

    run_matmul_stress_loop(device, num_iterations, use_trace)


@pytest.mark.skipif(is_blackhole(), reason="Unsupported on BH")
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 32768, "trace_region_size": 3506176}],
    indirect=True,
)
@pytest.mark.parametrize("num_iterations", [10000])
def test_matmul_deadlock_stress_wide(device, num_iterations):
    """
    Stress test with wider matmuls: more tiles per core → more back-pressure.
    These configs push larger output subblocks through the pipeline, increasing
    the chance of CB full / CB empty deadlock conditions.
    """
    torch.manual_seed(0)
    all_configs = {**VIT_MATMUL_CONFIGS, **WIDE_MATMUL_CONFIGS}
    total_matmuls = num_iterations * len(all_configs)
    logger.info(
        f"Matmul deadlock stress (wide): {num_iterations} iters × {len(all_configs)} matmuls "
        f"= {total_matmuls} block-sharded matmul ops (direct, no trace)"
    )

    # Pre-create weight tensors
    weights = {}
    for name, cfg in all_configs.items():
        torch_weight = torch.randn(1, 1, cfg["K"], cfg["N"], dtype=torch.bfloat16)
        weights[name] = ttnn.from_torch(
            torch_weight,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    logger.info(f"Running {num_iterations} iterations of {len(all_configs)} matmuls (direct, wide)...")
    for i in range(num_iterations):
        for name, cfg in all_configs.items():
            run_single_matmul(device, name, cfg, weights[name])
        if (i + 1) % 100 == 0:
            ttnn.synchronize_device(device)
            logger.info(f"  ...completed {i + 1}/{num_iterations} iterations")

    ttnn.synchronize_device(device)

    for w in weights.values():
        w.deallocate()


COPIES_PER_ITERATION = 5  # Multiple copies per trace replay to increase NoC contention


@pytest.mark.skipif(is_blackhole(), reason="Unsupported on BH")
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 32768, "num_command_queues": 2, "trace_region_size": 1753088}],
    indirect=True,
)
@pytest.mark.parametrize("num_iterations", [10000])
def test_matmul_deadlock_stress_2cq(device, num_iterations):
    """
    Stress test with 2CQ: trace replays matmuls on CQ0 while CQ1 does concurrent copies.
    This matches the exact failure scenario: CQ0 trace matmul deadlock + CQ1 copy stall.
    Now with heavier CQ1 traffic: multiple copies per trace replay iteration.
    """
    torch.manual_seed(0)
    total_matmuls = num_iterations * len(VIT_MATMUL_CONFIGS)
    total_copies = num_iterations * COPIES_PER_ITERATION
    logger.info(
        f"Matmul deadlock stress (2CQ): {num_iterations} iters × {len(VIT_MATMUL_CONFIGS)} matmuls "
        f"= {total_matmuls} matmul ops + {total_copies} CQ1 copies"
    )

    # Pre-create weight tensors
    weights = {}
    for name, cfg in VIT_MATMUL_CONFIGS.items():
        torch_weight = torch.randn(1, 1, cfg["K"], cfg["N"], dtype=torch.bfloat16)
        weights[name] = ttnn.from_torch(
            torch_weight,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    # Create multiple copy buffers on CQ1 (heavier NoC traffic)
    copy_buffers_host = []
    copy_buffers_device = []
    for c in range(COPIES_PER_ITERATION):
        copy_shape = (1, 1, 8 * 224, 64)  # batch * seqL_padded × small width
        torch_copy_data = torch.randn(*copy_shape, dtype=torch.bfloat16)
        tt_host = ttnn.from_torch(torch_copy_data, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
        tt_dev = tt_host.to(device, ttnn.DRAM_MEMORY_CONFIG)
        copy_buffers_host.append(tt_host)
        copy_buffers_device.append(tt_dev)

    # JIT compile all matmuls
    for name, cfg in VIT_MATMUL_CONFIGS.items():
        run_single_matmul(device, name, cfg, weights[name])
    ttnn.synchronize_device(device)

    # Capture trace of matmuls on CQ0
    trace_inputs = {}
    for name, cfg in VIT_MATMUL_CONFIGS.items():
        M, K, N = cfg["M"], cfg["K"], cfg["N"]
        input_mem_config = create_block_sharded_config(cfg["grid"], M, K, N)
        torch_input = torch.randn(1, 1, M, K, dtype=torch.bfloat16)
        trace_inputs[name] = ttnn.from_torch(
            torch_input,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=input_mem_config,
        )

    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    for name, cfg in VIT_MATMUL_CONFIGS.items():
        program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=cfg["grid"],
            in0_block_w=cfg["in0_block_w"],
            out_subblock_h=cfg["out_subblock_h"],
            out_subblock_w=cfg["out_subblock_w"],
            per_core_M=cfg["per_core_M"],
            per_core_N=cfg["per_core_N"],
            transpose_mcast=False,
            fused_activation=cfg["fused_activation"],
        )
        ttnn.matmul(
            trace_inputs[name],
            weights[name],
            program_config=program_config,
            memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
            dtype=ttnn.bfloat8_b,
        )
    ttnn.end_trace_capture(device, trace_id, cq_id=0)

    # Main loop: CQ0 replays trace, CQ1 does MULTIPLE copies — heavier NoC contention
    cq0_event = ttnn.record_event(device, 0)
    logger.info(
        f"Running 2CQ stress loop: {num_iterations} trace replays " f"with {COPIES_PER_ITERATION} CQ1 copies each..."
    )

    for i in range(num_iterations):
        # CQ0: replay trace (matmuls)
        ttnn.execute_trace(device, trace_id, cq_id=0, blocking=False)

        # CQ1: multiple copies while CQ0 runs matmuls (heavier NoC contention)
        ttnn.wait_for_event(1, cq0_event)
        for c in range(COPIES_PER_ITERATION):
            ttnn.copy_host_to_device_tensor(copy_buffers_host[c], copy_buffers_device[c], 1)
        cq1_event = ttnn.record_event(device, 1)
        ttnn.wait_for_event(0, cq1_event)
        cq0_event = ttnn.record_event(device, 0)

        if (i + 1) % 1000 == 0:
            ttnn.synchronize_device(device)
            logger.info(f"  ...completed {i + 1}/{num_iterations}")

    ttnn.synchronize_device(device)
    ttnn.release_trace(device, trace_id)

    for w in weights.values():
        w.deallocate()
