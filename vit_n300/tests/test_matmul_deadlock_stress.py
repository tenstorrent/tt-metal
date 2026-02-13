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


def run_single_matmul(device, name, cfg, weight_tensor, bias_tensor=None):
    """Run one block-sharded matmul (or linear with bias) with the given config."""
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

    # Match ViT model: packer_l1_acc=True enables PACKER_L1_ACC in the compute kernel
    compute_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )

    if bias_tensor is not None:
        # Use ttnn.linear with bias — matches actual ViT model (FUSE_BIAS path)
        output = ttnn.linear(
            tt_input,
            weight_tensor,
            bias=bias_tensor,
            program_config=program_config,
            memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
            dtype=ttnn.bfloat8_b,
            compute_kernel_config=compute_config,
        )
    else:
        output = ttnn.matmul(
            tt_input,
            weight_tensor,
            program_config=program_config,
            memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
            dtype=ttnn.bfloat8_b,
            compute_kernel_config=compute_config,
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


def run_fast_bias_matmul(device, name, cfg, weight_tensor, bias_tensor, input_dram):
    """Fast bias matmul: reshard DRAM input to L1 block-sharded, run linear, deallocate."""
    grid = cfg["grid"]
    input_mem_config = create_block_sharded_config(grid, cfg["M"], cfg["K"], cfg["N"])

    # Reshard from DRAM to L1 block-sharded (much faster than from_torch each time)
    tt_input = ttnn.to_memory_config(input_dram, input_mem_config)

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

    compute_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )

    output = ttnn.linear(
        tt_input,
        weight_tensor,
        bias=bias_tensor,
        program_config=program_config,
        memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
        dtype=ttnn.bfloat8_b,
        compute_kernel_config=compute_config,
    )
    tt_input.deallocate()
    output.deallocate()


@pytest.mark.skipif(is_blackhole(), reason="Unsupported on BH")
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 32768, "trace_region_size": 1753088}],
    indirect=True,
)
@pytest.mark.parametrize("num_iterations", [10000])
def test_matmul_with_bias(device, num_iterations):
    """
    Fast stress test matching the ACTUAL ViT model: ttnn.linear with bias.
    Pre-creates inputs in DRAM and reshards each iteration (avoids slow from_torch).
    1000 iters × 4 configs = 4000 ops, should complete in ~30-40s.
    With TT_METAL_OPERATION_TIMEOUT_SECONDS=10, hangs produce the fetch queue error.
    """
    import time

    torch.manual_seed(0)
    total_ops = num_iterations * len(VIT_MATMUL_CONFIGS)
    print(f"[bias] {num_iterations} iters × {len(VIT_MATMUL_CONFIGS)} = {total_ops} ops (FUSE_BIAS)", flush=True)

    # Pre-create weights, biases, AND inputs on device DRAM
    weights, biases, inputs = {}, {}, {}
    for name, cfg in VIT_MATMUL_CONFIGS.items():
        weights[name] = ttnn.from_torch(
            torch.randn(1, 1, cfg["K"], cfg["N"], dtype=torch.bfloat16),
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        biases[name] = ttnn.from_torch(
            torch.randn(1, 1, 1, cfg["N"], dtype=torch.bfloat16),
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        inputs[name] = ttnn.from_torch(
            torch.randn(1, 1, cfg["M"], cfg["K"], dtype=torch.bfloat16),
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    print(f"[bias] Starting iterations...", flush=True)
    t0 = time.time()
    for i in range(num_iterations):
        for name, cfg in VIT_MATMUL_CONFIGS.items():
            run_fast_bias_matmul(device, name, cfg, weights[name], biases[name], inputs[name])
        if (i + 1) % 100 == 0:
            ttnn.synchronize_device(device)
            elapsed = time.time() - t0
            ops_done = (i + 1) * len(VIT_MATMUL_CONFIGS)
            print(
                f"[bias] iter {i+1}/{num_iterations}  {ops_done} ops  {elapsed:.1f}s  ({ops_done/elapsed:.0f} ops/s)",
                flush=True,
            )

    ttnn.synchronize_device(device)
    elapsed = time.time() - t0
    print(f"[bias] DONE — {total_ops} ops in {elapsed:.1f}s ({total_ops/elapsed:.0f} ops/s) — NO HANG", flush=True)

    for d in [weights, biases, inputs]:
        for t in d.values():
            t.deallocate()


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

    import time

    # Pre-create weight AND bias tensors (FUSE_BIAS path - matches actual ViT)
    weights, biases = {}, {}
    for name, cfg in VIT_MATMUL_CONFIGS.items():
        weights[name] = ttnn.from_torch(
            torch.randn(1, 1, cfg["K"], cfg["N"], dtype=torch.bfloat16),
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        biases[name] = ttnn.from_torch(
            torch.randn(1, 1, 1, cfg["N"], dtype=torch.bfloat16),
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

    # JIT compile all matmuls with bias
    for name, cfg in VIT_MATMUL_CONFIGS.items():
        run_single_matmul(device, name, cfg, weights[name], biases[name])
    ttnn.synchronize_device(device)

    # Capture trace of matmuls with FUSE_BIAS on CQ0
    trace_inputs = {}
    for name, cfg in VIT_MATMUL_CONFIGS.items():
        M, K, N = cfg["M"], cfg["K"], cfg["N"]
        input_mem_config = create_block_sharded_config(cfg["grid"], M, K, N)
        trace_inputs[name] = ttnn.from_torch(
            torch.randn(1, 1, M, K, dtype=torch.bfloat16),
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
        compute_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=True,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        ttnn.linear(
            trace_inputs[name],
            weights[name],
            bias=biases[name],
            program_config=program_config,
            memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
            dtype=ttnn.bfloat8_b,
            compute_kernel_config=compute_config,
        )
    ttnn.end_trace_capture(device, trace_id, cq_id=0)

    # Main loop: CQ0 replays trace (matmuls+bias), CQ1 does copies — heavier NoC contention
    cq0_event = ttnn.record_event(device, 0)
    print(
        f"[2cq+bias] Running {num_iterations} trace replays " f"with {COPIES_PER_ITERATION} CQ1 copies each...",
        flush=True,
    )

    t0 = time.time()
    for i in range(num_iterations):
        # CQ0: replay trace (matmuls with bias)
        ttnn.execute_trace(device, trace_id, cq_id=0, blocking=False)

        # CQ1: multiple copies while CQ0 runs matmuls (heavier NoC contention)
        ttnn.wait_for_event(1, cq0_event)
        for c in range(COPIES_PER_ITERATION):
            ttnn.copy_host_to_device_tensor(copy_buffers_host[c], copy_buffers_device[c], 1)
        cq1_event = ttnn.record_event(device, 1)
        ttnn.wait_for_event(0, cq1_event)
        cq0_event = ttnn.record_event(device, 0)

        if (i + 1) % 500 == 0:
            ttnn.synchronize_device(device)
            elapsed = time.time() - t0
            print(f"[2cq+bias] iter {i+1}/{num_iterations}  ({elapsed:.1f}s)", flush=True)

    ttnn.synchronize_device(device)
    elapsed = time.time() - t0
    print(f"[2cq+bias] DONE — {num_iterations} trace replays in {elapsed:.1f}s — NO HANG", flush=True)
    ttnn.release_trace(device, trace_id)

    for w in weights.values():
        w.deallocate()
    for b in biases.values():
        b.deallocate()


@pytest.mark.skipif(is_blackhole(), reason="Unsupported on BH")
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 32768, "trace_region_size": 1753088}],
    indirect=True,
)
@pytest.mark.parametrize("num_iterations", [1000])
def test_matmul_l1acc_correctness(device, num_iterations):
    """
    Correctness test: run matmul+bias with packer_l1_acc=True and compare
    output against a golden reference computed with packer_l1_acc=False.
    If the L1_ACC race causes data corruption (wrong accumulation), the output
    will differ from the reference. This catches the first-order effect of the
    race condition even when it doesn't cause a hang.
    """
    import time

    torch.manual_seed(42)

    # Use the self_output config (smallest, fastest to compute golden)
    cfg = VIT_MATMUL_CONFIGS["self_output"]
    M, K, N = cfg["M"], cfg["K"], cfg["N"]
    grid = cfg["grid"]

    compute_config_l1acc = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )
    compute_config_no_l1acc = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=grid,
        in0_block_w=cfg["in0_block_w"],
        out_subblock_h=cfg["out_subblock_h"],
        out_subblock_w=cfg["out_subblock_w"],
        per_core_M=cfg["per_core_M"],
        per_core_N=cfg["per_core_N"],
        transpose_mcast=False,
        fused_activation=None,
    )

    # Create tensors
    torch_input = torch.randn(1, 1, M, K, dtype=torch.bfloat16)
    torch_weight = torch.randn(1, 1, K, N, dtype=torch.bfloat16)
    torch_bias = torch.randn(1, 1, 1, N, dtype=torch.bfloat16)

    input_mem_config = create_block_sharded_config(grid, M, K, N)

    weight_tt = ttnn.from_torch(
        torch_weight,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    bias_tt = ttnn.from_torch(
        torch_bias,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    input_dram = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Compute golden reference (no L1_ACC)
    tt_input_ref = ttnn.to_memory_config(input_dram, input_mem_config)
    ref_output = ttnn.linear(
        tt_input_ref,
        weight_tt,
        bias=bias_tt,
        program_config=program_config,
        memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
        dtype=ttnn.bfloat8_b,
        compute_kernel_config=compute_config_no_l1acc,
    )
    ref_output_dram = ttnn.to_memory_config(ref_output, ttnn.DRAM_MEMORY_CONFIG)
    ref_torch = ttnn.to_torch(ref_output_dram)
    ref_output.deallocate()
    ref_output_dram.deallocate()
    tt_input_ref.deallocate()
    ttnn.synchronize_device(device)

    print(f"[correctness] Golden computed. Running {num_iterations} L1_ACC iterations...", flush=True)
    errors_found = 0
    t0 = time.time()

    for i in range(num_iterations):
        tt_input = ttnn.to_memory_config(input_dram, input_mem_config)
        output = ttnn.linear(
            tt_input,
            weight_tt,
            bias=bias_tt,
            program_config=program_config,
            memory_config=ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG,
            dtype=ttnn.bfloat8_b,
            compute_kernel_config=compute_config_l1acc,
        )
        output_dram = ttnn.to_memory_config(output, ttnn.DRAM_MEMORY_CONFIG)
        result_torch = ttnn.to_torch(output_dram)
        output.deallocate()
        output_dram.deallocate()
        tt_input.deallocate()

        # Compare
        if not torch.allclose(result_torch, ref_torch, atol=0, rtol=0):
            diff = (result_torch - ref_torch).abs()
            max_diff = diff.max().item()
            num_diff = (diff > 0).sum().item()
            total = diff.numel()
            errors_found += 1
            print(
                f"  *** MISMATCH iter {i}: max_diff={max_diff:.6f} "
                f"num_diff={num_diff}/{total} ({100*num_diff/total:.2f}%) ***",
                flush=True,
            )
            if errors_found >= 10:
                print("  Stopping after 10 mismatches", flush=True)
                break

        if (i + 1) % 100 == 0:
            ttnn.synchronize_device(device)
            elapsed = time.time() - t0
            print(
                f"[correctness] iter {i+1}/{num_iterations} ({elapsed:.1f}s) " f"errors={errors_found}",
                flush=True,
            )

    ttnn.synchronize_device(device)
    elapsed = time.time() - t0
    weight_tt.deallocate()
    bias_tt.deallocate()
    input_dram.deallocate()

    print(
        f"[correctness] DONE — {num_iterations} iterations in {elapsed:.1f}s, " f"{errors_found} mismatches",
        flush=True,
    )
    if errors_found > 0:
        print(f"*** L1_ACC RACE DETECTED: {errors_found} mismatches ***", flush=True)
    assert errors_found == 0, f"L1_ACC race: {errors_found} mismatches in {num_iterations} iterations"
