# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Isolated W stream bench (run under `scripts/run_safe_pytest.sh --profile` for timing).

PS_VARIANTS="acc,pref,rot,..." (run order = list order); "acc" = TensorAccessor baseline, the rest are port
policies from ports.py over raw NoC addresses. PS_VERIFY=1 writes the landed slices back and checks them
bit-exactly against W (no timing value then). PS_N overrides W's width (domain sweep), PS_K its height.
"""

import os
from pathlib import Path

import pytest
import ttnn

import ttnn.operations.wq_b_linear.wq_b_linear_program_descriptor as pd
from ttnn.operations.wq_b_linear.perf_experiments.w_dram_port_spread.ports import policy

HERE = Path(__file__).parent
VARIANTS = os.environ.get("PS_VARIANTS", "acc,pref,fix11,rot,rot_core,core3,row3,nearest").split(",")
VERIFY = int(os.environ.get("PS_VERIFY", "0"))
N = int(os.environ.get("PS_N", "4096"))
K = int(os.environ.get("PS_K", "1536"))
W_DTYPE = getattr(ttnn, os.environ.get("PS_DTYPE", "bfloat8_b"))
CB_W = 1


def _descriptor(device, tw, out, variant):
    tile = tw.tile.tile_shape
    tensor_k_tiles = tw.shape[0] // tile[0]
    tensor_n_tiles = tw.shape[1] // tile[1]
    k_block_tiles = pd._largest_divisor_at_most(tensor_k_tiles, pd.K_BLOCK_TILES)
    num_k_blocks = tensor_k_tiles // k_block_tiles
    grid = device.compute_with_storage_grid_size()
    num_banks = ttnn.device.get_memory_view(device, ttnn.BufferType.DRAM).num_banks
    n_groups = pd._split_n(grid, tensor_n_tiles, num_banks)
    assignment = pd._assign_columns(n_groups, tensor_n_tiles, num_banks)
    ordered = [c for cs, _ in n_groups for c in cs]
    page = tw.buffer_page_size()
    kernels = []
    for cores, n_tiles in n_groups:
        core_set = pd._core_range_set(cores)
        ct = [
            CB_W,
            tensor_n_tiles,
            k_block_tiles,
            num_k_blocks,
            int(variant == "acc"),
            VERIFY,
            num_banks,
        ]
        ct += list(ttnn.TensorAccessorArgs(tw).get_compile_time_args())
        ct += list(ttnn.TensorAccessorArgs(out).get_compile_time_args())
        rt = ttnn.RuntimeArgs()
        for c in cores:
            n_start, nt, stride = assignment[(c.x, c.y)]
            assert stride in (1, num_banks)
            pol = policy(variant, ordered.index(c), c, device.worker_core_from_logical_core(c))
            rt[c.x][c.y] = [tw.buffer_address(), out.buffer_address(), n_start, nt, stride, *pol]
        kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=str(HERE / "kernels" / "bench_w_reader.cpp"),
                core_ranges=core_set,
                compile_time_args=ct,
                runtime_args=rt,
                config=ttnn.DataMovementConfigDescriptor(
                    processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_0
                ),
            )
        )
    all_cores = pd._core_range_set(ordered)
    cbs = [
        ttnn.CBDescriptor(
            total_size=k_block_tiles * num_k_blocks * n_groups[0][1] * page,
            core_ranges=all_cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=CB_W, data_format=tw.dtype, page_size=page)],
        )
    ]
    return ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)


@pytest.mark.parametrize("variant", VARIANTS)
def test_w_stream(device, variant):
    import torch

    torch.manual_seed(0)
    w = (torch.randn(K, N) * 0.02).bfloat16()
    tw = ttnn.from_torch(w, dtype=W_DTYPE, layout=ttnn.TILE_LAYOUT, device=device)
    out = ttnn.from_torch(torch.zeros(K, N).bfloat16(), dtype=W_DTYPE, layout=ttnn.TILE_LAYOUT, device=device)
    res = ttnn.generic_op([tw, out], _descriptor(device, tw, out, variant))
    ttnn.ReadDeviceProfiler(device)
    if VERIFY:
        assert torch.equal(ttnn.to_torch(res), ttnn.to_torch(tw)), variant
