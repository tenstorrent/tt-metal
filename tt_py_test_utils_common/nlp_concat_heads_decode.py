# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import torch
import ttnn.core
from tt_py_test_utils_common.comparison_funcs import comp_pcc
from tt_py_test_utils_common.utility_functions import nearest_32, torch2tt_tensor, tt2torch_tensor


def num_to_corerange(x):
    assert x < 8 or x % 8 == 0
    num_x = min(x, 8)
    num_y = x // num_x
    assert num_x * num_y == x
    return ttnn.CoreRange(
        ttnn.CoreCoord(0, 0),
        ttnn.CoreCoord(num_x - 1, num_y - 1),
    )


def run_test_concat_head(
    device, n_local_heads, padded_local_heads, head_dim, batch, input_sub_core_grids=None, compute_sub_core_grids=None
):
    ## Split Heads
    padded_batch = nearest_32(batch)
    seq_len = 1

    # Prepare input
    concat_head_input = torch.rand(1, batch, padded_local_heads, head_dim)

    if input_sub_core_grids is None:
        shard_grid = ttnn.CoreRangeSet({num_to_corerange(batch)})
    else:
        shard_grid = input_sub_core_grids

    # If the provided input sub core grids has enough cores, use that one for compute otherwise
    # we expect a compute sub core grid which must have at least n_local_heads cores to be provided

    if shard_grid.num_cores() >= n_local_heads:
        sub_core_grids = ttnn.num_cores_to_corerangeset_in_subcoregrids(
            shard_grid.bounding_box().start, n_local_heads, shard_grid, row_wise=True
        )
    elif compute_sub_core_grids is not None:
        sub_core_grids = compute_sub_core_grids
        assert (
            sub_core_grids.num_cores() >= n_local_heads
        ), "compute_sub_core_grids must have at least n_local_heads cores"
    else:
        # n_local_heads > input shard grid: auto-generate a compute grid that fits n_local_heads.
        # The op requires input batch <= 32 (one input core per user) but allows num_heads up to
        # the device's worker grid, so this branch is needed to exercise n_local_heads > 32 paths
        # in test_concat_head (which doesn't pass an explicit compute grid).
        sub_core_grids = ttnn.num_cores_to_corerangeset(n_local_heads, device.compute_with_storage_grid_size(), True)

    SCORES_BATCHED_MM_OUTPUT_MEMCFG = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            shard_grid,
            [
                padded_local_heads,  # Each core has padded_local_heads
                head_dim,  # head dim
            ],
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )
    WIDTH_SHARDED_MEMCFG = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1)

    # Prepare tt input
    concat_head_input_tt = torch2tt_tensor(concat_head_input, tt_device=None).to(
        device=device, mem_config=SCORES_BATCHED_MM_OUTPUT_MEMCFG
    )

    concat_head_output = ttnn.experimental.nlp_concat_heads_decode(
        concat_head_input_tt,
        num_heads=n_local_heads,
        sub_core_grids=sub_core_grids,
    )  # seqlen, 1, batch, hidden_size

    logger.debug(f"concat_head_output: {concat_head_output.memory_config()}")

    # Input: (1, 32, 32(8), 128)
    # Output: (1, 1, 32, 1024)
    concat_head_output_torch = concat_head_input[:, :, :n_local_heads].reshape(1, 1, batch, head_dim * n_local_heads)

    # compare
    concat_head_output_tt_cpu = tt2torch_tensor(concat_head_output)
    concat_head_output_tt_unpadded = concat_head_output_tt_cpu[:, :, :batch, :]
    out_pass_q, output_pcc_q = comp_pcc(concat_head_output_tt_unpadded, concat_head_output_torch)
    logger.debug(f"PCC value: {output_pcc_q}")
    assert out_pass_q
