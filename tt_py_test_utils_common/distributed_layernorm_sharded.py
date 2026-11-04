# SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import math


def rms_norm(x, gamma, eps):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * gamma


def create_input_and_weight_tensors(input_width, num_devices, seed, mean, std):
    torch.manual_seed(seed)
    input_shape = (1, 1, 32, input_width * num_devices)
    weights_shape = (1, 1, 1, input_width * num_devices)

    torch_input_tensor = torch.normal(mean, std, size=input_shape, dtype=torch.bfloat16)
    torch_weight = torch.normal(mean, std, size=weights_shape, dtype=torch.bfloat16)

    torch_input_chunks = torch.chunk(torch_input_tensor, num_devices, dim=-1)
    torch_weight_chunks = torch.chunk(torch_weight, num_devices, dim=-1)

    return torch_input_tensor, torch_weight, torch_input_chunks, torch_weight_chunks


def create_tt_tensors(
    torch_chunk, device, df, core_grid, input_width, is_weight=False, grid_offset=ttnn.CoreCoord(0, 0)
):
    tt_tensor = ttnn.from_torch(
        torch_chunk,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG if is_weight else ttnn.L1_MEMORY_CONFIG,
        dtype=df,
    )

    with device.cache_entries_counter.measure():
        if not is_weight:
            core_range = ttnn.CoreRange(
                grid_offset, ttnn.CoreCoord(core_grid[0] + grid_offset.x - 1, core_grid[1] + grid_offset.y - 1)
            )
            tt_sharded_config = ttnn.create_sharded_memory_config(
                shape=(32, input_width // (core_grid[0] * core_grid[1])),
                core_grid=ttnn.CoreRangeSet(
                    {
                        core_range,
                    }
                ),
                strategy=ttnn.ShardStrategy.WIDTH,
                use_height_and_width_as_shard_shape=True,
            )
            tt_tensor = ttnn.to_memory_config(tt_tensor, memory_config=tt_sharded_config)

        return tt_tensor


def compute_pre_allgather_stats(tt_input_tensor, core_grid, input_width, is_rmsnorm, residual_input_tensor=None):
    SHARDED_NORM_PRGM_CFG = ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=[core_grid[0], core_grid[1]],
        subblock_w=(input_width // (core_grid[0] * core_grid[1])) // 32,
        block_h=1,
        block_w=(input_width // (core_grid[0] * core_grid[1])) // 32,
        inplace=False,
    )

    if is_rmsnorm:
        return ttnn.rms_norm_pre_all_gather(
            tt_input_tensor, residual_input_tensor=residual_input_tensor, program_config=SHARDED_NORM_PRGM_CFG
        )
    else:
        return ttnn.layer_norm_pre_all_gather(
            tt_input_tensor, residual_input_tensor=residual_input_tensor, program_config=SHARDED_NORM_PRGM_CFG
        )


def compute_post_allgather_output(
    tt_input_tensor,
    tt_weights,
    tt_stats_tensor,
    eps,
    is_rmsnorm,
    core_grid,
    input_width,
    output_df,
    out_memory_config,
    tt_bias=None,
):
    SHARDED_NORM_PRGM_CFG = ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=(core_grid[0], core_grid[1]),
        subblock_w=(input_width // (core_grid[0] * core_grid[1])) // 32,
        block_h=1,
        block_w=(input_width // (core_grid[0] * core_grid[1])) // 32,
        inplace=False,
    )

    if is_rmsnorm:
        return ttnn.rms_norm_post_all_gather(
            tt_input_tensor,
            epsilon=eps,
            weight=tt_weights,
            bias=tt_bias,
            program_config=SHARDED_NORM_PRGM_CFG,
            stats=tt_stats_tensor,
            dtype=output_df,
            memory_config=out_memory_config,
        )
    else:
        return ttnn.layer_norm_post_all_gather(
            tt_input_tensor,
            epsilon=eps,
            weight=tt_weights,
            bias=tt_bias,
            program_config=SHARDED_NORM_PRGM_CFG,
            stats=tt_stats_tensor,
            dtype=output_df,
            memory_config=out_memory_config,
        )


def compute_reference_output(torch_input_tensor, torch_weight, is_rmsnorm, eps, torch_bias=None):
    if is_rmsnorm:
        gamma = torch_weight
        if gamma is None:
            gamma = torch.ones((1, 1, 1, torch_input_tensor.shape[-1]), dtype=torch_input_tensor.dtype)
        torch_output_tensor = rms_norm(torch_input_tensor, gamma, eps=eps)
        if torch_bias is not None:
            torch_output_tensor = torch_output_tensor + torch_bias
        return torch_output_tensor
    else:
        weight = None if torch_weight is None else torch_weight.squeeze(0).squeeze(0).squeeze(0)
        bias = None if torch_bias is None else torch_bias.squeeze(0).squeeze(0).squeeze(0)
        return torch.nn.functional.layer_norm(
            torch_input_tensor,
            (torch_input_tensor.shape[-1],),
            weight=weight,
            bias=bias,
            eps=eps,
        )


def create_output_memory_config(output_core_grid, input_shape):
    if isinstance(output_core_grid, tuple):
        output_core_range_set = ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(output_core_grid[0] - 1, output_core_grid[1] - 1)),
            ]
        )
    else:
        output_core_range_set = ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(
                    ttnn.CoreCoord(x, y),
                    ttnn.CoreCoord(x, y),
                )
                for x, y in output_core_grid
            ]
        )
    padded_out_w = math.ceil(input_shape[3] / output_core_range_set.num_cores() / 32) * 32
    output_memory_config = ttnn.create_sharded_memory_config(
        shape=(
            input_shape[0] * input_shape[1] * input_shape[2],
            padded_out_w,
        ),
        core_grid=output_core_range_set,
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    return output_memory_config
