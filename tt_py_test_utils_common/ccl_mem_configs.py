# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from math import ceil
import ttnn


def get_serializable_shard_specs(
    input_shape, input_cores, input_strategy, output_shape, output_cores, output_strategy, valid_tensor_shapes
):
    return {
        "input": {
            "shape": input_shape,
            "cores": input_cores,
            "strategy": input_strategy,
        },
        "output": {"shape": output_shape, "cores": output_cores, "strategy": output_strategy},
        "valid_tensor_shapes": valid_tensor_shapes,
    }


TILE_SIZE = 32


def _parse_serializable_shard_spec(serializable_shard_spec, mem_layout, output_shape, tile_size=TILE_SIZE):
    assert len(serializable_shard_spec) == 3

    shape, cores, strategy = tuple(serializable_shard_spec.values())

    if strategy == "w":
        strategy, layout = ttnn.ShardStrategy.WIDTH, ttnn.TensorMemoryLayout.WIDTH_SHARDED
    elif strategy == "h":
        strategy, layout = ttnn.ShardStrategy.HEIGHT, ttnn.TensorMemoryLayout.HEIGHT_SHARDED
    else:
        raise RuntimeError("Ivalid shard strategy option")

    if shape is None:
        core_grid = ttnn.CoreGrid(**dict(zip(("x", "y"), cores)))
        shard_spec = ttnn.create_sharded_memory_config(output_shape, core_grid, strategy).shard_spec
        if mem_layout == ttnn.TILE_LAYOUT:
            shard_spec.shape = [ceil(s / TILE_SIZE) * TILE_SIZE for s in shard_spec.shape]
        return shard_spec, layout
    else:
        core_grid = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(cores[0] - 1, cores[1] - 1))}
        )
        return ttnn.ShardSpec(core_grid, shape, ttnn.ShardOrientation.ROW_MAJOR), layout


def get_mem_configs(buffer_type, serializable_shard_specs, mem_layout, output_shape):
    if serializable_shard_specs is None:
        return ttnn.MemoryConfig(buffer_type=buffer_type), ttnn.MemoryConfig(buffer_type=buffer_type)
    else:
        input_spec, input_layout = _parse_serializable_shard_spec(serializable_shard_specs["input"], mem_layout, None)
        input_config = ttnn.MemoryConfig(input_layout, buffer_type, input_spec)

        output_spec, output_layout = _parse_serializable_shard_spec(
            serializable_shard_specs["output"], mem_layout, output_shape
        )
        output_config = ttnn.MemoryConfig(output_layout, buffer_type, output_spec)

        assert input_layout == output_layout

    return input_config, output_config
