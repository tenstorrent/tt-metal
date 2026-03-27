# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import ttnn

from .block_sharded_rms_norm_program_descriptor import create_program_descriptor


def block_sharded_rms_norm(
    input_tensor: ttnn.Tensor,
    *,
    epsilon: float = 1e-5,
    memory_config: ttnn.MemoryConfig | None = None,
) -> ttnn.Tensor:
    _validate_input(input_tensor)

    output_memory_config = memory_config if memory_config is not None else input_tensor.memory_config()
    if output_memory_config != input_tensor.memory_config():
        raise ValueError("block_sharded_rms_norm only supports output memory_config identical to the input")

    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(input_tensor.shape)),
        input_tensor.dtype,
        input_tensor.layout,
        input_tensor.device(),
        output_memory_config,
    )

    program_descriptor = create_program_descriptor(input_tensor, output_tensor, epsilon=epsilon)
    return ttnn.generic_op([input_tensor, output_tensor], program_descriptor)


def _validate_input(input_tensor: ttnn.Tensor) -> None:
    if input_tensor.layout != ttnn.TILE_LAYOUT:
        raise ValueError("block_sharded_rms_norm requires TILE_LAYOUT input")
    if input_tensor.dtype != ttnn.bfloat16:
        raise ValueError("block_sharded_rms_norm currently supports bfloat16 input only")
    if not input_tensor.is_sharded():
        raise ValueError("block_sharded_rms_norm requires a sharded input tensor")

    memory_config = input_tensor.memory_config()
    if memory_config.memory_layout != ttnn.TensorMemoryLayout.BLOCK_SHARDED:
        raise ValueError("block_sharded_rms_norm requires BLOCK_SHARDED input")
    if memory_config.buffer_type != ttnn.BufferType.L1:
        raise ValueError("block_sharded_rms_norm requires L1-backed shards")

    shard_spec = memory_config.shard_spec
    if shard_spec is None:
        raise ValueError("block_sharded_rms_norm requires a shard spec")
    if shard_spec.orientation != ttnn.ShardOrientation.ROW_MAJOR:
        raise ValueError("block_sharded_rms_norm currently supports ROW_MAJOR shard orientation only")

    shard_h, shard_w = shard_spec.shape
    if shard_h % 32 != 0 or shard_w % 32 != 0:
        raise ValueError("block_sharded_rms_norm requires shard height and width divisible by 32")
