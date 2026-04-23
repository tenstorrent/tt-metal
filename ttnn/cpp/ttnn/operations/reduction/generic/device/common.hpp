
// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace tt::tt_metal {

enum class ReduceOpMath { SUM, AVG, MAX, MIN, STD, VAR };

enum class ReduceOpDim { H, W, HW };

enum class ReduceOpParallelizationStrategy { MULTI_CORE_H, MULTI_CORE_W, MULTI_CORE_HW, SINGLE_CORE_HW };

}  // namespace tt::tt_metal

namespace ttnn::prim {

tt::tt_metal::ReduceOpParallelizationStrategy get_parallelization_strategy(
    const tt::tt_metal::Tensor& input_tensors, tt::tt_metal::ReduceOpDim reduce_dim);

// Builds a tilized TensorSpec for a reduction-style op output, given the
// already shape-adjusted output shape and the dimension that was reduced.
//
// Handles all currently supported output memory layouts:
//   - INTERLEAVED: returns the basic spec.
//   - WIDTH/HEIGHT/BLOCK_SHARDED: delegates to the corresponding TensorSpec
//     builder using the grid/orientation taken from `output_mem_config` if
//     available, otherwise falling back to `input_mem_config`.
//   - ND_SHARDED: copies the ND shard spec (from `output_mem_config` or, as a
//     fallback, `input_mem_config`) and sets the shard shape entries for the
//     reduced dim(s) to 1.
//
// `input_mem_config` is the memory config of the reduction's input tensor and
// is only consulted as a fallback when the output config omits a shard spec.
tt::tt_metal::TensorSpec build_reduce_output_tensor_spec(
    const tt::tt_metal::Shape& output_shape,
    tt::tt_metal::DataType output_dtype,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const tt::tt_metal::MemoryConfig& input_mem_config,
    tt::tt_metal::ReduceOpDim reduce_dim);

}  // namespace ttnn::prim
