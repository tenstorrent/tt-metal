
// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <bit>
#include <limits>
#include <string_view>

#include <tt-metalium/bfloat16.hpp>

#include "ttnn/tensor/tensor.hpp"

namespace tt::tt_metal {

enum class ReduceOpMath { SUM, AVG, MAX, MIN, STD, VAR };

enum class ReduceOpDim { H, W, HW };

enum class ReduceOpParallelizationStrategy { MULTI_CORE_H, MULTI_CORE_W, MULTI_CORE_HW, SINGLE_CORE_HW };

}  // namespace tt::tt_metal

namespace ttnn::prim {

// Identity element for the given reduction math op: -inf for MAX, +inf for MIN, 0 otherwise.
// Used by the dense RM paths to pad partial chunks without disturbing the result.
inline float get_reduce_pad_value(tt::tt_metal::ReduceOpMath reduce_math) {
    using tt::tt_metal::ReduceOpMath;
    return reduce_math == ReduceOpMath::MAX   ? -std::numeric_limits<float>::infinity()
           : reduce_math == ReduceOpMath::MIN ? std::numeric_limits<float>::infinity()
                                              : 0.0f;
}

// Bit pattern of the RM padding identity in the input's data format, ready to load into a CB tile.
inline uint32_t dense_rm_padding_identity_bits(tt::DataFormat df, tt::tt_metal::ReduceOpMath op) {
    const float v = get_reduce_pad_value(op);
    if (df == tt::DataFormat::Float32) {
        return std::bit_cast<uint32_t>(v);
    }
    const uint16_t bf16 = std::bit_cast<uint16_t>(bfloat16::truncate(v));
    return static_cast<uint32_t>(bf16);
}

tt::tt_metal::ReduceOpParallelizationStrategy get_parallelization_strategy(
    const tt::tt_metal::Tensor& input_tensors, tt::tt_metal::ReduceOpDim reduce_dim);

// Returns true if the fused-negate H reduce path's CBs fit in available L1.
// The reduce_h_neg compute kernel pushes ntiles tiles per inner-loop iteration;
// to make the FIFO write pointer wrap cleanly across all push sizes, c_4 (acc)
// and c_5 (ineg) are each sized at Ht * lcm(Wt_per_core_g1, Wt_per_core_g2)
// tiles.  For wide reductions this can exceed L1, in which case callers must
// fall back to external negation around a non-fused (regular) reduce.
bool h_reduce_negate_fits_in_l1(
    const tt::tt_metal::Tensor& input_tensor, const std::optional<tt::tt_metal::CoreRangeSet>& sub_core_grids);

// Builds a TensorSpec for a reduction-style op output, given the already
// shape-adjusted output shape and the dimension that was reduced.
//
// `output_layout` selects the physical layout of the result (TILE by default;
// pass ROW_MAJOR for the dense RM reduce paths).
//
// Handles all currently supported output memory layouts:
//   - INTERLEAVED: returns the basic spec.
//   - WIDTH/HEIGHT/BLOCK_SHARDED: delegates to the corresponding TensorSpec
//     builder using the grid/orientation taken from `output_mem_config` if
//     available, otherwise falling back to `input_mem_config`.
//   - ND_SHARDED (TILE output only): copies the ND shard spec (from
//     `output_mem_config` or, as a fallback, `input_mem_config`) and sets the
//     shard shape entries for the reduced dim(s) to 1.
//
// `input_mem_config` is the memory config of the reduction's input tensor and
// is only consulted as a fallback when the output config omits a shard spec.
tt::tt_metal::TensorSpec build_reduce_output_tensor_spec(
    const tt::tt_metal::Shape& output_shape,
    tt::tt_metal::DataType output_dtype,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const tt::tt_metal::MemoryConfig& input_mem_config,
    tt::tt_metal::ReduceOpDim reduce_dim,
    tt::tt_metal::Layout output_layout = tt::tt_metal::Layout::TILE);

// Enforces the documented contract that, for reduction-style ops, any sharded
// participant (input or output) must live in L1.  Sharded layouts and DRAM
// buffers use disjoint coordinate spaces (worker cores vs DRAM bank cores), so
// silently borrowing a grid across buffer types — as the shard-spec fallback
// in `build_reduce_output_tensor_spec` would otherwise allow — produces an
// invalid spec.  Pass an `op_name` (e.g. "reduce", "Std/Var reduction") for a
// readable error message.
void validate_reduce_sharded_buffer_types(
    const tt::tt_metal::MemoryConfig& input_mem_config,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    std::string_view op_name);

}  // namespace ttnn::prim
