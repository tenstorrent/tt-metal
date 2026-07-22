
// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <bit>
#include <cstdint>
#include <limits>
#include <string_view>
#include <vector>

#include <tt-metalium/bfloat16.hpp>

#include "ttnn/tensor/tensor.hpp"

namespace tt::tt_metal {
class Buffer;
class MeshTensor;
}  // namespace tt::tt_metal

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

// True when the reduce uses the SFPU path instead of the FPU GMPOOL/matmul path.
// Int32 MAX/SUM always use SFPU (FPU has no Int32 support); MIN is lowered to MAX + negate on the
// host before reaching the factories. Float32 SUM opts into SFPU only when the host requests the
// accurate ttnn.mean path (`use_sfpu_reduce`): the FPU path truncates fp32 to tf32,
// so accumulating register-to-register in the SFPU preserves full fp32. mean is lowered to SUM +
// a 1/N post-mul before this is consulted, so only SUM (never AVG) is checked for fp32.
inline bool use_sfpu_reduce_path(
    tt::tt_metal::DataType dtype, tt::tt_metal::ReduceOpMath math_op, bool use_sfpu_reduce = false) {
    using tt::tt_metal::ReduceOpMath;
    if (dtype == tt::tt_metal::DataType::INT32) {
        return math_op == ReduceOpMath::MAX || math_op == ReduceOpMath::SUM;
    }
    return use_sfpu_reduce && dtype == tt::tt_metal::DataType::FLOAT32 && math_op == ReduceOpMath::SUM;
}

// True when a non-unity scalar must be a post-reduce multiply instead of via the scaler CB: MAX/MIN,
// the Int32 SFPU path, and the accurate fp32 SFPU mean all ignore the scaler CB (fp32 matches AVG/SUM).
inline bool requires_post_mul(
    tt::tt_metal::ReduceOpMath math_op, tt::tt_metal::DataType dtype, float scaler, bool use_sfpu_reduce = false) {
    using tt::tt_metal::ReduceOpMath;
    if (scaler == 1.0f) {
        return false;
    }
    if (math_op == ReduceOpMath::MAX) {
        return true;
    }
    if (math_op == ReduceOpMath::SUM && dtype == tt::tt_metal::DataType::INT32) {
        return true;
    }
    return use_sfpu_reduce && dtype == tt::tt_metal::DataType::FLOAT32 &&
           (math_op == ReduceOpMath::SUM || math_op == ReduceOpMath::AVG);
}

// All RM-path locals derived from the input shape, tile geometry, and math op.
// One instance is populated at the top of the RM branch in each factory and consumed
// by the build_rm_*_ct_args helpers; both factories see the same field layout.
struct RmPlan {
    uint32_t H_logical;
    uint32_t W_logical;
    uint32_t Ht_rm;                  // ceil_div(H_logical, rm_rows_per_tile)
    uint32_t Wt;                     // ceil_div(W_padded,   tile_width)
    uint32_t rm_rows_per_tile;       // == tile_height
    uint32_t wt_tiles_per_chunk;     // W-reduce: min(8, max(1, Wt)); H-reduce: 1
    uint32_t ht_tiles_per_chunk;     // W-reduce: 1;                   H-reduce: min(8, max(1, Ht_rm))
    uint32_t chunk_row_bytes;        // wt_tiles_per_chunk * tile_width * src_datum_size
    uint32_t rm_staging_page_size;   // == chunk_row_bytes (one CB page = one chunk-wide RM row)
    uint32_t padding_identity_bits;  // dense_rm_padding_identity_bits(src_df, math_op)
    uint32_t src_datum_size;
    uint32_t dst_datum_size;
};

// Populate an RmPlan from the input's padded + logical shapes, tile geometry, data formats
// and the dim being reduced. Dim picks which of {wt,ht}_tiles_per_chunk is the variable
// chunk size and which is pinned to 1.
RmPlan make_rm_plan(
    const tt::tt_metal::Shape& padded_shape,
    const tt::tt_metal::Shape& logical_shape,
    uint32_t tile_height,
    uint32_t tile_width,
    tt::DataFormat src_cb_data_format,
    tt::DataFormat dst_cb_data_format,
    tt::tt_metal::ReduceOpMath math_op,
    tt::tt_metal::ReduceOpDim dim);

// The factory-level RM preconditions: interleaved I/O, SUM only, no negate, dim is H or W.
// `dim_label` is "Reduce W" / "Reduce H" for the fatal messages.
void validate_rm_preconditions(
    const tt::tt_metal::MeshTensor& input,
    const tt::tt_metal::MeshTensor& output,
    tt::tt_metal::ReduceOpMath math_op,
    bool negate,
    tt::tt_metal::ReduceOpDim dim,
    std::string_view dim_label);

// Build the reader compile-time args vector for the RM path (slots match
// reader_unary_reduce_rm.cpp). Returns scalar slots followed by TensorAccessorArgs(src).
std::vector<uint32_t> build_rm_reader_ct_args(
    const RmPlan& plan, uint32_t scaler_bits, const tt::tt_metal::MeshTensor& src, tt::tt_metal::ReduceOpDim dim);

// Build the writer compile-time args vector for the RM path (slots match
// writer_reduce_rm_scalar.cpp). Returns scalar slots followed by TensorAccessorArgs(dst).
std::vector<uint32_t> build_rm_writer_ct_args(
    const RmPlan& plan, const tt::tt_metal::MeshTensor& dst, tt::tt_metal::ReduceOpDim dim);

// Build the compute compile-time args vector for the RM path (slots match reduce_rm.cpp).
// `Ht_arg` is the per-core ht count (W path) or the global Ht_rm (H path); the helper
// keeps NC pinned at 1.
std::vector<uint32_t> build_rm_compute_ct_args(const RmPlan& plan, uint32_t Ht_arg, uint32_t post_mul_scaler_bits);

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
