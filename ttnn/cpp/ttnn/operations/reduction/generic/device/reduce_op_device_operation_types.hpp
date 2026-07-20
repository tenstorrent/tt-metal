// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "common.hpp"

namespace ttnn::prim {

struct ReduceParams {
    tt::tt_metal::ReduceOpMath math_op{};
    tt::tt_metal::ReduceOpDim dim{};
    float scaler{1.0f};
    tt::tt_metal::MemoryConfig output_mem_config;
    tt::tt_metal::DataType output_dtype{tt::tt_metal::DataType::INVALID};
    ttnn::DeviceComputeKernelConfig compute_kernel_config;
    std::optional<tt::tt_metal::CoreRangeSet> sub_core_grids;
    bool negate{false};
    // For min/max with a non-unity scalar, the GMPOOL hardware path (reduce_tile LLK) only
    // respects the exponent of the scaler. To produce numerically correct results for any
    // scalar, the host instead requests reduction with `scaler=1.0` and applies the user
    // scalar afterwards via SFPU post-multiplication (mul_unary_tile) inside the compute
    // kernel, gated by the REDUCE_POST_MUL define. When `post_mul_scaler == 1.0f`, the
    // post-multiplication path is disabled and the existing reduce-only flow runs unchanged.
    float post_mul_scaler{1.0f};
    // Dense row-major path for **mean only** (generic_reductions dispatches AVG over W/H): host enables only when
    // constraints match tilized mean (4D, BF16/FLOAT32, interleaved I/O); AVG is lowered to SUM + scaler before
    // launch. Other ROW_MAJOR reductions tilize and use the standard tile kernels. Exactly one of the two flags
    // may be set at a time (validated in validate_on_program_cache_miss).
    bool row_major_w_dense_path{false};
    bool row_major_h_dense_path{false};
    // Accurate fp32 mean: route Float32 SUM through the SFPU (full fp32); set from
    // ttnn.mean(fast_and_approximate_mode=False).
    bool use_sfpu_reduce{false};
    // H-axis split (RM-H dense path only). 1 = normal H-reduce (output H=1). When >1 the reduce
    // is segmented into `num_h_slices` contiguous H ranges, each reduced independently, producing a
    // (N, C, num_h_slices, W) partial tensor that a second H-reduce collapses to (N, C, 1, W).
    // Lets the op use NC*Wt*num_h_slices cores instead of NC*Wt on tall-H shapes. See reduce_op.cpp.
    uint32_t num_h_slices{1};
};

}  // namespace ttnn::prim
