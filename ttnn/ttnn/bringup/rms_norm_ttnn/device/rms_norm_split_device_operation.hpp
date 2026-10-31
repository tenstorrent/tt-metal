// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// ttnn.bringup.rms_norm's split program (CHANGELOG 6): the opt-in options `stats_out`, `partial_stats`,
// `input_window` and `dtype`.  A separate device operation, so the default program (rms_norm_ttnn_*) is untouched:
// the host entry routes here only when one of these options is set.
//
//   stats_out=True       -> [.., R, 32 B] fp32 TILE: every element is a partial sum of x^2 over the window's
//                           columns (B column blocks, chosen by the op); the row's sum of squares is the sum of
//                           ALL its elements.  An all_gather of it over the chips that hold the other columns is
//                           what `partial_stats` takes.
//   partial_stats=s      -> y = x * rsqrt(sum of every element of s's row / stats_width + eps) [* weight]
//   (neither)            -> the statistic over the window itself (a local norm of the window)
//   input_window=(c0,c1) -> only the input's columns [c0, c1) (tile-aligned) are read and normalized
//   dtype                -> the output's dtype (default the input's)

#pragma once

#include <cstdint>
#include <optional>
#include <variant>

#include <tt-metalium/program_descriptors.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::bringup::rms_norm_ttnn {

struct RmsNormSplitParams {
    float epsilon = 1e-12f;
    uint32_t stats_width = 0;  // the full row width the mean divides by
    uint32_t col_start = 0;    // input column window [col_start, col_end)
    uint32_t col_end = 0;
    bool stats_out = false;
    tt::tt_metal::DataType out_dtype = tt::tt_metal::DataType::BFLOAT16;
    tt::tt_metal::MemoryConfig output_mem_config;
    tt::tt_metal::MathFidelity math_fidelity = tt::tt_metal::MathFidelity::HiFi4;
    bool fp32_dest_acc_en = true;
    bool math_approx_mode = false;
    bool dst_full_sync_en = false;
    // Fork option max_grid (deepseek_v41_flash_d_p FIX.15, agent rule 18: at most 110 Tensix cores an op on a Galaxy):
    // the op runs on the rectangle (0, 0) .. (max_grid.x - 1, max_grid.y - 1) of the device's compute grid. Default
    // (none) = the whole grid. Hashed.
    std::optional<tt::tt_metal::CoreCoord> max_grid = std::nullopt;
};

struct RmsNormSplitInputs {
    Tensor input;
    std::optional<Tensor> weight;
    std::optional<Tensor> partial_stats;
    std::optional<Tensor> output;  // `output_tensor` (P.16, default absent = the op allocates)
};

struct RmsNormSplitProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const RmsNormSplitParams& attrs, const RmsNormSplitInputs& tensor_args, Tensor& output);
};

struct RmsNormSplitDeviceOperation {
    using operation_attributes_t = RmsNormSplitParams;
    using tensor_args_t = RmsNormSplitInputs;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<RmsNormSplitProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

// The split program's work split: B column blocks of WB tiles per row (B divides the window's tile count).
struct SplitPlan {
    uint32_t Ht = 0;     // row tiles (all leading dims folded)
    uint32_t Wt_in = 0;  // input row, tiles
    uint32_t c0t = 0;    // window start, tiles
    uint32_t Wt_win = 0;
    uint32_t B = 1;
    uint32_t WB = 0;
    uint32_t BLK = 1;
    uint32_t G = 0;  // partial_stats tiles per row
};
SplitPlan split_plan(const RmsNormSplitParams& attrs, const RmsNormSplitInputs& tensor_args, uint32_t num_cores);

}  // namespace ttnn::operations::bringup::rms_norm_ttnn

namespace ttnn::prim::bringup {
ttnn::Tensor rms_norm_split(
    const ttnn::Tensor& input,
    const std::optional<ttnn::Tensor>& weight,
    const std::optional<ttnn::Tensor>& partial_stats,
    const ttnn::operations::bringup::rms_norm_ttnn::RmsNormSplitParams& attrs,
    const std::optional<ttnn::Tensor>& output = std::nullopt);
}  // namespace ttnn::prim::bringup
