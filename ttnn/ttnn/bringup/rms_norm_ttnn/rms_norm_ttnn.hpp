// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <utility>
#include <variant>

#include <tt-metalium/program_descriptors.hpp>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::bringup::rms_norm_ttnn {

// The caller's `program_config`, as read off whatever object was passed (rms_norm_ttnn.py tells the
// two variants apart by which FIELDS are present, never by type).  The binding fills it with
// getattr(); every field is the raw value, checked on the host (rms_norm_ttnn.cpp).
struct ProgramConfigArg {
    bool use_welford = false;
    bool sharded_variant = false;                     // hasattr(compute_with_storage_grid_size)
    std::optional<std::pair<int64_t, int64_t>> grid;  // compute_with_storage_grid_size, if not None
    std::optional<int64_t> block_h;                   // absent -> the shard's own
    std::optional<int64_t> block_w;
    int64_t subblock_w = 0;  // int(getattr(pc, "subblock_w", 0) or 0)
    bool inplace = false;
};

// Either compute-config object type (A6): passed through / copied field-for-field.
using ComputeConfigArg = std::variant<tt::tt_metal::ComputeConfigDescriptor, ttnn::DeviceComputeKernelConfig>;

// A7: HiFi4 math, approximate SFPU, 16-bit DEST.
tt::tt_metal::ComputeConfigDescriptor default_compute_kernel_config();
tt::tt_metal::ComputeConfigDescriptor normalize_compute_kernel_config(const std::optional<ComputeConfigArg>& cfg);

// Everything one call returns: the normalized output, whether program_config.inplace made it the
// input tensor itself, and -- only under `return_residual_sum` -- the residual sum t = x + r.
struct RmsNormResult {
    ttnn::Tensor output;
    bool inplace = false;
    std::optional<ttnn::Tensor> residual_sum;
};

// The op with every option.  `return_residual_sum` (default off) also returns t = x + r, exactly as the
// op computes it before the statistics, at the input's dtype and layout, placed at
// `residual_sum_memory_config` (default: the input's memory config).  It needs a residual_input_tensor and
// a TILE input.  Off, the call is the one it was before the option existed: same refusals, same program,
// no second tensor.
RmsNormResult rms_norm_full(
    const ttnn::Tensor& input_tensor,
    double epsilon,
    const std::optional<const ttnn::Tensor>& weight,
    const std::optional<const ttnn::Tensor>& bias,
    const std::optional<const ttnn::Tensor>& residual_input_tensor,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<ProgramConfigArg>& program_config,
    const std::optional<ComputeConfigArg>& compute_kernel_config,
    bool return_residual_sum,
    const std::optional<ttnn::MemoryConfig>& residual_sum_memory_config);

// RMSNorm over the last dimension (rms_norm_ttnn.py's rms_norm_ttnn()).  The second member is true
// when program_config.inplace made the output the input tensor itself.
std::pair<ttnn::Tensor, bool> rms_norm_with_inplace(
    const ttnn::Tensor& input_tensor,
    double epsilon,
    const std::optional<const ttnn::Tensor>& weight,
    const std::optional<const ttnn::Tensor>& bias,
    const std::optional<const ttnn::Tensor>& residual_input_tensor,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<ProgramConfigArg>& program_config,
    const std::optional<ComputeConfigArg>& compute_kernel_config);

ttnn::Tensor rms_norm(
    const ttnn::Tensor& input_tensor,
    double epsilon = 1e-12,
    const std::optional<const ttnn::Tensor>& weight = std::nullopt,
    const std::optional<const ttnn::Tensor>& bias = std::nullopt,
    const std::optional<const ttnn::Tensor>& residual_input_tensor = std::nullopt,
    const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<ProgramConfigArg>& program_config = std::nullopt,
    const std::optional<ComputeConfigArg>& compute_kernel_config = std::nullopt);

// The fused residual form: (y, t) with t = input_tensor + residual_input_tensor, i.e. rms_norm_full with
// return_residual_sum on.  For a transformer's residual stream, where t is the next residual add's input.
std::pair<ttnn::Tensor, ttnn::Tensor> rms_norm_with_residual_sum(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& residual_input_tensor,
    double epsilon = 1e-12,
    const std::optional<const ttnn::Tensor>& weight = std::nullopt,
    const std::optional<const ttnn::Tensor>& bias = std::nullopt,
    const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<ttnn::MemoryConfig>& residual_sum_memory_config = std::nullopt,
    const std::optional<ProgramConfigArg>& program_config = std::nullopt,
    const std::optional<ComputeConfigArg>& compute_kernel_config = std::nullopt);

}  // namespace ttnn::operations::bringup::rms_norm_ttnn

namespace ttnn::bringup {
using ::ttnn::operations::bringup::rms_norm_ttnn::rms_norm;
}  // namespace ttnn::bringup
