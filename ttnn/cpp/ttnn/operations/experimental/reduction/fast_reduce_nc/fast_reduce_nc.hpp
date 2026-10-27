// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/core.hpp"
#include <tt-metalium/core_coord.hpp>

namespace ttnn::experimental::reduction {

// `output_dtype` sets the dtype produced by the final prim::fast_reduce_nc stage;
// `fp32_intermediate_stages` keeps non-final stages in FLOAT32 to avoid double-rounding.
// FLOAT32 stages sum on the SFPU at full precision when fp32_dest_acc_en is set, unless
// `fast_and_approximate_mode` selects the faster FPU path (tf32).
ttnn::Tensor fast_reduce_nc(
    const ttnn::Tensor& input,
    ttsl::Span<const int32_t> dims,
    const std::optional<const Tensor>& output,
    const ttnn::MemoryConfig& memory_config,
    std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config,
    const std::optional<tt::tt_metal::CoreRangeSet>& sub_core_grids = std::nullopt,
    const std::optional<tt::tt_metal::DataType>& output_dtype = std::nullopt,
    bool fp32_intermediate_stages = false,
    bool fast_and_approximate_mode = false);

// Single non-spatial reduction with two contiguous, tile-aligned output channel regions.
std::tuple<Tensor, Tensor> fast_reduce_nc_split(
    const Tensor& input,
    int32_t dim,
    uint32_t split_output_width,
    const MemoryConfig& memory_config,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config);

}  // namespace ttnn::experimental::reduction
