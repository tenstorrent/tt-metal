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
ttnn::Tensor fast_reduce_nc(
    const ttnn::Tensor& input,
    ttsl::Span<const int32_t> dims,
    const std::optional<const Tensor>& output,
    const ttnn::MemoryConfig& memory_config,
    std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config,
    const std::optional<tt::tt_metal::CoreRangeSet>& sub_core_grids = std::nullopt,
    const std::optional<tt::tt_metal::DataType>& output_dtype = std::nullopt,
    bool fp32_intermediate_stages = false);

}  // namespace ttnn::experimental::reduction
