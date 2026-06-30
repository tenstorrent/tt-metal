// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::per_token_cast_back {

ttnn::Tensor per_token_cast_back(
    const ttnn::Tensor& input_e4m3,
    const ttnn::Tensor& input_scale,
    const std::optional<tt::tt_metal::DataType>& output_dtype = std::nullopt,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
    // Optional "masked decompress" mode. When the three routing tensors are supplied, input_e4m3 is
    // treated as a per-device dispatch buffer: only the rows covered by this device's expert regions
    // (defined by expert_token_counts / expert_region_offsets) are decompressed, and each valid row's
    // per-128-block fp32 scales are read from the metadata tail (fields 5..). input_scale is unused in
    // this mode. All three tensors plus experts_per_chip / dispatch_group_size are required together,
    // or none of them.
    const std::optional<ttnn::Tensor>& expert_token_counts = std::nullopt,
    const std::optional<ttnn::Tensor>& expert_region_offsets = std::nullopt,
    const std::optional<ttnn::Tensor>& metadata = std::nullopt,
    std::optional<uint32_t> experts_per_chip = std::nullopt,
    std::optional<uint32_t> dispatch_group_size = std::nullopt);

}  // namespace ttnn::operations::experimental::deepseek_prefill::per_token_cast_back

namespace ttnn {
using operations::experimental::deepseek_prefill::per_token_cast_back::per_token_cast_back;
}  // namespace ttnn
