// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <optional>

#include "dispatch_tilize.hpp"
#include "device/dispatch_tilize_device_operation.hpp"

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::dispatch_tilize {

ttnn::Tensor dispatch_tilize(
    const ttnn::Tensor& input_tensor,
    const std::optional<ttnn::Tensor>& expert_region_offsets,
    const std::optional<ttnn::Tensor>& total_counts_per_expert,
    const std::optional<tt::tt_metal::DataType>& output_dtype,
    uint32_t experts_per_chip,
    const std::optional<tt::tt_metal::MemoryConfig>& output_memory_config) {
    const auto out_dtype = output_dtype.value_or(input_tensor.dtype());
    const auto memory_config = output_memory_config.value_or(input_tensor.memory_config());

    return ttnn::prim::dispatch_tilize(
        input_tensor, expert_region_offsets, total_counts_per_expert, out_dtype, experts_per_chip, memory_config);
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::dispatch_tilize
