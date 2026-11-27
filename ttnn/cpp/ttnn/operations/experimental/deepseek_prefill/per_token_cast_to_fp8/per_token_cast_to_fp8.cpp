// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "per_token_cast_to_fp8.hpp"

#include "device/per_token_cast_to_fp8_device_operation.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::per_token_cast_to_fp8 {

std::tuple<ttnn::Tensor, ttnn::Tensor> per_token_cast_to_fp8(
    const ttnn::Tensor& input_tensor,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    bool round_scale_to_power_of_two) {
    // The outputs are DRAM only: an L1 input defaults them to DRAM rather than to its own memory config.
    const auto output_memory_config = memory_config.value_or(
        input_tensor.memory_config().is_dram() ? input_tensor.memory_config() : ttnn::DRAM_MEMORY_CONFIG);
    return ttnn::prim::per_token_cast_to_fp8(input_tensor, output_memory_config, round_scale_to_power_of_two);
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::per_token_cast_to_fp8
