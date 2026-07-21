// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "per_token_cast_back.hpp"
#include "device/per_token_cast_back_device_operation.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::per_token_cast_back {

ttnn::Tensor per_token_cast_back(
    const ttnn::Tensor& input_e4m3,
    const ttnn::Tensor& input_scale,
    const std::optional<tt::tt_metal::DataType>& output_dtype,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    bool compute_is_bf16) {
    return ttnn::prim::per_token_cast_back(
        input_e4m3,
        input_scale,
        output_dtype.value_or(tt::tt_metal::DataType::BFLOAT16),
        memory_config.value_or(input_e4m3.memory_config()),
        compute_is_bf16);
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::per_token_cast_back
