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
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt);

}  // namespace ttnn::operations::experimental::deepseek_prefill::per_token_cast_back

namespace ttnn {
using operations::experimental::deepseek_prefill::per_token_cast_back::per_token_cast_back;
}  // namespace ttnn
