// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/types.hpp"
#include <ranges>
#include "ttnn/decorators.hpp"

namespace ttnn {
namespace operations::data_movement {

struct PadSpecDim {
    uint32_t before_elements;
    uint32_t after_elements;
};

}  // namespace operations::data_movement

// This function signature is similar to pytorch's signature
// Any rank tensor supported
ttnn::Tensor pad(
    const ttnn::Tensor& input_tensor,
    const ttnn::SmallVector<operations::data_movement::PadSpecDim>& padding,
    float value,
    bool use_multicore,
    const std::optional<MemoryConfig>& memory_config_arg = std::nullopt);

ttnn::Tensor pad(
    const ttnn::Tensor& input_tensor,
    const ttnn::SmallVector<std::array<uint32_t, 2>>& padding,
    float value,
    bool use_multicore = false,
    const std::optional<MemoryConfig>& memory_config_arg = std::nullopt);

// legacy API
ttnn::Tensor pad(
    const ttnn::Tensor& input_tensor,
    const tt::tt_metal::Array4D& output_padded_shape,
    const tt::tt_metal::Array4D& input_tensor_start,
    float value,
    bool use_multicore = false,
    const std::optional<MemoryConfig>& memory_config_arg = std::nullopt);

}  // namespace ttnn
