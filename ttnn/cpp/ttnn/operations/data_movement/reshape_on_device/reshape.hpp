// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"


namespace ttnn {
namespace operations::data_movement {

struct ReshapeOperation {
    static ttnn::Tensor invoke(
        uint8_t queue_id,
        const ttnn::Tensor& input_tensor,
        int N,
        int C,
        int H,
        int W,
        const std::optional<MemoryConfig>& memory_config_arg);

    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        int N,
        int C,
        int H,
        int W,
        const std::optional<MemoryConfig>& memory_config);

    static ttnn::Tensor invoke(const ttnn::Tensor& input_tensor, int N, int C, int H, int W);
};


}  // namespace operations::data_movement

// TODO: unify with ttnn::reshape in core.cpp
constexpr auto reshape_on_device = ttnn::register_operation<"ttnn::reshape_on_device", ttnn::operations::data_movement::ReshapeOperation>();

}  // namespace ttnn
