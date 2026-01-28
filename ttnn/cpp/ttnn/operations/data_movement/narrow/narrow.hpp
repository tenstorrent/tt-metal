// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"

namespace ttnn {
namespace operations::data_movement {

struct NarrowOperation {
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor, const int32_t narrow_dim, const int32_t narrow_start, const uint32_t length);
};

}  // namespace operations::data_movement
constexpr auto narrow = ttnn::register_operation<"ttnn::narrow", ttnn::operations::data_movement::NarrowOperation>();
}  // namespace ttnn
