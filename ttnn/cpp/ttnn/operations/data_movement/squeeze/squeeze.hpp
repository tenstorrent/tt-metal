// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include <tt_stl/small_vector.hpp>

namespace ttnn {
namespace operations::data_movement {

struct SqueezeOperation {
    // Note: dim is passed by non-const reference because it's convenient to modify it for processing
    static ttnn::Tensor invoke(const ttnn::Tensor& input_tensor, const ttnn::SmallVector<int>& dim);
    static ttnn::Tensor invoke(const ttnn::Tensor& input_tensor, int dim);
    static ttnn::Tensor invoke(const ttnn::Tensor& input_tensor);
};

}  // namespace operations::data_movement

constexpr auto squeeze = ttnn::register_operation<"ttnn::squeeze", ttnn::operations::data_movement::SqueezeOperation>();

}  // namespace ttnn
