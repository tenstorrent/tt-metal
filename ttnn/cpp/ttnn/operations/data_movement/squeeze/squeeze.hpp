// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"


namespace ttnn {
namespace operations::data_movement {

struct SqueezeOperation {
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        const int dim
        );

};


}  // namespace operations::data_movement

constexpr auto squeeze = ttnn::register_operation<"ttnn::squeeze", ttnn::operations::data_movement::SqueezeOperation>();

}  // namespace ttnn
