// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"

namespace ttnn {
namespace operations::experimental::reduction {

struct CumprodOperation {
    static Tensor invoke(
        const Tensor& input_tensor,
        int64_t dim  // TODO(jbbieniek): int8_t?
    );
};

}  // namespace operations::experimental::reduction

namespace experimental {
constexpr auto cumprod = ttnn::register_operation_with_auto_launch_op<
    "ttnn::experimental::cumprod",
    ttnn::operations::experimental::reduction::CumprodOperation>();

}  // namespace experimental
}  // namespace ttnn
