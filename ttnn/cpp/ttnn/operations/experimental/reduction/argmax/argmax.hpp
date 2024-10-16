// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"

namespace ttnn {
namespace operations::experimental {

namespace reduction {

struct ArgmaxOperation {
    static Tensor invoke(const Tensor& input_tensor,
                         int64_t dim,
                         bool all,
                         const std::optional<MemoryConfig>& memory_config = std::nullopt);
};

struct ArgminOperation {
    static Tensor invoke(const Tensor& input_tensor,
                         int64_t dim,
                         bool all,
                         const std::optional<MemoryConfig>& memory_config = std::nullopt);
};

}  // namespace reduction
}  // namespace operations::experimental

namespace experimental {

constexpr auto argmax =
    ttnn::register_operation_with_auto_launch_op<"ttnn::experimental::argmax",
                                                 ttnn::operations::experimental::reduction::ArgmaxOperation>();

constexpr auto argmin =
    ttnn::register_operation_with_auto_launch_op<"ttnn::experimental::argmin",
                                                 ttnn::operations::experimental::reduction::ArgminOperation>();

}  // namespace experimental
}  // namespace ttnn
