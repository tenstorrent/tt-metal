// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/run_operation.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/operations/core/core.hpp"

namespace ttnn {
namespace operations::experimental {

struct Argmax {
  static Tensor _argmax(const Tensor& input_t, int64_t _dim, bool all, const std::optional<MemoryConfig>& output_mem_config = std::nullopt);
};

struct ExecuteArgMax {
    static Tensor operator()(
        const Tensor& input_tensor,
        int64_t dim,
        bool all,
        const std::optional<MemoryConfig>& memory_config = std::nullopt) {
        return Argmax::_argmax(input_tensor, dim, all, memory_config);
    }
};

}  // namespace operations::experimental

namespace experimental {

constexpr auto argmax = ttnn::register_operation_with_auto_launch_op<
  "ttnn::experimental::argmax",
  ttnn::operations::experimental::ExecuteArgMax>();

}
}  // namespace ttnn
