// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"

namespace ttnn {
namespace operations::experimental::transformer {

struct RotateHalfOperation {
    static Tensor operator()(const Tensor& input_tensor, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);
};

}  // namespace operations::experimental::transformer

namespace experimental {
constexpr auto rotate_half = ttnn::register_operation_with_auto_launch_op<
    "ttnn::experimental::rotate_half",
    ttnn::operations::experimental::transformer::RotateHalfOperation>();

} // namespace experimental
} // namespace ttnn
