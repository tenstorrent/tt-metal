// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"

namespace ttnn {
namespace operations::transformer {

struct RotateHalfOperation {
    static Tensor operator()(const Tensor& input_tensor, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);
};

}  // namespace operations::transformer

namespace transformer {
constexpr auto rotate_half = ttnn::register_operation_with_auto_launch_op<
    "ttnn::transformer::rotate_half",
    ttnn::operations::transformer::RotateHalfOperation>();

} // namespace transformer
} // namespace ttnn
