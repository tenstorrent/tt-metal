
// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "ttnn/decorators.hpp"
namespace ttnn::operations::moreh::moreh_dot {

struct MorehDot {

    static Tensor invoke(
        const ttnn::Tensor& input_tensor_a,
        const ttnn::Tensor& input_tensor_b,
        const DataType output_dtype,
        const std::optional<MemoryConfig> &output_mem_config);
};
}

namespace ttnn {
constexpr auto moreh_dot =
    ttnn::register_operation_with_auto_launch_op<"ttnn::moreh_dot", ttnn::operations::moreh::moreh_dot::MorehDot>();
}
