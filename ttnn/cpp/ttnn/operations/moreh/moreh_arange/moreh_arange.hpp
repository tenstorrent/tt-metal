// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"

namespace ttnn::operations::moreh::moreh_arange {
struct MorehArange {
    static Tensor invoke(
        float start,
        float end,
        float step,
        const Tensor& any,
        const std::optional<Tensor>& output_tensor,
        bool untilize_out,
        const std::optional<DataType>& output_dtype,
        const std::optional<MemoryConfig>& output_memory_config);
};
}  // namespace ttnn::operations::moreh::moreh_arange

namespace ttnn {
constexpr auto moreh_arange = ttnn::
    register_operation_with_auto_launch_op<"ttnn::moreh_arange", ttnn::operations::moreh::moreh_arange::MorehArange>();
}
