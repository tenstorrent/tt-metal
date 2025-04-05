// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/decorators.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace tt {
namespace tt_metal {
enum class DataType;
struct MemoryConfig;
}  // namespace tt_metal
}  // namespace tt

namespace ttnn::operations::moreh::moreh_arange {
struct MorehArange {
    static Tensor invoke(
        float start,
        float end,
        float step,
        const Tensor& any,
        const std::optional<Tensor>& output,
        bool untilize_out,
        const std::optional<DataType>& dtype,
        const std::optional<MemoryConfig>& memory_config);
};
}  // namespace ttnn::operations::moreh::moreh_arange

namespace ttnn {
constexpr auto moreh_arange = ttnn::
    register_operation_with_auto_launch_op<"ttnn::moreh_arange", ttnn::operations::moreh::moreh_arange::MorehArange>();
}
