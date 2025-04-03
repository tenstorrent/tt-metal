// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>
#include <array>
#include <optional>

#include "ttnn/decorators.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace tt {
namespace tt_metal {
enum class DataType;
}  // namespace tt_metal
}  // namespace tt

namespace ttnn {
namespace operations {
namespace downsample {

struct ExecuteDownsample {
    static Tensor invoke(
        const Tensor& input_tensor_a, std::array<uint32_t, 5> downsample_params, std::optional<DataType> dtype);
};
}  // namespace downsample
}  // namespace operations

constexpr auto downsample =
    ttnn::register_operation_with_auto_launch_op<"ttnn::downsample", ttnn::operations::downsample::ExecuteDownsample>();
}  // namespace ttnn
