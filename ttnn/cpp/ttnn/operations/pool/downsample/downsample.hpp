// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "device/downsample_op.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/run_operation.hpp"

namespace ttnn {
namespace operations {
namespace downsample {

struct ExecuteDownsample {
    static Tensor operator()(
        const Tensor& input_tensor_a, std::array<uint32_t, 5> downsample_params, std::optional<DataType> dtype) {
        auto dtype_ = dtype.has_value() ? dtype.value() : input_tensor_a.get_dtype();
        auto output_tensor = operation::run(Downsample{downsample_params, dtype_}, {input_tensor_a}).front();
        return output_tensor;
        // return ttnn::operations::data_movement::downsample(input_tensor_a, downsample_params, dtype);
    }
};
}  // namespace downsample
}  // namespace operations

constexpr auto downsample =
    ttnn::register_operation_with_auto_launch_op<"ttnn::downsample", ttnn::operations::downsample::ExecuteDownsample>();
}  // namespace ttnn
