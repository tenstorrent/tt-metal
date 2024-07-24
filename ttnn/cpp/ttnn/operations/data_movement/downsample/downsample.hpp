// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/types.hpp"
#include "ttnn/operations/core/core.hpp"

#include "ttnn/run_operation.hpp"

#include "device/downsample_op.hpp"


namespace ttnn {
namespace operations {
namespace data_movement {

struct ExecuteDownsample {
    static Tensor execute_on_worker_thread(
        const Tensor& input_tensor_a, std::array<uint32_t, 5> downsample_params, std::optional<DataType> dtype) {
        auto dtype_ = dtype.has_value() ? dtype.value():input_tensor_a.get_dtype();
        auto output_tensor = operation::run(
            Downsample{downsample_params, dtype_},
            {input_tensor_a}).front();
        return output_tensor;
        // return ttnn::operations::data_movement::downsample(input_tensor_a, downsample_params, dtype);
    }
};
} // data_movement
} // operations
constexpr auto downsample = ttnn::register_operation<"ttnn::downsample", ttnn::operations::data_movement::ExecuteDownsample>();
} // data_movement
