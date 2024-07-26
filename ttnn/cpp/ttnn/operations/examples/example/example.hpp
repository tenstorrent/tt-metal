
// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "device/example_device_operation.hpp"

namespace ttnn::operations::examples {

// This is the main operation that will be called by the user
struct ExampleOperation {
    // This how the user can call the operation
    static Tensor operator()(uint8_t queue_id, const Tensor &input_tensor) {
        return ttnn::device_operation::run<ExampleDeviceOperation>(
            queue_id,
            ExampleDeviceOperation::operation_attributes_t{.attribute = true, .some_other_attribute = 42},
            ExampleDeviceOperation::tensor_args_t{input_tensor});
    }

    // This how the user can call the operation
    static Tensor operator()(const Tensor &input_tensor) { return operator()(0, input_tensor); }

    // operator() can be overloaded as many times as needed to provide all desired APIs
};

}  // namespace ttnn::operations::examples

namespace ttnn {

// Register the operation. The name, in this case, "ttnn::example" should match the namespace of the operation
// And the name will be directly mapped to python, where it will become "ttnn.example"
constexpr auto example = ttnn::register_operation<"ttnn::example", operations::examples::ExampleOperation>();

// Alternatively, the operation can be registered as asynchronous
// constexpr auto example = ttnn::register_operation_with_auto_launch_op<"ttnn::example", operations::examples::ExampleOperation>();

}  // namespace ttnn
