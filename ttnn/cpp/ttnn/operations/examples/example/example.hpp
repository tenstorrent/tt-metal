
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

    // operator() can be overloaded to take any number of arguments

    // operator() doesn't imply anything about async or sync execution and the user needs to be aware of
    // that

    // If the user wants to make the operation async automatically, then `execute_on_worker_thread` should be used
    // instead of `operator()`
};

}  // namespace ttnn::operations::examples

namespace ttnn {

// Register the operation. The name, in this case, "ttnn::example" should match the namespace of the operation
// And the name will be directly mapped to python, where it will become "ttnn.example"
constexpr auto example = ttnn::register_operation<"ttnn::example", operations::examples::ExampleOperation>();

}  // namespace ttnn
