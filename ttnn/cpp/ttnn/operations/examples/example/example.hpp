
// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "device/example_device_operation.hpp"

namespace ttnn::operations::examples {

// This is the main operation that will be called by the user
struct ExampleOperation {
    // This is the main function that will be called by the user
    static Tensor execute_on_main_thread(uint8_t queue_id, const Tensor &input_tensor) {
        return ttnn::device_operation::run<ExampleDeviceOperation>(
            queue_id,
            ExampleDeviceOperation::operation_attributes_t{.attribute = true, .some_other_attribute = 42},
            ExampleDeviceOperation::tensor_args_t{input_tensor});
    }

    // This is the main function that will be called by the user
    static Tensor execute_on_main_thread(const Tensor &input_tensor) { return execute_on_main_thread(0, input_tensor); }
};

}  // namespace ttnn::operations::examples

namespace ttnn {

// Register the operation. The name, in this case, "ttnn::example" should match the namespace of the operation
// And the name will be directly mapped to python, where it will become "ttnn.example"
constexpr auto example = ttnn::register_operation<operations::examples::ExampleOperation>("ttnn::example");

}  // namespace ttnn
