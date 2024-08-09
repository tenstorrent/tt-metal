
// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "device/example_device_operation.hpp"

namespace ttnn::operations::examples {

struct ExampleOperation {
    // Map API arguments to the device operation
    using device_operation_t = ExampleDeviceOperation;

    static std::tuple<ExampleDeviceOperation::operation_attributes_t, ExampleDeviceOperation::tensor_args_t> map_args_to_device_operation(const Tensor& input_tensor) {
        return {
            ExampleDeviceOperation::operation_attributes_t{true, 42},
            ExampleDeviceOperation::tensor_args_t{input_tensor}
        };
    }
};

}  // namespace ttnn::operations::examples

namespace ttnn {

// Register the operation. The name, in this case, "ttnn::example" should match the namespace of the operation
// And the name will be directly mapped to python, where it will become "ttnn.example"
constexpr auto example = ttnn::register_operation<"ttnn::example", operations::examples::ExampleOperation>();

// Alternatively, the operation can be registered as asynchronous
// constexpr auto example = ttnn::register_operation_with_auto_launch_op<"ttnn::example", operations::examples::ExampleOperation>();

}  // namespace ttnn


namespace ttnn::operations::examples {

// A composite operation is an operation that calls multiple operations in sequence
// It is written using operator() and can be used to call multiple primitive and/or composite operations
struct CompositeExampleOperation {
    // Map API arguments to the device operation

    static Tensor operator()(const Tensor& input_tensor) {
        auto copy = example(input_tensor);
        auto another_copy = example(copy);
        return another_copy;
    }
};

}  // namespace ttnn::operations::examples

namespace ttnn {
constexpr auto composite_example = ttnn::register_operation<"ttnn::composite_example", operations::examples::CompositeExampleOperation>();
}  // namespace ttnn
