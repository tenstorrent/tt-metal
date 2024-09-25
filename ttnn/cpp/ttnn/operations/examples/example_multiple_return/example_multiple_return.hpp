
// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "device/example_multiple_return_device_operation.hpp"

namespace ttnn::operations::examples {

// A composite operation is an operation that calls multiple operations in sequence
// It is written using invoke and can be used to call multiple primitive and/or composite operations
struct CompositeExampleMutipleReturnOperation {
    // The user will be able to call this method as `Tensor output = ttnn::composite_example(input_tensor)` after the op
    // is registered
    static std::vector<std::optional<Tensor>> invoke(const Tensor& input_tensor, bool return_output1, bool return_output2);

    static std::vector<Tensor> create_async_output_tensors(
        const std::vector<Tensor>& input_tensors, const std::vector<std::optional<const Tensor>>& optional_inputs);

    // The parameters of this function must be identical to those of invoke.
    static std::vector<bool> create_async_return_flag(
        const Tensor& input_tensor, bool return_output1, bool return_output2
    );
};

}  // namespace ttnn::operations::examples

namespace ttnn {
constexpr auto composite_example_multiple_return = ttnn::register_operation_with_auto_launch_op<"ttnn::composite_example_multiple_return",operations::examples::CompositeExampleMutipleReturnOperation>();
}  // namespace ttnn
