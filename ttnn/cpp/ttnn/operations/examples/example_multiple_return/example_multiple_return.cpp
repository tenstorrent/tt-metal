
// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "example_multiple_return.hpp"

namespace ttnn::operations::examples {

std::vector<std::optional<Tensor>> CompositeExampleMutipleReturnOperation::invoke(const Tensor& input_tensor) {
    return prim::example_multiple_return(input_tensor);
}

std::vector<Tensor> CompositeExampleMutipleReturnOperation::create_async_output_tensors(
    const std::vector<Tensor>& input_tensors, const std::vector<std::optional<const Tensor>>& optional_inputs) {
    const auto& input_tensor = input_tensors.at(0);
    return {
        Tensor(operation::get_workers_for_op_output({input_tensor})),
        Tensor(operation::get_workers_for_op_output({input_tensor}))};
}

}  // namespace ttnn::operations::examples
