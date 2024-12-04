
// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "example_multiple_return.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::examples {

std::vector<std::optional<Tensor>> CompositeExampleMutipleReturnOperation::invoke(
    const Tensor& input_tensor, bool return_output1, bool return_output2) {
    return prim::example_multiple_return(input_tensor, return_output1, return_output2);
}

OptionalTensors CompositeExampleMutipleReturnOperation::create_async_optional_output_tensors(
    const Tensor& input_tensor, bool return_output1, bool return_output2) {
    return {
        return_output1 ? std::optional<Tensor>(operation::get_workers_for_op_output({input_tensor})) : std::nullopt,
        return_output2 ? std::optional<Tensor>(operation::get_workers_for_op_output({input_tensor})) : std::nullopt};
}

}  // namespace ttnn::operations::examples
