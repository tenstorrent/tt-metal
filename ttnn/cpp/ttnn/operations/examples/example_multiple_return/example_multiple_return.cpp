
// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "example_multiple_return.hpp"

using namespace tt::tt_metal;

namespace ttnn {

std::vector<std::optional<Tensor>> composite_example_multiple_return(
    const Tensor& input_tensor, bool return_output1, bool return_output2) {
    return prim::example_multiple_return(input_tensor, return_output1, return_output2);
}

}  // namespace ttnn
