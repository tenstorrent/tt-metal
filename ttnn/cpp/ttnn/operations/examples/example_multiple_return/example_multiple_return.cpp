
// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "example_multiple_return.hpp"
#include "ttnn/graph/composite_trace.hpp"

using namespace tt::tt_metal;

namespace ttnn {

std::vector<std::optional<Tensor>> composite_example_multiple_return(
    const Tensor& input_tensor, bool return_output1, bool return_output2) {
    ttnn::graph::ScopedCompositeTrace _trace("ttnn::composite_example_multiple_return");
    return prim::example_multiple_return(input_tensor, return_output1, return_output2);
}

}  // namespace ttnn
