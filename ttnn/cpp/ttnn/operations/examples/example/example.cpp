// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "example.hpp"
#include "ttnn/graph/composite_trace.hpp"

namespace ttnn {

// A composite operation is an operation that calls multiple operations in sequence
// It is written using invoke and can be used to call multiple primitive and/or composite operations
Tensor composite_example(const Tensor& input_tensor) {
    ttnn::graph::ScopedCompositeTrace _trace("ttnn::composite_example");
    auto copy = prim::example(input_tensor);
    auto another_copy = prim::example(copy);
    return another_copy;
}

}  // namespace ttnn
