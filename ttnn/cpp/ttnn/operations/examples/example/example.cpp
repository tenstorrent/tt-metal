
// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "example.hpp"

namespace ttnn {

// A composite operation is an operation that calls multiple operations in sequence
// It is written using invoke and can be used to call multiple primitive and/or composite operations
Tensor composite_example(const Tensor& input_tensor) {
    auto copy = prim::example(input_tensor);
    auto another_copy = prim::example(copy);
    return another_copy;
}

}  // namespace ttnn
