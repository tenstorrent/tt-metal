// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Exercise: Implement the operation

#include "eltwise_add.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"

namespace ttnn::operations::onboarding::exercise {

Tensor EltwiseAdd::invoke(const ttnn::Tensor& a, const ttnn::Tensor& b) {
    // Call the existing ttnn::add operation
    return ttnn::add(a, b);
}

}  // namespace ttnn::operations::onboarding::exercise
