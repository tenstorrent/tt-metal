// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Exercise: Implement the operation

#include "eltwise_add.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"

namespace ttnn::operations::onboarding::exercise {

Tensor EltwiseAdd::invoke(const ttnn::Tensor& a, const ttnn::Tensor& b) {
    (void)a;
    (void)b;
    // TODO: Implement
    TT_THROW("Not implemented");
}

}  // namespace ttnn::operations::onboarding::exercise
