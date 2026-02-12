// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Exercise: Implement the matmul_add operation

#include "matmul_add.hpp"

namespace ttnn::operations::onboarding::exercise {

Tensor MatmulAdd::invoke(const ttnn::Tensor& a, const ttnn::Tensor& b, const ttnn::Tensor& c) {
    (void)a;
    (void)b;
    (void)c;
    // TODO: Implement
    TT_THROW("Not implemented");
}

}  // namespace ttnn::operations::onboarding::exercise
