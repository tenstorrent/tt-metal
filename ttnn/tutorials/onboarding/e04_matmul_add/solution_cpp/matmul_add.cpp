// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "matmul_add.hpp"
#include "device/matmul_add_device_operation.hpp"

namespace ttnn::operations::onboarding {

Tensor MatmulAdd::invoke(const ttnn::Tensor& a, const ttnn::Tensor& b, const ttnn::Tensor& c) {
    return ttnn::prim::onboarding_matmul_add(a, b, c);
}

}  // namespace ttnn::operations::onboarding
