// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Exercise: Define the matmul_add operation interface

#pragma once

#include "ttnn/decorators.hpp"

namespace ttnn::operations::onboarding::exercise {

struct MatmulAdd {
    // TODO: Define invoke()
    static ttnn::Tensor invoke(const ttnn::Tensor& a, const ttnn::Tensor& b, const ttnn::Tensor& c);
};

}  // namespace ttnn::operations::onboarding::exercise

namespace ttnn {

// TODO: Register the operation
constexpr auto e04_matmul_add =
    ttnn::register_operation<"ttnn::e04_matmul_add", ttnn::operations::onboarding::exercise::MatmulAdd>();

}  // namespace ttnn
