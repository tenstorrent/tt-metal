// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Exercise: Define the operation interface

#pragma once

#include "ttnn/decorators.hpp"

namespace ttnn::operations::onboarding::exercise {

struct EltwiseAdd {
    // TODO: Define invoke()
    static ttnn::Tensor invoke(const ttnn::Tensor& a, const ttnn::Tensor& b);
};

}  // namespace ttnn::operations::onboarding::exercise

namespace ttnn {

// TODO: Register the operation
constexpr auto e03_eltwise_add =
    ttnn::register_operation<"ttnn::e03_eltwise_add", ttnn::operations::onboarding::exercise::EltwiseAdd>();

}  // namespace ttnn
