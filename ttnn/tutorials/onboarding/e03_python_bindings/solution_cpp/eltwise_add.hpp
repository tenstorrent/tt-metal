// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"

namespace ttnn::operations::onboarding {

struct EltwiseAdd {
    static ttnn::Tensor invoke(const ttnn::Tensor& a, const ttnn::Tensor& b);
};

}  // namespace ttnn::operations::onboarding

namespace ttnn {
constexpr auto s03_eltwise_add =
    ttnn::register_operation<"ttnn::s03_eltwise_add", ttnn::operations::onboarding::EltwiseAdd>();
}  // namespace ttnn
