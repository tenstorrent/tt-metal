// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"

namespace ttnn::operations::onboarding {

struct MatmulAdd {
    static ttnn::Tensor invoke(const ttnn::Tensor& a, const ttnn::Tensor& b, const ttnn::Tensor& c);
};

}  // namespace ttnn::operations::onboarding

namespace ttnn {
constexpr auto s04_matmul_add =
    ttnn::register_operation<"ttnn::s04_matmul_add", ttnn::operations::onboarding::MatmulAdd>();
}  // namespace ttnn
