// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"

namespace ttnn::operations::onboarding {

struct SignBuggy {
    static ttnn::Tensor invoke(const ttnn::Tensor& input);
};

}  // namespace ttnn::operations::onboarding

namespace ttnn {
constexpr auto e05_sign = ttnn::register_operation<"ttnn::e05_sign", ttnn::operations::onboarding::SignBuggy>();
}  // namespace ttnn
