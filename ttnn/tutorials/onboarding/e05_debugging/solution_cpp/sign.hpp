// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"

namespace ttnn::operations::onboarding {

struct Sign {
    static ttnn::Tensor invoke(const ttnn::Tensor& input);
};

}  // namespace ttnn::operations::onboarding

namespace ttnn {
constexpr auto s05_sign = ttnn::register_operation<"ttnn::s05_sign", ttnn::operations::onboarding::Sign>();
}  // namespace ttnn
