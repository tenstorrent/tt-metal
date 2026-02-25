// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "sign.hpp"
#include "device/sign_device_operation.hpp"

namespace ttnn::operations::onboarding {

Tensor SignBuggy::invoke(const Tensor& input) { return ttnn::prim::onboarding_sign_buggy(input); }

}  // namespace ttnn::operations::onboarding
