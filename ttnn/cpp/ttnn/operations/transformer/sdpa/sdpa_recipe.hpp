// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "sdpa_precision_policy.hpp"
#include "ttnn/operations/transformer/sdpa_config.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::transformer::sdpa::detail {

Tensor run_recipe(
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const PrecisionPolicy& policy,
    const std::optional<SDPAProgramConfig>& program_config);

}  // namespace ttnn::operations::transformer::sdpa::detail
