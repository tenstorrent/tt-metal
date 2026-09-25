// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/transformer/sdpa/sdpa_precision_policy.hpp"

namespace ttnn::operations::transformer::sdpa::detail {

struct ResolvedNumerics {
    DeviceComputeKernelConfig compute;
    bool exp_approx_mode;
    std::optional<PrecisionPolicy> policy;
};

// Resolve intent without claiming a kernel implements it. Recipe dispatch and
// prepared-input/shape validation remain mandatory before any recipe launch.
// A missing recipe preserves all existing compute fields and exp defaults.
ResolvedNumerics resolve_numerics(
    tt::ARCH arch,
    const std::optional<RecipeSelection>& recipe,
    const std::optional<DeviceComputeKernelConfig>& compute,
    std::optional<bool> exp_approx_mode);

}  // namespace ttnn::operations::transformer::sdpa::detail
