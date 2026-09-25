// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/transformer/sdpa/sdpa_numerics.hpp"

#include <tt_stl/assert.hpp>

namespace ttnn::operations::transformer::sdpa::detail {

ResolvedNumerics resolve_numerics(
    tt::ARCH arch,
    const std::optional<RecipeSelection>& recipe,
    const std::optional<DeviceComputeKernelConfig>& compute,
    std::optional<bool> exp_approx_mode) {
    if (!recipe.has_value()) {
        return {
            init_device_compute_kernel_config(arch, compute, tt::tt_metal::MathFidelity::HiFi2, true, false, false),
            exp_approx_mode.value_or(true),
            std::nullopt};
    }
    if (arch != tt::ARCH::BLACKHOLE) {
        TT_THROW("Explicit SDPA precision recipes are qualified only on Blackhole");
    }
    // ComputeKernelConfig has no per-field presence information. Reject the
    // combination rather than silently discarding an explicit caller request,
    // including one hidden behind an empty constructor's LoFi default.
    if (compute.has_value()) {
        TT_THROW("Specify either an SDPA precision recipe or a compute kernel config, not both");
    }
    if (exp_approx_mode.has_value() && !exp_approx_mode.value()) {
        TT_THROW("exp_approx_mode=false conflicts with the selected SDPA precision recipe");
    }
    const auto policy = resolve_precision_policy(recipe.value());
    DeviceComputeKernelConfig effective{
        .math_fidelity = policy.pv_fidelity,
        .math_approx_mode = true,
        .fp32_dest_acc_en = policy.fp32_destination,
        .packer_l1_acc = false,
        .dst_full_sync_en = false};
    // C's HiFi4 QK must be implemented explicitly by the compute policy; the
    // generic config alone intentionally describes its HiFi2 PV matmul.
    return {effective, true, policy};
}

}  // namespace ttnn::operations::transformer::sdpa::detail
