// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "rmsnorm_pre_all_gather_bw.hpp"

#include <string_view>

#include "rmsnorm_distributed_bw_utils.hpp"
#include "ttnn/operations/reduction/generic/generic_reductions.hpp"

namespace ttnn {

Tensor rms_norm_pre_all_gather_bw(
    const Tensor& input_tensor,
    const Tensor& output_grad,
    const Tensor& stats,
    float epsilon,
    const std::optional<const Tensor>& weight,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config) {
    namespace bw = ttnn::operations::normalization::rmsnorm_distributed_bw;
    constexpr std::string_view op_name = "rms_norm_pre_all_gather_bw";
    bw::validate_bw_inputs(input_tensor, output_grad, weight, op_name);
    bw::validate_stats_tensor(stats, input_tensor, "stats", op_name);

    auto kernel_config = bw::resolve_compute_kernel_config(compute_kernel_config, input_tensor);
    const uint32_t local_width = input_tensor.logical_shape()[3];
    auto rms = bw::rms_from_gathered_stats(stats, local_width, epsilon, kernel_config);
    auto local_sum = ttnn::sum(
        bw::x_times_gained(input_tensor, output_grad, rms, weight),
        /*dim_arg=*/3,
        /*keep_dim=*/true,
        std::nullopt,
        kernel_config);
    return bw::to_stats_layout(local_sum);
}

}  // namespace ttnn
