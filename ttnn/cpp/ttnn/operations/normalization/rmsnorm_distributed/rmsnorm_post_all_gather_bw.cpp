// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "rmsnorm_post_all_gather_bw.hpp"

#include <string_view>

#include <tt_stl/assert.hpp>
#include "rmsnorm_distributed_bw_utils.hpp"

namespace ttnn {

std::vector<std::optional<Tensor>> rms_norm_post_all_gather_bw(
    const Tensor& input_tensor,
    const Tensor& output_grad,
    const Tensor& stats,
    const Tensor& bw_stats,
    float epsilon,
    const std::optional<const Tensor>& weight,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config) {
    namespace bw = ttnn::operations::normalization::rmsnorm_distributed_bw;
    constexpr std::string_view op_name = "rms_norm_post_all_gather_bw";
    bw::validate_bw_inputs(input_tensor, output_grad, weight, op_name);
    bw::validate_stats_tensor(stats, input_tensor, "stats", op_name);
    bw::validate_stats_tensor(bw_stats, input_tensor, "bw_stats", op_name);

    // Both statistics are divided by local_width * num_devices to recover a full-row mean, so a
    // pair gathered over different device sets would scale rms and the gradient differently and
    // still produce plausible-looking numbers.
    const uint32_t stats_devices = bw::num_devices_in_stats(stats);
    const uint32_t bw_stats_devices = bw::num_devices_in_stats(bw_stats);
    TT_FATAL(
        stats_devices == bw_stats_devices,
        "{}: stats holds one tile column for each of {} devices but bw_stats holds {}; both must be all-gathered "
        "over the same devices",
        op_name,
        stats_devices,
        bw_stats_devices);

    auto kernel_config = bw::resolve_compute_kernel_config(compute_kernel_config, input_tensor);
    const uint32_t local_width = input_tensor.logical_shape()[3];
    auto rms = bw::rms_from_gathered_stats(stats, local_width, epsilon, kernel_config);
    auto scale = bw::mean_from_gathered_stats(bw_stats, local_width, kernel_config);
    return bw::apply_backward(input_tensor, output_grad, rms, scale, weight, kernel_config);
}

}  // namespace ttnn
