// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "rmsnorm_pre_all_gather.hpp"

#include "ttnn/operations/experimental/transformer/fused_distributed_rmsnorm/device/fused_rmsnorm_pre_all_gather_device_operation.hpp"

namespace ttnn::experimental {

ttnn::Tensor wan_fused_rmsnorm_pre_allgather(
    const ttnn::Tensor& input_tensor,
    const DataType dtype,
    const std::optional<const DeviceComputeKernelConfig> compute_kernel_config,
    const std::optional<MemoryConfig>& /*memory_config*/) {
    auto arch = input_tensor.device()->arch();
    // Due to hardware bug (#38306), HiFi4 + fp32_dest_acc_en can sometime produce incorrect results on Wormhole.
    // Use HiFi3 when fp32_dest_acc_en is True on Wormhole (less likely to give bad results).
    const auto math_fidelity =
        arch == tt::ARCH::WORMHOLE_B0 ? tt::tt_metal::MathFidelity::HiFi3 : tt::tt_metal::MathFidelity::HiFi4;
    ttnn::verify_numerical_configuration(arch, compute_kernel_config);
    auto kernel_config_val =
        init_device_compute_kernel_config(arch, compute_kernel_config, math_fidelity, false, true, false);
    return ttnn::prim::fused_rmsnorm_pre_all_gather(input_tensor, dtype, kernel_config_val);
}

}  // namespace ttnn::experimental
