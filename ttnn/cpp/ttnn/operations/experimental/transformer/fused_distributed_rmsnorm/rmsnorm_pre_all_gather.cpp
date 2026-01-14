// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "rmsnorm_pre_all_gather.hpp"

#include "ttnn/operations/experimental/transformer/fused_distributed_rmsnorm/device/fused_rmsnorm_pre_all_gather_device_operation.hpp"

namespace ttnn::operations::experimental::transformer {

ttnn::Tensor ExecuteFusedRMSNormPreAllGather::invoke(
    const ttnn::Tensor& input_tensor,
    const DataType dtype,
    const std::optional<const DeviceComputeKernelConfig> compute_kernel_config,
    const std::optional<MemoryConfig>& /*memory_config*/) {
    auto arch = input_tensor.device()->arch();
    auto kernel_config_val =
        init_device_compute_kernel_config(arch, compute_kernel_config, MathFidelity::HiFi4, false, true, false);
    return ttnn::prim::fused_rmsnorm_pre_all_gather(input_tensor, dtype, kernel_config_val);
}

}  // namespace ttnn::operations::experimental::transformer
