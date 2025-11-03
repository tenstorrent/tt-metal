// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "rmsnorm_pre_all_gather.hpp"

#include "ttnn/operations/experimental/ccl/fused_distributed_rmsnorm/device/rmsnorm_pre_all_gather_op.hpp"

namespace operation = tt::tt_metal::operation;

namespace ttnn::operations::experimental::ccl {

ttnn::Tensor ExecuteFusedRMSNormPreAllGather::invoke(
    const ttnn::Tensor& input_tensor,
    const DataType dtype,
    const std::optional<const DeviceComputeKernelConfig> compute_kernel_config,
    const std::optional<MemoryConfig>& memory_config) {
    auto arch = input_tensor.device()->arch();
    auto kernel_config_val =
        init_device_compute_kernel_config(arch, compute_kernel_config, MathFidelity::HiFi4, false, true, false);
    return operation::run(
               FusedRMSNormPreAllGather{.dtype = dtype, .compute_kernel_config = kernel_config_val}, {input_tensor})
        .at(0);
}

}  // namespace ttnn::operations::experimental::ccl
