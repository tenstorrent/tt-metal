// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "rmsnorm_pre_all_gather.hpp"

#include "ttnn/operations/normalization/layernorm_distributed/device/layernorm_pre_all_gather_op.hpp"

namespace ttnn::operations::normalization {

ttnn::Tensor ExecuteRMSNormPreAllGather::invoke(
    const ttnn::Tensor& input_tensor,
    const DataType dtype,
    const std::optional<const DeviceComputeKernelConfig> compute_kernel_config) {
    auto arch = input_tensor.storage_type() == StorageType::DEVICE ? DeviceArch(input_tensor.device()) : DeviceArch(ttnn::operations::experimental::auto_format::AutoFormat::GetDefaultDevice());
    auto kernel_config_val = init_device_compute_kernel_config(arch, compute_kernel_config, MathFidelity::HiFi4, true, false, false);
    return operation::run(
                LayerNormPreAllGather{
                    .norm_type = LayerNormDistributedType::RMSNORM,
                    .dtype = dtype,
                    .compute_kernel_config = kernel_config_val},
                {input_tensor}).at(0);
}

}  // namespace ttnn::operations::normalization
