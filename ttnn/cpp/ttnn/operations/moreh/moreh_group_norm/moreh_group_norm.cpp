// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_group_norm.hpp"

#include "device/moreh_group_norm_device_operation.hpp"

namespace ttnn::operations::moreh::moreh_group_norm {
std::vector<std::optional<Tensor>> MorehGroupNorm::invoke(
    const Tensor& input,
    const uint32_t num_groups,
    const float eps,
    const std::optional<const Tensor> gamma,
    const std::optional<const Tensor> beta,
    const std::vector<bool>& are_required_outputs,
    const std::optional<const Tensor> output,
    const std::optional<const Tensor> mean,
    const std::optional<const Tensor> rstd,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<MemoryConfig>& mean_memory_config,
    const std::optional<MemoryConfig>& rstd_memory_config,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {
    return ttnn::prim::moreh_group_norm(input,
                                        num_groups,
                                        eps,
                                        gamma,
                                        beta,
                                        are_required_outputs,
                                        output,
                                        mean,
                                        rstd,
                                        memory_config,
                                        mean_memory_config,
                                        rstd_memory_config,
                                        compute_kernel_config);
}
std::vector<Tensor> MorehGroupNorm::create_async_output_tensors(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_inputs) {
    return {
        Tensor(operation::get_workers_for_op_output(input_tensors, optional_inputs)),
        Tensor(operation::get_workers_for_op_output(input_tensors, optional_inputs)),
        Tensor(operation::get_workers_for_op_output(input_tensors, optional_inputs)),
    };
}
std::vector<bool> MorehGroupNorm::create_async_return_flag(
    const Tensor& input,
    const uint32_t num_groups,
    const float eps,
    const std::optional<const Tensor> gamma,
    const std::optional<const Tensor> beta,
    const std::vector<bool>& are_required_outputs,
    const std::optional<const Tensor> output,
    const std::optional<const Tensor> mean,
    const std::optional<const Tensor> rstd,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<MemoryConfig>& mean_memory_config,
    const std::optional<MemoryConfig>& rstd_memory_config,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {
    return are_required_outputs;
}
}  // namespace ttnn::operations::moreh::moreh_group_norm
