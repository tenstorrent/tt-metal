
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_nll_loss_backward.hpp"

#include "device/moreh_nll_loss_backward_device_operation.hpp"

namespace ttnn::operations::moreh::moreh_nll_loss_backward {

Tensor MorehNllLossBackward::invoke(
    const Tensor &target_tensor,
    const Tensor &output_grad_tensor,
    const bool reduction_mean,
    const std::optional<Tensor>& weight_tensor,
    const std::optional<Tensor>& input_grad_tensor,
    const std::optional<Tensor>& divisor_tensor,
    const int32_t ignore_index,
    const std::optional<ttnn::MemoryConfig> &memory_config,
    std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config) {
    return prim::moreh_nll_loss_backward(
        target_tensor,
        output_grad_tensor,
        reduction_mean,
        weight_tensor,
        input_grad_tensor,
        divisor_tensor,
        ignore_index,
        memory_config,
        compute_kernel_config);
}

}  // namespace ttnn::operations::moreh::moreh_nll_loss_backward
