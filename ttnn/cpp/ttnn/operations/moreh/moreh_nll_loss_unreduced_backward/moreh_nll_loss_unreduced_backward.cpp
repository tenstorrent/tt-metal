
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_nll_loss_unreduced_backward.hpp"

#include "device/moreh_nll_loss_unreduced_backward_device_operation.hpp"

namespace ttnn::operations::moreh::moreh_nll_loss_unreduced_backward {

Tensor MorehNllLossUnreducedBackward::invoke(
    const Tensor &target_tensor,
    const Tensor &output_grad_tensor,
    const std::optional<Tensor> &weight_tensor,
    const std::optional<Tensor> &input_grad_tensor,
    const int32_t ignore_index,
    const std::optional<ttnn::MemoryConfig> &memory_config,
    std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config) {
    return prim::moreh_nll_loss_unreduced_backward(target_tensor,
                                                   output_grad_tensor,
                                                   weight_tensor,
                                                   input_grad_tensor,
                                                   ignore_index,
                                                   memory_config,
                                                   compute_kernel_config);
}

}  // namespace ttnn::operations::moreh::moreh_nll_loss_unreduced_backward
