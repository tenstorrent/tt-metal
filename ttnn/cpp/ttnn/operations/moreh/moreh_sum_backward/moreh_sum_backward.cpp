// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_sum_backward.hpp"

#include "device/moreh_sum_backward_device_operation.hpp"
#include "ttnn/operations/moreh/moreh_helper_functions.hpp"

namespace ttnn::operations::moreh::moreh_sum_backward {
Tensor MorehSumBackward::invoke(const Tensor& output_grad,
                                const std::optional<Tensor>& input,
                                std::optional<std::variant<int64_t, std::vector<int64_t>>> dim,
                                bool keepdim,
                                const std::optional<Tensor>& input_grad,
                                const std::optional<MemoryConfig>& memory_config,
                                const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {
    TT_FATAL((input.has_value() || input_grad.has_value()), "either input or input_grad must have a value");
    uint32_t rank = input.has_value() ? input->get_shape().value.rank() : input_grad->get_shape().value.rank();
    std::vector<int64_t> dims = tt::operations::primary::get_dim(dim, rank);
    std::sort(dims.begin(), dims.end());
    return ttnn::prim::moreh_sum_backward(
        output_grad, input, dims, keepdim, input_grad, memory_config, compute_kernel_config);
}
}  // namespace ttnn::operations::moreh::moreh_sum_backward
