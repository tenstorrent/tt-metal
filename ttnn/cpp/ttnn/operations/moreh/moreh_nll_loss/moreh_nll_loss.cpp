
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_nll_loss.hpp"

#include "moreh_nll_loss_helper.hpp"
#include "moreh_nll_loss_step1/device/moreh_nll_loss_step1_device_operation.hpp"
#include "moreh_nll_loss_step2/device/moreh_nll_loss_step2_device_operation.hpp"
#include "ttnn/cpp/ttnn/operations/moreh/moreh_sum/moreh_sum.hpp"

namespace ttnn::operations::moreh::moreh_nll_loss {

Tensor MorehNllLoss::invoke(
    const Tensor &input_tensor,
    const Tensor &target_tensor,
    const std::string reduction,
    const std::optional<const Tensor> weight_tensor,
    const std::optional<const Tensor> divisor_tensor,
    const std::optional<const Tensor> output_tensor,
    const int32_t ignore_index,
    const std::optional<MemoryConfig> &memory_config,
    const std::optional<const DeviceComputeKernelConfig> compute_kernel_config) {
    const auto compute_kernel_config_val = init_device_compute_kernel_config(target_tensor.device()->arch(), compute_kernel_config, MathFidelity::HiFi4);
    if (reduction == MEAN) {
        TT_FATAL(divisor_tensor.has_value(), "Divisor tensor must not be empty");

        const Tensor &step1_result = prim::moreh_nll_loss_step1(
            target_tensor,
            weight_tensor,
            ignore_index,
            reduction,
            output_tensor.has_value() ? output_tensor.value().get_dtype() : input_tensor.get_dtype(),
            input_tensor.get_shape().value[1],
            memory_config,
            compute_kernel_config_val);

        ttnn::moreh_sum(
            step1_result, std::nullopt, false, divisor_tensor.value(), memory_config, compute_kernel_config_val);

        const Tensor &step2_result = prim::moreh_nll_loss_step2(
            input_tensor,
            target_tensor,
            reduction,
            weight_tensor,
            divisor_tensor,
            output_tensor,
            ignore_index,
            memory_config,
            compute_kernel_config_val);
        return ttnn::moreh_sum(step2_result, std::nullopt, false, output_tensor, memory_config, compute_kernel_config_val);
    } else if (reduction == SUM) {
        const Tensor &step2_result = prim::moreh_nll_loss_step2(
            input_tensor,
            target_tensor,
            reduction,
            weight_tensor,
            std::nullopt,
            output_tensor,
            ignore_index,
            memory_config,
            compute_kernel_config_val);
        return ttnn::moreh_sum(step2_result, std::nullopt, false, output_tensor, memory_config, compute_kernel_config_val);
    }

    return prim::moreh_nll_loss_step2(
        input_tensor,
        target_tensor,
        reduction,
        weight_tensor,
        std::nullopt,
        output_tensor,
        ignore_index,
        memory_config,
        compute_kernel_config_val);
}

}  // namespace ttnn::operations::moreh::moreh_nll_loss
