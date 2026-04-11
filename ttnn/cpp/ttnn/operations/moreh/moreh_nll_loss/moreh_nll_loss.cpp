
// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_nll_loss.hpp"

#include "moreh_nll_loss_helper.hpp"
#include "moreh_nll_loss_step1/device/moreh_nll_loss_step1_device_operation.hpp"
#include "moreh_nll_loss_step2/device/moreh_nll_loss_step2_device_operation.hpp"
#include "ttnn/operations/moreh/moreh_sum/moreh_sum.hpp"
#include "ttnn/operations/reduction/generic/generic_reductions.hpp"

namespace ttnn {

Tensor moreh_nll_loss(
    const Tensor& input_tensor,
    const Tensor& target_tensor,
    const std::string& reduction,
    const std::optional<Tensor>& weight_tensor,
    const std::optional<Tensor>& divisor_tensor,
    const std::optional<Tensor>& output_tensor,
    const int32_t ignore_index,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {
    using namespace operations::moreh;

    const auto compute_kernel_config_val =
        init_device_compute_kernel_config(target_tensor.device()->arch(), compute_kernel_config, tt::tt_metal::MathFidelity::HiFi4);
    if (reduction == MEAN) {
        TT_FATAL(divisor_tensor.has_value(), "Divisor tensor must not be empty");

        const Tensor& step1_result = prim::moreh_nll_loss_step1(
            target_tensor,
            weight_tensor,
            ignore_index,
            reduction,
            output_tensor.has_value() ? output_tensor.value().dtype() : input_tensor.dtype(),
            input_tensor.padded_shape()[1],
            memory_config,
            compute_kernel_config_val);

        // Use ttnn::sum instead of moreh_sum for the divisor computation.
        // moreh_sum produces incorrect results for small tensors (e.g. shape [5])
        // due to a tile-padding interaction bug.
        Tensor divisor_computed = ttnn::sum(step1_result, std::nullopt, false, memory_config);
        // Write back into the caller-provided divisor_tensor so the backward pass can read it.
        moreh_sum(step1_result, std::nullopt, false, divisor_tensor, memory_config, compute_kernel_config_val);
        // But use the correctly computed value for step2.
        std::optional<Tensor> divisor_opt = divisor_computed;

        const Tensor& step2_result = prim::moreh_nll_loss_step2(
            input_tensor,
            target_tensor,
            reduction,
            weight_tensor,
            divisor_opt,
            output_tensor,
            ignore_index,
            memory_config,
            compute_kernel_config_val);
        return moreh_sum(step2_result, std::nullopt, false, output_tensor, memory_config, compute_kernel_config_val);
    }
    if (reduction == SUM) {
        const Tensor& step2_result = prim::moreh_nll_loss_step2(
            input_tensor,
            target_tensor,
            reduction,
            weight_tensor,
            std::nullopt,
            output_tensor,
            ignore_index,
            memory_config,
            compute_kernel_config_val);
        return moreh_sum(step2_result, std::nullopt, false, output_tensor, memory_config, compute_kernel_config_val);
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

}  // namespace ttnn
