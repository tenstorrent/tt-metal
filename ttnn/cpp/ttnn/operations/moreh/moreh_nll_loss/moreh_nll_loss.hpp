
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::operations::moreh::moreh_nll_loss {

struct MorehNllLoss {
    static Tensor invoke(
        const Tensor& input_tensor,
        const Tensor& target_tensor,
        const std::string& reduction,
        const std::optional<Tensor>& weight_tensor,
        const std::optional<Tensor>& divisor_tensor,
        const std::optional<Tensor>& output_tensor,
        const int32_t ignore_index,
        const std::optional<MemoryConfig>& memory_config,
        const std::optional<DeviceComputeKernelConfig>& compute_kernel_config);
};

}  // namespace ttnn::operations::moreh::moreh_nll_loss

namespace ttnn {
constexpr auto moreh_nll_loss =
    ttnn::register_operation<"ttnn::moreh_nll_loss", operations::moreh::moreh_nll_loss::MorehNllLoss>();
}  // namespace ttnn
