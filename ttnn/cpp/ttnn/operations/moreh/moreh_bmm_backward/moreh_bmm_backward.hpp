// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/moreh_matmul/moreh_matmul_op.hpp"
#include "tt_metal/host_api.hpp"

namespace ttnn::operations::moreh::moreh_bmm_backward {
struct MorehBmm {
    static std::vector<std::optional<Tensor>> invoke(
        const Tensor& output_grad,
        const Tensor& input,
        const Tensor& mat2,
        const std::vector<bool>& are_required_outputs,
        std::optional<const Tensor>& input_grad,
        std::optional<const Tensor>& mat2_grad,
        const MemoryConfig& input_grad_mem_config,
        const MemoryConfig& mat2_grad_mem_config,
        std::optional<const ttnn::DeviceComputeKernelConfig>& compute_kernel_config);
};
}  // namespace ttnn::operations::moreh::moreh_bmm_backward

namespace ttnn {
constexpr auto moreh_bmm_backward =
    ttnn::register_operation<"ttnn::moreh_bmm_backward", ttnn::operations::moreh::moreh_bmm_backward::MorehBmm>();
}
