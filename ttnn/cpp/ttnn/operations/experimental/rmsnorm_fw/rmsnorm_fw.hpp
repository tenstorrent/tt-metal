
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace ttnn::operations::experimental {

struct RMSNormForwardOperation {
    // static std::vector<Tensor> invoke(
    static std::tuple<Tensor, Tensor> invoke(
        const Tensor& input_tensor,
        const Tensor& gamma_tensor,
        bool return_intermediates = true,
        float epsilon = 1e-6F);
};
}  // namespace ttnn::operations::experimental
namespace ttnn::experimental {
constexpr auto rmsnorm_fw = ttnn::register_operation_with_auto_launch_op<
    "ttnn::experimental::rmsnorm_fw",
    ttnn::operations::experimental::RMSNormForwardOperation>();
}  // namespace ttnn::experimental
