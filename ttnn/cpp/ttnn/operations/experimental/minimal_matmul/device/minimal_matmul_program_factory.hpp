// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operation.hpp"

namespace ttnn::operations::experimental::minimal_matmul::detail {

tt::tt_metal::operation::ProgramWithCallbacks minimal_matmul_factory(
    const Tensor& input_tensor,
    const Tensor& weight_tensor,
    const std::optional<const Tensor>& bias_tensor,
    const MinimalMatmulConfig& config,
    const Tensor& output_tensor,
    const DeviceComputeKernelConfig& compute_kernel_config);

}  // namespace ttnn::operations::experimental::minimal_matmul::detail
