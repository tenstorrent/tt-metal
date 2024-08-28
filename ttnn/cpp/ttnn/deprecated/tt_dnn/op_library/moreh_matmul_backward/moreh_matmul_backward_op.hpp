/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include "ttnn/deprecated/tt_dnn/op_library/compute_kernel_config.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace tt {

namespace operations {

namespace primary {

using namespace tt_metal;

std::vector<std::optional<Tensor>> moreh_matmul_backward(
    const Tensor &output_grad,
    const Tensor &input,
    const Tensor &other,
    const std::vector<bool> &are_required_outputs = std::vector<bool>{true, true},
    std::optional<const Tensor> input_grad = std::nullopt,
    std::optional<const Tensor> other_grad = std::nullopt,
    const MemoryConfig &mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);

}  // namespace primary

}  // namespace operations

}  // namespace tt
