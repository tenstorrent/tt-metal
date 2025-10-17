// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/functions.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/run_operation.hpp"

namespace ttnn::operations::reduction::accumulation {

tt::tt_metal::operation::ProgramWithCallbacks ema_multi_core(
    const Tensor& a,
    Tensor& output,
    float alpha,
    CoreCoord grid_size,
    const DeviceComputeKernelConfig& compute_kernel_config);

struct Ema {
    float alpha;
    CoreCoord grid_size;
    MemoryConfig output_mem_config;
    const DeviceComputeKernelConfig compute_kernel_config;

    void validate(const std::vector<Tensor>& input_tensors) const;
    std::vector<TensorSpec> compute_output_specs(const std::vector<Tensor>& input_tensors) const;
    std::vector<Tensor> create_output_tensors(
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<Tensor>>& optional_output_tensors) const;
    tt::tt_metal::operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const;
};

}  // namespace ttnn::operations::reduction::accumulation
