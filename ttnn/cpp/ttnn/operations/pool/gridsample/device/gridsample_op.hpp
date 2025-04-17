// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::operations::gridsample {

struct gridsample {
    std::vector<float> normalized_grid;
    std::string mode;
    bool align_corners;
    const tt::tt_metal::MemoryConfig output_mem_config_;
    const DeviceComputeKernelConfig compute_kernel_config_;

    void validate(const std::vector<Tensor>& input_tensors) const;
    ttnn::Shape compute_output_shape(const ttnn::Shape& input_shape, const ttnn::Shape& grid_shape) const;
    std::vector<TensorSpec> compute_output_specs(const std::vector<Tensor>& input_tensors) const;
    tt::tt_metal::operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const;
};

tt::tt_metal::operation::ProgramWithCallbacks gridsample_rm_single_core(
    const Tensor& input,
    Tensor& output,
    const Tensor& reshaped_input,
    const std::vector<float>& normalized_grid,
    const std::string& mode,
    bool align_corners);
}  // namespace ttnn::operations::gridsample
