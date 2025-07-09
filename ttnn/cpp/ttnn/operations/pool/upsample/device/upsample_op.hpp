// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::operations::upsample {

enum class UpSampleParallelizationStrategy { MULTI_CORE, SINGLE_CORE };

struct UpSample {
    const int scale_factor_h_;
    const int scale_factor_w_;
    const std::string mode_;
    const tt::tt_metal::MemoryConfig output_mem_config_;
    const DeviceComputeKernelConfig compute_kernel_config_;

    void validate(const std::vector<Tensor>& input_tensors) const;
    std::vector<TensorSpec> compute_output_specs(const std::vector<Tensor>& input_tensors) const;
    tt::tt_metal::operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const;
    UpSampleParallelizationStrategy get_parallelization_strategy(const std::vector<Tensor>& input_tensors) const;
};

tt::tt_metal::operation::ProgramWithCallbacks upsample_single_core(
    const Tensor& input, Tensor& output, uint32_t scale_factor_h, uint32_t scale_factor_w);
tt::tt_metal::operation::ProgramWithCallbacks upsample_multi_core(
    const Tensor& input, Tensor& output, uint32_t scale_factor_h, uint32_t scale_factor_w);
tt::tt_metal::operation::ProgramWithCallbacks bilinear_multi_core(
    const Tensor& input,
    Tensor& output,
    uint32_t scale_factor_h,
    uint32_t scale_factor_w,
    DeviceComputeKernelConfig compute_kernel_config_);

}  // namespace ttnn::operations::upsample
