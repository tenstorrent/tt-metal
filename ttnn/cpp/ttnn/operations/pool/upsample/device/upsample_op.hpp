// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::operations::upsample {

enum class UpSampleParallelizationStrategy { MULTI_CORE, SINGLE_CORE };

struct UpSample {
    const int scale_factor_h_;
    const int scale_factor_w_;
    const string mode_;
    const MemoryConfig output_mem_config_;
    const DeviceComputeKernelConfig compute_kernel_config_;

    void validate(const std::vector<Tensor> &input_tensors) const;
    std::vector<tt::tt_metal::LegacyShape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors) const;
    operation::ProgramWithCallbacks create_program(const std::vector<Tensor> &input_tensors,
                                                   std::vector<Tensor> &output_tensors) const;
    UpSampleParallelizationStrategy get_parallelization_strategy(const std::vector<Tensor> &input_tensors) const;
};

operation::ProgramWithCallbacks upsample_single_core(const Tensor &input,
                                                     Tensor &output,
                                                     const uint32_t scale_factor_h,
                                                     const uint32_t scale_factor_w);
operation::ProgramWithCallbacks upsample_multi_core(const Tensor &input,
                                                    Tensor &output,
                                                    const uint32_t scale_factor_h,
                                                    const uint32_t scale_factor_w);
operation::ProgramWithCallbacks bilinear_multi_core(const Tensor &input,
                                                    Tensor &output,
                                                    const uint32_t scale_factor_h,
                                                    const uint32_t scale_factor_w,
                                                    const DeviceComputeKernelConfig compute_kernel_config_);

}  // namespace ttnn::operations::upsample
