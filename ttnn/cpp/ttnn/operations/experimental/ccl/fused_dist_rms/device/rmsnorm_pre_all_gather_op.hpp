// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/core.hpp"

// Local fused RMSNorm implementation does not depend on normalization headers.

namespace ttnn::operations::experimental::ccl {

tt::tt_metal::operation::ProgramWithCallbacks fused_rmsnorm_pre_allgather_multi_core(
    const Tensor& a, Tensor& output, DeviceComputeKernelConfig compute_kernel_config);

struct FusedRMSNormPreAllGather {
    const DataType dtype;
    const DeviceComputeKernelConfig compute_kernel_config;

    void validate(const std::vector<Tensor>& input_tensors) const;
    std::vector<TensorSpec> compute_output_specs(const std::vector<Tensor>& input_tensors) const;
    tt::tt_metal::operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const;
};

}  // namespace ttnn::operations::experimental::ccl
