// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::operations::reduction::reduce_avg_w_rm {
using namespace tt::tt_metal;

struct operation_attributes_t {
    const std::optional<MemoryConfig> output_mem_config;
    const std::optional<DeviceComputeKernelConfig> compute_kernel_config;
};

struct tensor_args_t {
    const Tensor& input;
};

using tensor_return_value_t = Tensor;

using spec_return_value_t = TensorSpec;

}  // namespace ttnn::operations::reduction::reduce_avg_w_rm
