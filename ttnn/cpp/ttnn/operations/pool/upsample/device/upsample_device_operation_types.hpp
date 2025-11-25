// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::operations::pool::upsample {

struct operation_attributes_t {
    int scale_factor_h;
    int scale_factor_w;
    std::string mode;
    tt::tt_metal::MemoryConfig output_mem_config;
    DeviceComputeKernelConfig compute_kernel_config;
};

struct tensor_args_t {
    const Tensor input_tensor;
};

using tensor_return_value_t = Tensor;
using spec_return_value_t = TensorSpec;

}  // namespace ttnn::operations::pool::upsample
