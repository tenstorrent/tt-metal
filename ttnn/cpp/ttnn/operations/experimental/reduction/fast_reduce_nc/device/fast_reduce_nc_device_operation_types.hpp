// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::operations::experimental::reduction::detail {

struct operation_attributes_t {
    const int32_t dim;
    const tt::tt_metal::MemoryConfig output_mem_config;
    const ttnn::DeviceComputeKernelConfig compute_kernel_config;
};

struct tensor_args_t {
    const Tensor input;
    std::optional<Tensor> preallocated_output;
};

}  // namespace ttnn::operations::experimental::reduction::detail
