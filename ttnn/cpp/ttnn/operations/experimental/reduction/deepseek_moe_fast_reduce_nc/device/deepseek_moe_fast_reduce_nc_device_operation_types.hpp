// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::operations::experimental::reduction::deepseek_moe_fast_reduce_nc::detail {

struct operation_attributes_t {
    uint32_t dim;
    tt::tt_metal::MemoryConfig output_memory_config;
    ttnn::DeviceComputeKernelConfig compute_kernel_config;
};

struct tensor_args_t {
    ttnn::Tensor input_tensor;
};

using spec_return_value_t = ttnn::TensorSpec;
using tensor_return_value_t = std::vector<ttnn::Tensor>;

}  // namespace ttnn::operations::experimental::reduction::deepseek_moe_fast_reduce_nc::detail
