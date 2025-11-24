// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::operations::kv_cache {
enum class UpdateCacheOpParallelizationStrategy { MULTI_CORE };

enum class UpdateCacheOpType { FILL, UPDATE };

struct operation_attributes_t {
    const uint32_t batch_idx;
    const uint32_t update_idx;
    const uint32_t batch_offset;
    const UpdateCacheOpType op_type;
    const std::optional<const DeviceComputeKernelConfig> compute_kernel_config;
};

struct tensor_args_t {
    const Tensor cache;
    const Tensor input;
};

using tensor_return_value_t = Tensor;
using spec_return_value_t = TensorSpec;

}  // namespace ttnn::operations::kv_cache
