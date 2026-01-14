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
    uint32_t batch_idx = 0;
    uint32_t update_idx = 0;
    uint32_t batch_offset = 0;
    UpdateCacheOpType op_type = UpdateCacheOpType::FILL;
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config;
};

struct tensor_args_t {
    Tensor cache;
    Tensor input;
};

}  // namespace ttnn::operations::kv_cache
