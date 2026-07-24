// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <tuple>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::prim {
enum class UpdateCacheOpParallelizationStrategy { MULTI_CORE };

enum class UpdateCacheOpType { FILL, UPDATE };

struct KvCacheParams {
    uint32_t batch_idx = 0;
    uint32_t update_idx = 0;
    uint32_t batch_offset = 0;
    UpdateCacheOpType op_type = UpdateCacheOpType::FILL;
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config;

    static constexpr auto attribute_names = std::forward_as_tuple("op_type");
    auto attribute_values() const { return std::forward_as_tuple(this->op_type); }
};

struct KvCacheInputs {
    Tensor cache;
    Tensor input;
};

}  // namespace ttnn::prim
