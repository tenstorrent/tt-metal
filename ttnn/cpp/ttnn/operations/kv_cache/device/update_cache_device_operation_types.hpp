// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

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
    // FILL with input_row_offset: rows of the input copied, starting at the row the tensor holds.
    uint32_t num_input_rows = 0;
};

struct KvCacheInputs {
    Tensor cache;
    Tensor input;
    // FILL only: ``[1]`` INT32 / UINT32 first input row to copy, read on device so a trace can pick
    // the rows per replay.
    std::optional<Tensor> input_row_offset;
};

}  // namespace ttnn::prim
