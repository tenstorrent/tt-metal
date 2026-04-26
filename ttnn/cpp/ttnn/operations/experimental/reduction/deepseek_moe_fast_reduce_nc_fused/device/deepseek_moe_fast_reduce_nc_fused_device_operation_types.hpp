// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::experimental::prim {

struct DeepseekMoEFastReduceNCFusedParams {
    uint32_t reduce_dim;
    uint64_t split_size;
    tt::tt_metal::MemoryConfig output_memory_config;
    ttnn::DeviceComputeKernelConfig compute_kernel_config;
};

// Two input tensors:
//   input_tensor  - activations [experts_k, 1, tokens, hidden], TILE layout, L1
//   scores_tensor - expert scores [tokens, 1, seq, experts_k], ROW_MAJOR layout, DRAM
struct DeepseekMoEFastReduceNCFusedInputs {
    ttnn::Tensor input_tensor;
    ttnn::Tensor scores_tensor;
};

}  // namespace ttnn::experimental::prim
