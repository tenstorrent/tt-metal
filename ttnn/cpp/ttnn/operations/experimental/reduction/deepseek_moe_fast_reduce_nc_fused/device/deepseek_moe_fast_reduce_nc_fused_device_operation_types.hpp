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
    uint32_t cluster_axis;
    tt::tt_metal::MemoryConfig output_memory_config;
    uint32_t num_shared_experts;
    float shared_expert_scale;
    ttnn::DeviceComputeKernelConfig compute_kernel_config;
};

// Input tensors:
//   input_tensor           - activations [experts_k, 1, tokens, hidden], TILE layout, L1
//   scores_tensor          - expert scores [tokens, 1, seq, experts_k], ROW_MAJOR layout, DRAM
//   expert_indices_tensor  - per-token expert indices (matches all_to_all_dispatch convention)
//   expert_mapping_tensor  - expert-to-device mapping (matches all_to_all_dispatch convention)
struct DeepseekMoEFastReduceNCFusedInputs {
    ttnn::Tensor input_tensor;
    ttnn::Tensor scores_tensor;
    ttnn::Tensor expert_indices_tensor;
    ttnn::Tensor expert_mapping_tensor;
};

}  // namespace ttnn::experimental::prim
