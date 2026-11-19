// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/transformer/sdpa_config.hpp"

namespace ttnn::prim {

struct RingDistributedSDPAParams {
    uint32_t ring_size = 0;
    std::optional<uint32_t> ring_id;
    std::optional<float> scale;
    tt::tt_metal::MemoryConfig output_mem_config;
    std::optional<ttnn::operations::transformer::SDPAProgramConfig> program_config;
    DeviceComputeKernelConfig compute_kernel_config;
    std::optional<int64_t> chunk_start_idx;
    // Precision recipe (device/recipe/); unset: the legacy kernels.
    std::optional<ttnn::operations::transformer::SDPAPrecision> precision;
    // Recipe calls: the L1 their layout may use (recipe_l1_budget), set before the program-cache lookup.
    uint64_t recipe_l1_budget = 0;
};

struct RingDistributedSDPAInputs {
    ttnn::Tensor q;
    ttnn::Tensor k;
    ttnn::Tensor v;
    std::optional<ttnn::Tensor> page_table;
};

}  // namespace ttnn::prim
