// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/transformer/sdpa_config.hpp"
#include <tuple>

namespace ttnn::prim {

struct RingDistributedSDPAParams {
    uint32_t ring_size = 0;
    std::optional<uint32_t> ring_id;
    std::optional<float> scale;
    tt::tt_metal::MemoryConfig output_mem_config;
    std::optional<ttnn::operations::transformer::SDPAProgramConfig> program_config;
    DeviceComputeKernelConfig compute_kernel_config;
    std::optional<int64_t> chunk_start_idx;

    static constexpr auto attribute_names = std::forward_as_tuple(
        "ring_size",
        "ring_id",
        "scale",
        "output_mem_config",
        "program_config",
        "compute_kernel_config",
        "chunk_start_idx");
    auto attribute_values() const {
        return std::forward_as_tuple(
            ring_size, ring_id, scale, output_mem_config, program_config, compute_kernel_config, chunk_start_idx);
    }
};

struct RingDistributedSDPAInputs {
    ttnn::Tensor q;
    ttnn::Tensor k;
    ttnn::Tensor v;
    std::optional<ttnn::Tensor> page_table;

    static constexpr auto attribute_names = std::forward_as_tuple("q", "k", "v", "page_table");
    auto attribute_values() const { return std::forward_as_tuple(q, k, v, page_table); }
};

}  // namespace ttnn::prim
