// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"
#include "ttnn/distributed/types.hpp"
#include <tt-metalium/experimental/fabric/fabric.hpp>

namespace ttnn::operations::experimental::deepseek_prefill::dispatch_fabric2d {

// Every field is part of the program-cache key.
struct DispatchFabric2dParams {
    ttnn::MeshDevice* device = nullptr;
    uint32_t experts_per_chip = 2;
    uint32_t num_routed_experts = 8;
    uint32_t num_experts_per_tok = 2;
    uint32_t metadata_len = 3;
    // Token capacity of a chip's dispatch buffer. A token past it is dropped but still advances its expert's
    // count, because page numbers must match the offsets table, which counts every routed token.
    uint32_t max_dispatch_buffer_token_size = 64;
    uint32_t seq_len_per_chip = 640;
    uint32_t axis = 0;
    uint32_t num_links = 2;
    // The kernel reads padding_config under a compile-time branch, so this must be in the cache key.
    bool has_padding_config = false;
    tt::tt_fabric::Topology topology = tt::tt_fabric::Topology::Mesh;
    tt::tt_metal::MemoryConfig output_mem_config{
        tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM};
    // Cores the op may use, resolved from the caller's sub-device.
    CoreRangeSet worker_core_range_set;
};

struct DispatchFabric2dInputs {
    ttnn::Tensor input_tensor;
    ttnn::Tensor indices_tensor;
    ttnn::Tensor expert_offsets_tensor;
    ttnn::Tensor expert_dispatch_table_tensor;
    // The last source chip's chunk for expert e ends at expert_token_counts[e] + expert_region_offsets[e].
    ttnn::Tensor expert_token_counts;
    ttnn::Tensor expert_region_offsets;
    // [real_token_count, pad_side]. With right padding (pad_side 0) only the first real_token_count tokens
    // are routed; other sides are ignored.
    std::optional<ttnn::Tensor> padding_config;
};

}  // namespace ttnn::operations::experimental::deepseek_prefill::dispatch_fabric2d
