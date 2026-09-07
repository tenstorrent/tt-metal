// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"
#include "ttnn/distributed/types.hpp"
#include <tt-metalium/experimental/fabric/fabric.hpp>

namespace ttnn::operations::experimental::deepseek_prefill::dispatch_fabric2d {

// See the dispatch_fabric2d nanobind docstring for what each tensor carries. The framework reflects over
// these members for the program-cache hash and to find the mesh, so every field is part of the cache key.
struct DispatchFabric2dParams {
    ttnn::MeshDevice* device = nullptr;
    uint32_t experts_per_chip = 2;
    uint32_t num_routed_experts = 8;
    uint32_t num_experts_per_tok = 2;
    uint32_t metadata_len = 3;
    // Total token capacity of the destination dispatch buffer, shared across that chip's experts.
    // Also the in-kernel bound: the production op drops a token past it while still advancing the
    // per-expert counter, and this op has to make the same choice to land the same pages.
    uint32_t max_dispatch_buffer_token_size = 64;
    uint32_t seq_len_per_chip = 640;
    uint32_t axis = 0;
    uint32_t num_links = 2;
    tt::tt_fabric::Topology topology = tt::tt_fabric::Topology::Mesh;
    tt::tt_metal::MemoryConfig output_mem_config{
        tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM};
};

struct DispatchFabric2dInputs {
    ttnn::Tensor input_tensor;
    ttnn::Tensor indices_tensor;
    ttnn::Tensor expert_offsets_tensor;
    ttnn::Tensor expert_dispatch_table_tensor;
};

}  // namespace ttnn::operations::experimental::deepseek_prefill::dispatch_fabric2d
