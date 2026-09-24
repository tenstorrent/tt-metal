// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <vector>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"
#include "ttnn/distributed/types.hpp"
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>

namespace ttnn::operations::experimental::deepseek_prefill::combine_fabric2d {

// See the combine_fabric2d nanobind docstring for what each tensor carries. The framework reflects over
// these members for the program-cache hash and to find the mesh, so every field is part of the cache key.
struct CombineFabric2dParams {
    ttnn::MeshDevice* device = nullptr;
    uint32_t experts_per_chip = 2;
    uint32_t num_experts_per_tok = 2;
    uint32_t seq_len_per_chip = 640;
    uint32_t axis = 0;
    uint32_t num_links = 2;
    tt::tt_fabric::Topology topology = tt::tt_fabric::Topology::Mesh;
    tt::tt_metal::MemoryConfig output_mem_config{
        tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM};
};

struct CombineFabric2dInputs {
    ttnn::Tensor dispatched_buffer;
    ttnn::Tensor dispatched_metadata;
    ttnn::Tensor expert_token_counts;
    ttnn::Tensor expert_region_offsets;
    ttnn::Tensor expert_offsets;
};

}  // namespace ttnn::operations::experimental::deepseek_prefill::combine_fabric2d
