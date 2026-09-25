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

namespace ttnn::operations::experimental::deepseek_prefill::hybrid_routed_expert_ffn::combine {

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
    // The routed expert's hybrid_token_threshold. Experts are walked in the order the routed expert
    // finishes them -- count <= threshold first, then the rest -- and it MUST be the value that op used.
    uint32_t hybrid_token_threshold = 0;
    // Overlapped with the routed expert in one program: before reading an expert's rows, wait until the
    // routed expert reports it written. Off when the op runs alone, where the rows are there at launch.
    bool wait_for_routed_expert = false;
    // How many routed-expert writer cores report to the collector, each once per expert slot per pass.
    uint32_t routed_expert_writers = 0;
    // The routed expert's cores, one rectangle, and the global semaphore on them the collector sets once it
    // has zeroed its count array and reports may start.
    tt::tt_metal::CoreRange routed_expert_cores{tt::tt_metal::CoreCoord{0, 0}, tt::tt_metal::CoreCoord{0, 0}};
    uint32_t routed_expert_go_addr = 0;
    tt::tt_metal::MemoryConfig output_mem_config{
        tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM};
};

struct CombineFabric2dInputs {
    ttnn::Tensor dispatched_buffer;
    ttnn::Tensor dispatched_metadata;
    ttnn::Tensor expert_token_counts;
    ttnn::Tensor expert_region_offsets;
    ttnn::Tensor expert_offsets;
    // (dispatch groups, ring extent, experts_per_chip) global expert ids, REPLICATED on every chip: the same
    // table the routed expert takes a per-device slice of. A relay needs other chips' rows, not just its own.
    ttnn::Tensor global_expert_idx_table;
};

}  // namespace ttnn::operations::experimental::deepseek_prefill::hybrid_routed_expert_ffn::combine
