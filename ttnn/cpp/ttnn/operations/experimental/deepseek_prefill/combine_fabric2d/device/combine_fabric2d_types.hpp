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

// See the combine_fabric2d nanobind docstring for what each tensor carries.
//
// `device` is FIRST so the framework's get_first_object_of_type<MeshDevice*>() over attribute_values()
// finds the mesh.
struct CombineFabric2dParams {
    ttnn::MeshDevice* device = nullptr;
    uint32_t dispatch_group_size = 8;
    uint32_t experts_per_chip = 2;
    uint32_t num_experts_per_tok = 2;
    uint32_t seq_len_per_chip = 640;
    uint32_t axis = 0;
    uint32_t num_links = 2;
    tt::tt_fabric::Topology topology = tt::tt_fabric::Topology::Mesh;
    tt::tt_metal::MemoryConfig output_mem_config{
        tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM};
    // Accepted only as false: unrouted output slots are left as-allocated, as production leaves them
    // when asked not to zero.
    bool init_zeros = false;

    static constexpr auto attribute_names = std::forward_as_tuple(
        "device",
        "dispatch_group_size",
        "experts_per_chip",
        "num_experts_per_tok",
        "seq_len_per_chip",
        "axis",
        "num_links",
        "topology",
        "output_mem_config",
        "init_zeros");
    auto attribute_values() const {
        return std::forward_as_tuple(
            device,
            dispatch_group_size,
            experts_per_chip,
            num_experts_per_tok,
            seq_len_per_chip,
            axis,
            num_links,
            topology,
            output_mem_config,
            init_zeros);
    }
};

struct CombineFabric2dInputs {
    ttnn::Tensor dispatched_buffer;
    ttnn::Tensor dispatched_metadata;
    ttnn::Tensor expert_token_counts;
    ttnn::Tensor expert_region_offsets;
    ttnn::Tensor expert_offsets;
};

}  // namespace ttnn::operations::experimental::deepseek_prefill::combine_fabric2d
