// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"
#include "ttnn/distributed/types.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::moe_fanout_reach {

// See the moe_fanout_reach nanobind docstring for what each field means. The framework reflects over
// these members for the program-cache hash and to find the mesh, so every field is part of the cache key.
struct MoeFanoutReachParams {
    ttnn::MeshDevice* device = nullptr;
    uint32_t num_routed_experts = 8;
    uint32_t num_experts_per_tok = 2;
    // The ring extent this table describes. Checked against the mesh extent on `axis` rather than
    // trusted: the two disagreeing is a table of the wrong width, which dispatch_fabric2d would then
    // reject or, worse, read past.
    uint32_t dispatch_group_size = 4;
    // The destination buffer's shared token capacity. A pick past it is dropped while its per-expert
    // counter still advances, and reach is defined on what SURVIVES that rule.
    uint32_t max_dispatch_buffer_token_size = 64;
    uint32_t axis = 0;
    tt::tt_metal::MemoryConfig output_mem_config{
        tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM};
    // Every core the op may use, resolved from the caller's sub-device in the front end. The cores are
    // what the program is built from and what the cache key has to distinguish, so the device op never
    // holds a SubDeviceId.
    CoreRangeSet worker_core_range_set;
};

struct MoeFanoutReachInputs {
    // Top-k expert ids per token, the same tensor dispatch_fabric2d is handed. Sharing one tensor is
    // the point: reach describes what THAT routing sends, and a second copy is a second chance for the
    // two to disagree, which strands the axis rather than producing wrong numbers.
    ttnn::Tensor indices_tensor;
    ttnn::Tensor expert_dispatch_table_tensor;
    // This device's row of the offsets table: where each expert's run from THIS chip starts. It is the
    // seed of the per-expert allocator, which is what decides whether a pick is dropped.
    ttnn::Tensor global_dispatch_offsets;
};

}  // namespace ttnn::operations::experimental::deepseek_prefill::moe_fanout_reach
