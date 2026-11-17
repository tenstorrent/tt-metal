// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <optional>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

struct SelectFinalCarryParams {
    uint32_t sequence_parallel_axis;
    uint32_t local_rows;
    uint32_t num_links;
    tt::tt_fabric::Topology topology;
    tt::tt_metal::MemoryConfig output_mem_config;
    // One worker core per fabric link (the first num_links are used).
    std::vector<tt::tt_metal::CoreCoord> cores;
    // L1 addresses of the caller's barrier and arrival global semaphores.
    std::array<uint32_t, 2> semaphore_addresses{};
    uint32_t slab_index_scale = 1;
    uint32_t slab_index_offset = 0;
};

struct SelectFinalCarryInputs {
    Tensor rank_final;
    Tensor prefix_final;
    Tensor actual_start;
    std::optional<Tensor> actual_end;
    // A [entries, B*H, K, V] slab the final state is written into, in place. Pass both or neither.
    std::optional<Tensor> slab;
    std::optional<Tensor> slab_index;
};

}  // namespace ttnn::experimental::prim
