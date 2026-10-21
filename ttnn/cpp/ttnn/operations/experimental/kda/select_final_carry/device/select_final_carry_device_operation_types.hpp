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
    // The prefix's final state is valid only on the last chronological rank (the SP relay leaves it there), so
    // without a split tail that rank multicasts it instead of every rank copying its own.
    bool prefix_on_last_rank = false;
    // The worker cores, when given: the first num_links send over the fabric and all of them copy.
    std::optional<std::vector<tt::tt_metal::CoreCoord>> cores;
    // L1 addresses of the caller's barrier and arrival global semaphores; without them each cached workload allocates
    // its own.
    std::optional<std::array<uint32_t, 2>> semaphore_addresses;
};

struct SelectFinalCarryInputs {
    Tensor rank_final;
    Tensor prefix_final;
    Tensor actual_start;
    std::optional<Tensor> actual_end;
};

}  // namespace ttnn::experimental::prim
