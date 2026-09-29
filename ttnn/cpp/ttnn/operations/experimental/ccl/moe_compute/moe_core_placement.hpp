// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include "ttnn/distributed/types.hpp"

namespace ttnn::operations::ccl::common {

struct MoEComputeCoreSelection {
    std::vector<tt::tt_metal::CoreCoord> tilize_cores;
    std::vector<tt::tt_metal::CoreCoord> matmul_cores;
    CoreRangeSet tilize_core_range_set;
    CoreRangeSet matmul_core_range_set;
    CoreRangeSet tilize_matmul_core_range_set;
    CoreRangeSet combine_core_range_set;
    CoreRangeSet combine_matmul_core_range_set;
    CoreRangeSet all_worker_cores_range_set;
    std::vector<tt::tt_metal::CoreCoord> combine_cores;
    CoreRange tilize_bounding_box;
    CoreRange matmul_bounding_box;
    // The prefill rings' cores, ring-major: ring_cores[r * ring_size + position]; ring 0 is matmul_cores. One ring
    // (prefill_rings <= 1) makes this matmul_cores itself.
    std::vector<tt::tt_metal::CoreCoord> ring_cores;
    CoreRangeSet ring_core_range_set;
};

// The NoC virtual channel of each ring core's writes (the a2a ring, the credits): position p starts from p & 3 and
// takes the first channel no earlier core on the same row holds, so the cores of one row (two per ring) never share
// a channel while four are free; past four cores on a row (three rings) the least used channel repeats. Ring 0's
// channels are the historical ones.
std::vector<uint32_t> ring_core_vchannels(const std::vector<tt::tt_metal::CoreCoord>& ring_cores, uint32_t ring_size);

MoEComputeCoreSelection select_moe_compute_cores(
    ttnn::MeshDevice* mesh_device,
    uint32_t combine_token_parallel_cores,
    uint32_t combine_data_parallel_cores,
    uint32_t hidden_size,
    const CoreRangeSet& mux_core_range_set,
    uint32_t bh_ring_size,
    uint32_t prefill_rings = 1);

}  // namespace ttnn::operations::ccl::common
