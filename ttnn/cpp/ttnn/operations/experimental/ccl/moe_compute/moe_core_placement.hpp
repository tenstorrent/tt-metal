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
    std::vector<CoreCoord> tilize_cores;
    std::vector<CoreCoord> matmul_cores;
    CoreRangeSet tilize_core_range_set;
    CoreRangeSet matmul_core_range_set;
    CoreRangeSet tilize_matmul_core_range_set;
    CoreRangeSet combine_core_range_set;
    CoreRangeSet combine_matmul_core_range_set;
    CoreRangeSet all_worker_cores_range_set;
    std::vector<CoreCoord> combine_cores;
    CoreRange tilize_bounding_box;
    CoreRange matmul_bounding_box;
};

MoEComputeCoreSelection select_moe_compute_cores(
    ttnn::MeshDevice* mesh_device,
    uint32_t combine_token_parallel_cores,
    uint32_t combine_data_parallel_cores,
    uint32_t hidden_size,
    const CoreRangeSet& mux_core_range_set,
    uint32_t bh_ring_size);

}  // namespace ttnn::operations::ccl::common
