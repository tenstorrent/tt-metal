// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/udm/block_utils.hpp"
#include "tt_metal/udm/tensor_builder.hpp"
#include <tt_stl/assert.hpp>

namespace tt::tt_metal::udm {

GcoresInfo map_tensor_to_gcores(const TensorBuilder& builder, const ttnn::Tensor& tensor, int partition_dim) {
    // TODO: Implement tensor to gcores mapping
    //
    // Algorithm:
    // 1. Determine data sharding from tensor.tensor_topology().placements()
    //    - Which tensor dims are sharded across which mesh dims
    //
    // 2. Determine work partition dimension
    //    - If partition_dim == -1: choose orthogonal to sharding dim
    //    - Else: use specified partition_dim
    //
    // 3. For each work unit (e.g., each row if partition_dim == 0):
    //    - Data for this work unit spans multiple grids (due to sharding)
    //    - Need 1 gcore per grid that owns part of this work unit
    //
    // 4. Assign gcores from appropriate grids:
    //    - Determine which grids own data for each work unit
    //    - Pick cores from those grids' grids
    //    - Distribute work evenly within each grid
    //
    // 5. Calculate pages_per_gcore:
    //    - Based on local tensor shape and work partition
    //
    // 6. Populate mapping vectors:
    //    - gcore_to_device_id[i]: which grid owns gcore i
    //    - gcore_to_block_page_start[i]: starting page in global space
    //
    // Example for (4, 16) tensor width-sharded on 1×4, partition_dim=0:
    //   - 4 rows to distribute
    //   - Each row spans 4 grids (width-sharded)
    //   - Assign 1 gcore per grid per row = 16 gcores total
    //   - Gcore 0: grid 0, row 0, pages 0-3
    //   - Gcore 1: grid 1, row 0, pages 4-7
    //   - Gcore 4: grid 0, row 1, pages 16-19
    //   - etc.

    TT_FATAL(false, "map_tensor_to_gcores not yet implemented");
    return GcoresInfo{};
}

}  // namespace tt::tt_metal::udm
