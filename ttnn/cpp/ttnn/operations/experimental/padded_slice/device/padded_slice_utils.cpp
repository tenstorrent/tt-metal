// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "padded_slice_utils.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::experimental::detail {

uint32_t get_num_cores_channels_from_sharded_tensor(const Tensor& tensor) {
    auto shard_spec = tensor.shard_spec().value();
    auto core_grid = shard_spec.grid;

    bool rm_orientation = shard_spec.orientation == ShardOrientation::ROW_MAJOR;

    uint32_t num_cores_channels = 1;
    if (tensor.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED) {
        if (rm_orientation) {
            num_cores_channels = core_grid.bounding_box().grid_size().x;
        } else {
            num_cores_channels = core_grid.bounding_box().grid_size().y;
        }
    } else if (tensor.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED) {
        num_cores_channels = core_grid.num_cores();
    }
    return num_cores_channels;
}

}  // namespace ttnn::operations::experimental::detail
