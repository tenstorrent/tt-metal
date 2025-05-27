// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tuple>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::data_movement::detail {

// Utility function
uint32_t calculate_starting_idx_h(const Tensor& tensor, uint32_t num_slices, uint32_t slice_index);

struct WidthShardingReshardSegment {
    uint32_t write_size = 0;
    uint32_t read_offset = 0;
    uint32_t bank_id = 0;
    uint32_t write_offset = 0;
};

// Precompute a set of reads/writes for each core needed to perform a width-sharded reshard operations
std::tuple<std::vector<std::vector<WidthShardingReshardSegment>>, uint32_t, uint32_t, uint32_t>
compute_width_sharding_reshard_segments(
    const std::array<uint32_t, 2>& local_shard_shape,
    const std::array<uint32_t, 2>& remote_shard_shape,
    const std::vector<CoreCoord>& local_cores,
    const std::vector<CoreCoord>& remote_cores,
    const tt::tt_metal::BufferType& remote_buffer_type,
    const CoreType& remote_core_type,
    tt::tt_metal::IDevice* device,
    uint32_t element_size);

}  // namespace ttnn::operations::data_movement::detail
