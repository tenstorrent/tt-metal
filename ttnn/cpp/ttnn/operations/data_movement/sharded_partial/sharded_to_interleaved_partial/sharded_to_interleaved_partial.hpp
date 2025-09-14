// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"

namespace ttnn {
namespace operations::data_movement {

struct ShardedToInterleavedPartialOperation {
    static ttnn::Tensor invoke(
        QueueId queue_id,
        const ttnn::Tensor& input_tensor,
        const ttnn::Tensor& cache_tensor,
        int64_t& num_slices,
        int64_t& slice_index,
        const std::optional<MemoryConfig>& memory_config_arg,
        const std::optional<DataType>& data_type_arg);
};

}  // namespace operations::data_movement

constexpr auto sharded_to_interleaved_partial = ttnn::register_operation<
    "ttnn::sharded_to_interleaved_partial",
    ttnn::operations::data_movement::ShardedToInterleavedPartialOperation>();

}  // namespace ttnn
