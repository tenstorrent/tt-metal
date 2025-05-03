// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"

namespace ttnn {
namespace operations::data_movement {

struct ShardedToInterleavedOperation {
    static ttnn::Tensor invoke(
        QueueId queue_id,
        const ttnn::Tensor& input_tensor,
        const MemoryConfig& memory_config,
        const std::optional<DataType>& output_dtype,
        const std::optional<bool>& is_l1_aligned = std::nullopt);
};

}  // namespace operations::data_movement

constexpr auto sharded_to_interleaved = ttnn::register_operation<
    "ttnn::sharded_to_interleaved",
    ttnn::operations::data_movement::ShardedToInterleavedOperation>();

}  // namespace ttnn
