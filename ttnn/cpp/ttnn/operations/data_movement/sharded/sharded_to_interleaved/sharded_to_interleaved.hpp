// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"

namespace ttnn {
namespace operations::data_movement {

struct ShardedToInterleavedOperation {
    static ttnn::Tensor invoke(uint8_t queue_id,
                               const ttnn::Tensor& input_tensor,
                               const MemoryConfig& memory_config,
                               const std::optional<DataType>& output_dtype);
};

}  // namespace operations::data_movement

constexpr auto sharded_to_interleaved =
    ttnn::register_operation_with_auto_launch_op<"ttnn::sharded_to_interleaved",
                                                 ttnn::operations::data_movement::ShardedToInterleavedOperation>();

}  // namespace ttnn
