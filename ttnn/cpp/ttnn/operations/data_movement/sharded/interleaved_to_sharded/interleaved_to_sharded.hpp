// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/types.hpp"
#include <tt-metalium/core_coord.hpp>

namespace ttnn {

ttnn::Tensor interleaved_to_sharded(
    const ttnn::Tensor& input_tensor,
    const MemoryConfig& sharded_memory_config,
    const std::optional<DataType>& data_type_arg = std::nullopt,
    const std::optional<bool>& keep_l1_aligned = std::nullopt,
    const std::optional<Tensor>& preallocated_output = std::nullopt);

ttnn::Tensor interleaved_to_sharded(
    const ttnn::Tensor& input_tensor,
    const std::variant<CoreCoord, CoreRangeSet>& grid,
    std::array<uint32_t, 2> shard_shape,
    TensorMemoryLayout shard_scheme,
    tt::tt_metal::ShardOrientation shard_orientation,
    const std::optional<DataType>& data_type_arg = std::nullopt,
    const std::optional<bool>& keep_l1_aligned = std::nullopt);

}  // namespace ttnn
