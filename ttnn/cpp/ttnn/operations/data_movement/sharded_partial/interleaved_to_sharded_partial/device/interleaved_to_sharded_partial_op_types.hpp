// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"
#include <tt-metalium/core_coord.hpp>

namespace ttnn::operations::data_movement::interleaved_to_sharded_partial {

struct operation_attributes_t {
    tt::tt_metal::CoreCoord grid_size;
    tt::tt_metal::ShardSpec shard_spec = tt::tt_metal::ShardSpec(tt::tt_metal::CoreRangeSet(), {0, 0});
    uint32_t num_slices{};
    uint32_t slice_index{};
    tt::tt_metal::MemoryConfig output_mem_config;
    tt::tt_metal::DataType output_dtype{tt::tt_metal::DataType::INVALID};
};

struct tensor_args_t {
    tt::tt_metal::Tensor input_tensor;
};

using spec_return_value_t = TensorSpec;
using tensor_return_value_t = Tensor;

}  // namespace ttnn::operations::data_movement::interleaved_to_sharded_partial
