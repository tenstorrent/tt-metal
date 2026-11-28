// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/core/to_memory_config/to_memory_config_op.hpp"
#include "ttnn/operations/data_movement/sharded/sharded_to_interleaved/sharded_to_interleaved.hpp"
#include "device/sharded_to_interleaved_device_operation.hpp"

namespace ttnn {

ttnn::Tensor sharded_to_interleaved(
    const ttnn::Tensor& input_tensor, const MemoryConfig& memory_config, const std::optional<DataType>& output_dtype) {
    // The early return is for a tensor that is already interleaved, which is_sharded() answers directly.
    // Testing shard_spec() instead silently no-ops on ND sharding: NdShardSpec does not always fold into a
    // legacy 2-D shard_spec() (see TensorSpec::populate_legacy_shard_spec_from_nd), so shard_spec() is empty
    // for ROUND_ROBIN_1D with more shards than cores, CONTIGUOUS_1D, and 3-D shard shapes.
    if (!input_tensor.is_sharded()) {
        return input_tensor;
    }

    // A sharded tensor with no legacy 2-D shard_spec() is ND sharding, which the device op does not walk
    // and whose own validation rejects it. Route through the same path ttnn::to_memory_config takes for
    // those, which converts by value and honours both memory_config and output_dtype. Note that every
    // legacy-sharded tensor also carries an nd_shard_spec (populate_nd_shard_spec_from_legacy), so the
    // discriminator is the legacy shard_spec(), not nd_shard_spec().
    if (!input_tensor.shard_spec().has_value()) {
        return ttnn::to_memory_config(input_tensor, memory_config, output_dtype.value_or(input_tensor.dtype()));
    }

    return ttnn::prim::sharded_to_interleaved(input_tensor, memory_config, output_dtype.value_or(input_tensor.dtype()));
}

}  // namespace ttnn
