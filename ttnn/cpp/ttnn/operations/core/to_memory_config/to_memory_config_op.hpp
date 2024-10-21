// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/cpp/ttnn/operations/data_movement/sharded/reshard/device/reshard_op.hpp"
#include "ttnn/cpp/ttnn/operations/data_movement/sharded/interleaved_to_sharded/device/interleaved_to_sharded_op.hpp"
#include "ttnn/cpp/ttnn/operations/data_movement/sharded/sharded_to_interleaved/device/sharded_to_interleaved_op.hpp"
#include "ttnn/types.hpp"
#include "ttnn/cpp/ttnn/operations/data_movement/copy/device/copy_device_operation.hpp"

namespace ttnn {

namespace operations {

namespace core {

struct ToMemoryConfig {

    // TODO: Move to cpp once we merge with tt_eager
    static Tensor invoke(
        const ttnn::Tensor& tensor, const ttnn::MemoryConfig& memory_config, std::optional<ttnn::DataType> dtype = std::nullopt) {
        // Temporary until we see why buffer data not being populated
        const auto original_shape = tensor.get_shape();

        const auto original_memory_config = ttnn::get_memory_config(tensor);
        if (original_memory_config.has_value() && original_memory_config.value() == memory_config) {
            return tensor;
        }

        if (memory_config.is_sharded()) {
            // to_sharded path
            if (tensor.is_sharded()) {
                // reshard
                const auto input_memory_config = ttnn::get_memory_config(tensor);
                const auto input_shard_spec = input_memory_config.value().shard_spec.value();
                const auto output_shard_spec = memory_config.shard_spec.value();
                if (tensor.get_layout() == ttnn::TILE_LAYOUT ||
                    input_shard_spec.shape[1] == output_shard_spec.shape[1]) {
                    if (dtype.has_value()) {
                        throw std::runtime_error(
                            "dtype cannot be specified when converting sharded tensor to sharded tensor");
                    }
                    return operation::run(
                               data_movement::ReshardDeviceOperation{
                                   .output_mem_config = memory_config,
                               },
                               {tensor}, {}, {std::nullopt})
                        .at(0);
                } else {
                    // for row-major tensors where shard-spec[1] is different for input shard and output shard

                    TT_FATAL(memory_config.is_sharded(), "Error");
                    Tensor temp = operation::run(
                                      data_movement::ShardedToInterleavedDeviceOperation{
                                          .output_mem_config = ttnn::DRAM_MEMORY_CONFIG,
                                          .output_dtype = dtype.value_or(tensor.get_dtype())},
                                      {tensor}
                                      )
                                      .at(0);
                    return operation::run(
                               data_movement::InterleavedToShardedDeviceOperation{
                                   .output_mem_config = memory_config,
                                   .output_dtype = dtype.value_or(temp.get_dtype())},
                               {temp})
                        .at(0);
                }
            } else {
                auto bbox = memory_config.shard_spec.value().grid.bounding_box();
                CoreCoord grid_size(bbox.end_coord.x + 1, bbox.end_coord.y + 1);
                return operation::run(
                           data_movement::InterleavedToShardedDeviceOperation{
                               .output_mem_config = memory_config,
                               .output_dtype = dtype.value_or(tensor.get_dtype())},
                           {tensor})
                    .at(0);
            }
        } else {
            // to_interleaved path
            if (tensor.is_sharded()) {
                return operation::run(
                           data_movement::ShardedToInterleavedDeviceOperation{
                               .output_mem_config = memory_config,
                               .output_dtype = dtype.value_or(tensor.get_dtype())},
                           {tensor})
                    .at(0);
            } else {
                // L1 to DRAM or DRAM to L1
                return operation::run(ttnn::operations::data_movement::CopyDeviceOperation{memory_config, dtype.value_or(tensor.get_dtype())}, {tensor}).at(0);
            }
        }

        return tensor;
    }
};

}  // namespace core
}  // namespace operations
}  // namespace ttnn
