// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <optional>

#include "tensor/tensor.hpp"
// #include "third_party/magic_enum/magic_enum.hpp"
// #include "tt_eager/tensor/owned_buffer_functions.hpp"
// #include "tt_eager/tensor/tensor_utils.hpp"
#include "tt_eager/tt_dnn/op_library/compute_kernel_config.hpp"
#include "tt_eager/tt_dnn/op_library/copy/copy_op.hpp"
#include "tt_eager/tt_dnn/op_library/run_operation.hpp"
#include "tt_eager/tt_dnn/op_library/sharded/sharded_op.hpp"
// #include "tt_metal/host_api.hpp"
// #include "tt_metal/impl/dispatch/command_queue.hpp"
// #include "ttnn/core.hpp"
#include "ttnn/operations/core.hpp"
#include "ttnn/types.hpp"

namespace ttnn {

namespace operations {

namespace core {

struct ToMemoryConfig {
    static inline const std::array<TensorSchema, 1> input_tensor_schemas() {
        return {ttnn::TensorSchema{
            1,
            4,
            {ttnn::bfloat16, ttnn::bfloat8_b, ttnn::bfloat4_b, ttnn::float32, ttnn::uint16, ttnn::uint32, ttnn::int32},
            {ttnn::ROW_MAJOR_LAYOUT, ttnn::TILE_LAYOUT},
            true,
            true,
            false,
            false}};
    }

    template <typename... Args>
    static auto input_tensors_to_validate(const Tensor& tensor_arg, Args&&... args) {
        return std::make_tuple(tensor_arg);
    };

    template <typename... Args>
    static Tensor create_async_output_tensors(const ttnn::Tensor& tensor_arg, Args&&... args) {
        return {{Tensor(operation::get_workers_for_op_output({tensor_arg}))}};
    }

    // TODO: Move to cpp once we merge with tt_eager
    static Tensor execute(
        const ttnn::Tensor tensor,
        const ttnn::MemoryConfig& memory_config,
        std::optional<ttnn::DataType> dtype) {
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
                        throw runtime_error(
                            "dtype cannot be specified when converting sharded tensor to sharded tensor");
                    }
                    TT_FATAL(tensor.shard_spec().has_value());
                    TT_FATAL(memory_config.is_sharded());
                    return operation::run(
                               Reshard{
                                   .output_mem_config = memory_config,
                               },
                               {tensor})
                        .at(0);
                } else {
                    // for row-major tensors where shard-spec[1] is different for input shard and output shard

                    TT_FATAL(memory_config.is_sharded());
                    Tensor temp = operation::run(
                                      Sharded{
                                          .grid_size = tensor.device()->compute_with_storage_grid_size(),
                                          .shard_spec = input_shard_spec,
                                          .sharded_op_type = ShardedOpType::ShardedToInterleaved,
                                          .output_mem_config = ttnn::DRAM_MEMORY_CONFIG,
                                          .output_dtype = dtype.value_or(tensor.get_dtype())},
                                      {tensor})
                                      .at(0);
                    return operation::run(
                               Sharded{
                                   .grid_size = temp.device()->compute_with_storage_grid_size(),
                                   .shard_spec = output_shard_spec,
                                   .sharded_op_type = ShardedOpType::InterleavedToSharded,
                                   .output_mem_config = memory_config,
                                   .output_dtype = dtype.value_or(temp.get_dtype())},
                               {temp})
                        .at(0);
                }
            } else {
                TT_FATAL(memory_config.is_sharded());
                auto bbox = memory_config.shard_spec.value().grid.bounding_box();
                CoreCoord grid_size(bbox.end.x + 1, bbox.end.y + 1);
                return operation::run(
                           Sharded{
                               .grid_size = grid_size,
                               .shard_spec = memory_config.shard_spec.value(),
                               .sharded_op_type = ShardedOpType::InterleavedToSharded,
                               .output_mem_config = memory_config,
                               .output_dtype = dtype.value_or(tensor.get_dtype())},
                           {tensor})
                    .at(0);
            }
        } else {
            // to_interleaved path
            if (tensor.is_sharded()) {
                TT_FATAL(tensor.shard_spec().has_value());
                auto shard_spec = tensor.shard_spec().value();
                return operation::run(
                           Sharded{
                               .grid_size = tensor.device()->compute_with_storage_grid_size(),
                               .shard_spec = shard_spec,
                               .sharded_op_type = ShardedOpType::ShardedToInterleaved,
                               .output_mem_config = memory_config,
                               .output_dtype = dtype.value_or(tensor.get_dtype())},
                           {tensor})
                    .at(0);
            } else {
                // L1 to DRAM or DRAM to L1
                return operation::run(Copy{memory_config, dtype.value_or(tensor.get_dtype())}, {tensor}).at(0);
            }
        }

        return tensor;
    }
};

}  // namespace core
}  // namespace operations
}  // namespace ttnn
