// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "interleaved_to_sharded_partial_op_types.hpp"
#include "interleaved_to_sharded_partial_program_factory.hpp"

namespace ttnn::operations::data_movement {

struct InterleavedToShardedPartialDeviceOperation {
    using operation_attributes_t = interleaved_to_sharded_partial::operation_attributes_t;
    using tensor_args_t = interleaved_to_sharded_partial::tensor_args_t;
    using spec_return_value_t = interleaved_to_sharded_partial::spec_return_value_t;
    using tensor_return_value_t = interleaved_to_sharded_partial::tensor_return_value_t;
    using program_factory_t = std::variant<detail::InterleavedToShardedPartialProgramFactory>;

    static program_factory_t select_program_factory(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static void validate_on_program_cache_miss(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static void validate_on_program_cache_hit(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static spec_return_value_t compute_output_specs(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static tt::stl::hash::hash_t compute_program_hash(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const Tensor& input_tensor,
        const CoreCoord& grid_size,
        const tt::tt_metal::ShardSpec& shard_spec,
        uint32_t num_slices,
        uint32_t slice_index,
        const tt::tt_metal::MemoryConfig& output_mem_config,
        const tt::tt_metal::DataType& output_dtype);
};

}  // namespace ttnn::operations::data_movement

namespace ttnn::prim {
constexpr auto interleaved_to_sharded_partial = ttnn::register_operation<
    "ttnn::prim::interleaved_to_sharded_partial",
    ttnn::operations::data_movement::InterleavedToShardedPartialDeviceOperation>();
}  // namespace ttnn::prim
