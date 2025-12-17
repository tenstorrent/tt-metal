// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>
#include "ttnn/decorators.hpp"

#include "ttnn/tensor/types.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "interleaved_to_sharded_op_types.hpp"
#include "interleaved_to_sharded_program_factory.hpp"

namespace ttnn::operations::data_movement {

struct InterleavedToShardedDeviceOperation {
    using operation_attributes_t = interleaved_to_sharded::operation_attributes_t;
    using tensor_args_t = interleaved_to_sharded::tensor_args_t;
    using spec_return_value_t = interleaved_to_sharded::spec_return_value_t;
    using tensor_return_value_t = interleaved_to_sharded::tensor_return_value_t;

    using program_factory_t = std::variant<interleaved_to_sharded::InterleavedToShardedProgramFactory>;

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
        const tt::tt_metal::MemoryConfig& output_mem_config,
        const tt::tt_metal::DataType& output_dtype,
        bool keep_l1_aligned,
        const std::optional<Tensor>& preallocated_output = std::nullopt);
};

}  // namespace ttnn::operations::data_movement

namespace ttnn::prim {
constexpr auto interleaved_to_sharded = ttnn::register_operation<
    "ttnn::prim::interleaved_to_sharded",
    ttnn::operations::data_movement::InterleavedToShardedDeviceOperation>();
}
