// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "sharded_to_interleaved_device_operation_types.hpp"
#include "sharded_to_interleaved_program_factory.hpp"

namespace ttnn::operations::data_movement {

struct ShardedToInterleavedDeviceOperation {
    using operation_attributes_t = ttnn::operations::data_movement::sharded_to_interleaved_operation_attributes_t;
    using tensor_args_t = ttnn::operations::data_movement::sharded_to_interleaved_tensor_args_t;
    using spec_return_value_t = ttnn::operations::data_movement::sharded_to_interleaved_spec_return_value_t;
    using tensor_return_value_t = ttnn::operations::data_movement::sharded_to_interleaved_tensor_return_value_t;

    using program_factory_t = std::variant<program::ShardedToInterleavedProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const Tensor& input_tensor,
        const tt::tt_metal::MemoryConfig& output_mem_config,
        const tt::tt_metal::DataType& output_dtype,
        bool is_l1_aligned = false,
        const std::optional<Tensor>& preallocated_output = std::nullopt);
};

}  // namespace ttnn::operations::data_movement

namespace ttnn::prim {
constexpr auto sharded_to_interleaved = ttnn::register_operation<
    "ttnn::prim::sharded_to_interleaved",
    ttnn::operations::data_movement::ShardedToInterleavedDeviceOperation>();
}  // namespace ttnn::prim
