// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/data_movement/sharded_partial/sharded_to_interleaved_partial/device/sharded_to_interleaved_partial_device_operation_types.hpp"
#include "ttnn/operations/data_movement/sharded_partial/sharded_to_interleaved_partial/device/sharded_to_interleaved_partial_program_factory.hpp"
#include "ttnn/decorators.hpp"

namespace ttnn::operations::data_movement {

struct ShardedToInterleavedPartialDeviceOperation {
    using operation_attributes_t =
        ttnn::operations::data_movement::sharded_to_interleaved_partial_operation_attributes_t;
    using tensor_args_t = ttnn::operations::data_movement::sharded_to_interleaved_partial_tensor_args_t;
    using spec_return_value_t = ttnn::operations::data_movement::partial_spec_return_value_t;
    using tensor_return_value_t = ttnn::operations::data_movement::partial_tensor_return_value_t;

    using program_factory_t = std::variant<program::ShardedToInterleavedPartialProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

    tt::tt_metal::operation::OpPerformanceModelGeneral<tensor_return_value_t> create_op_performance_model(
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& output_tensor) const;

    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const Tensor& input_tensor,
        const Tensor& cache_tensor,
        uint32_t num_slices,
        uint32_t slice_index,
        const tt::tt_metal::MemoryConfig& output_mem_config,
        const tt::tt_metal::DataType& output_dtype);
};

}  // namespace ttnn::operations::data_movement

namespace ttnn::prim {
constexpr auto sharded_to_interleaved_partial = ttnn::register_operation<
    "ttnn::prim::sharded_to_interleaved_partial",
    ttnn::operations::data_movement::ShardedToInterleavedPartialDeviceOperation>();
}  // namespace ttnn::prim
