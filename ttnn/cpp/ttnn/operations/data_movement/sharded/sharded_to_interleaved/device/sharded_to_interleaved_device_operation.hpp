// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/data_movement/sharded/sharded_to_interleaved/device/sharded_to_interleaved_device_operation_types.hpp"
#include "ttnn/operations/data_movement/sharded/sharded_to_interleaved/device/sharded_to_interleaved_program_factory.hpp"
#include "ttnn/decorators.hpp"

namespace ttnn::prim {

struct ShardedToInterleavedDeviceOperation {
    using operation_attributes_t = ShardedToInterleavedParams;
    using tensor_args_t = ShardedToInterleavedInputs;
    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;

    using program_factory_t = std::variant<ShardedToInterleavedProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

    tt::tt_metal::operation::OpPerformanceModelGeneral<tensor_return_value_t> create_op_performance_model(
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& output_tensor) const;
};

Tensor sharded_to_interleaved(
    const Tensor& input_tensor,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const tt::tt_metal::DataType& output_dtype,
    const std::optional<Tensor>& preallocated_output = std::nullopt);

}  // namespace ttnn::prim
