// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/data_movement/sharded_partial/sharded_to_interleaved_partial/device/sharded_to_interleaved_partial_device_operation_types.hpp"
#include "ttnn/operations/data_movement/sharded_partial/sharded_to_interleaved_partial/device/sharded_to_interleaved_partial_program_factory.hpp"
#include "ttnn/decorators.hpp"

namespace ttnn::prim {

struct ShardedToInterleavedPartialDeviceOperation {
    using operation_attributes_t = ShardedToInterleavedPartialParams;
    using tensor_args_t = ShardedToInterleavedPartialInputs;
    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;

    using program_factory_t = std::variant<ShardedToInterleavedPartialProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static TensorSpec compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static Tensor create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

    tt::tt_metal::operation::OpPerformanceModelGeneral<Tensor> create_op_performance_model(
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        Tensor& output_tensor) const;
};

Tensor sharded_to_interleaved_partial(
    const Tensor& input_tensor,
    const Tensor& cache_tensor,
    uint32_t num_slices,
    uint32_t slice_index,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const tt::tt_metal::DataType& output_dtype);
}  // namespace ttnn::prim
