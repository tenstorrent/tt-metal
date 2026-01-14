// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/decorators.hpp"
#include "bcast_device_operation_types.hpp"
#include "bcast_multi_core_h_program_factory.hpp"
#include "bcast_sharded_h_program_factory.hpp"
#include "bcast_sharded_h_optimised_program_factory.hpp"
#include "bcast_multi_core_w_program_factory.hpp"
#include "bcast_multi_core_hw_program_factory.hpp"

namespace ttnn::operations::data_movement::bcast {

struct BcastDeviceOperation {
    using operation_attributes_t = bcast::operation_attributes_t;
    using tensor_args_t = bcast::tensor_args_t;
    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<
        program::BcastMultiCoreHProgramFactory,
        program::BcastShardedHProgramFactory,
        program::BcastShardedHOptimisedProgramFactory,
        program::BcastMultiCoreWProgramFactory,
        program::BcastMultiCoreHWProgramFactory>;

    static program_factory_t select_program_factory(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static void validate_on_program_cache_hit(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static void validate_on_program_cache_miss(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static spec_return_value_t compute_output_specs(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static tt::stl::hash::hash_t compute_program_hash(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static tt::tt_metal::operation::OpPerformanceModelGeneral<tensor_return_value_t> create_op_performance_model(
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value);
};

}  // namespace ttnn::operations::data_movement::bcast

namespace ttnn::prim {
ttnn::operations::data_movement::bcast::BcastDeviceOperation::tensor_return_value_t bcast(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    ttnn::BcastOpMath bcast_op,
    ttnn::BcastOpDim bcast_dim,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    bool in_place,
    const std::optional<Tensor>& preallocated_output);
}  // namespace ttnn::prim
