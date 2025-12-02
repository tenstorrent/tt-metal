// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <variant>

#include "ttnn/decorators.hpp"
#include "ttnn/tensor/tensor.hpp"

#include "reduce_op_device_operation_types.hpp"
#include "reduce_op_program_factory.hpp"

namespace ttnn::operations::reduction::generic {

struct ReduceDeviceOperation {
    using operation_attributes_t = operation_attributes_t;
    using tensor_args_t = tensor_args_t;
    using spec_return_value_t = ReduceSpecReturnValue;
    using tensor_return_value_t = ReduceTensorReturnValue;

    using program_factory_t = std::variant<
        program::ReduceSingleCoreHwProgramFactory,
        program::ReduceMultiCoreHProgramFactory,
        program::ReduceMultiCoreWProgramFactory>;

    static program_factory_t select_program_factory(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const Tensor& input_tensor,
        tt::tt_metal::ReduceOpMath reduce_math,
        tt::tt_metal::ReduceOpDim reduce_dim,
        float scaler,
        const MemoryConfig& output_mem_config,
        const std::optional<DataType>& output_dtype,
        const ttnn::DeviceComputeKernelConfig& compute_kernel_config,
        const std::optional<CoreRangeSet>& sub_core_grids);

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
};

}  // namespace ttnn::operations::reduction::generic

namespace ttnn::prim {
constexpr auto reduce =
    ttnn::register_operation<"ttnn::prim::reduce", ttnn::operations::reduction::generic::ReduceDeviceOperation>();
}  // namespace ttnn::prim
