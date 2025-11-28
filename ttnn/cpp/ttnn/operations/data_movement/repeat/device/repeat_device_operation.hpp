// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/run_operation.hpp"
#include "ttnn/decorators.hpp"
#include "repeat_operation_types.hpp"
#include "host/repeat_program_factory.hpp"

namespace ttnn::operations::data_movement::repeat {

struct RepeatDeviceOperation {
    using operation_attributes_t = operation_attributes_t;
    using tensor_args_t = tensor_args_t;
    using spec_return_value_t = spec_return_value_t;
    using tensor_return_value_t = tensor_return_value_t;
    using program_factory_t =
        std::variant<program::RepeatProgramFactorySecondDim, program::RepeatProgramFactoryLastDim>;

    static program_factory_t select_program_factory(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static void validate_on_program_cache_hit(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);
    static void validate_on_program_cache_miss(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    // use this one, skip the input
    static spec_return_value_t compute_output_specs(
        const operation_attributes_t& operation_attributes, const tensor_args_t& input_tensors);
    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes, const tensor_args_t& input_tensors);

    tt::tt_metal::operation::OpPerformanceModelGeneral<std::vector<Tensor>> create_op_performance_model(
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& output_tensor) const;

    // the compiler will take care of invoke
    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const Tensor& input,
        const ttnn::SmallVector<uint32_t>& repetition_vector,
        const uint32_t m_num_repeats,
        const bool m_is_last_dim,
        const tt::tt_metal::MemoryConfig& output_mem_config);
};
}  // namespace ttnn::operations::data_movement::repeat

namespace ttnn::prim {
constexpr auto repeat =
    ttnn::register_operation<"ttnn::prim::repeat", ttnn::operations::data_movement::repeat::RepeatDeviceOperation>();
}  // namespace ttnn::prim
