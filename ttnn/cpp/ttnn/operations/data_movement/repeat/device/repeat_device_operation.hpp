// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/run_operation.hpp"
#include "repeat_operation_types.hpp"
namespace ttnn {

using operation_attributes_t = operations::data_movement::repeat::operation_attributes_t;
using tensor_args_t = operations::data_movement::repeat::tensor_args_t;
using spec_return_value_t = operations::data_movement::repeat::spec_return_value_t;
using tensor_return_value_t = operations::data_movement::repeat::tensor_return_value_t;

struct RepeatDeviceOperation {
    const uint32_t m_num_repeats;
    const bool m_is_last_dim;
    tt::tt_metal::MemoryConfig m_output_mem_config;

    // Required functions to all tensor op functions
    // use for cache hit, reuse for miss
    void validate(const std::vector<Tensor>& input_tensors) const;

    static void validate_on_program_cache_hit(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);
    static void validate_on_program_cache_miss(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);
    // use this one, skip the input
    std::vector<TensorSpec> compute_output_specs(const std::vector<Tensor>& input_tensors) const;
    spec_return_value_t compute_output_specs(
        const operation_attributes_t& operation_attributes, const tensor_args_t& input_tensors) const;
    // need to implement a program factory
    tt::tt_metal::operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const;
    // use this one with appropriate types?
    tt::tt_metal::operation::OpPerformanceModelGeneral<std::vector<Tensor>> create_op_performance_model(
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors,
        std::vector<Tensor>& output_tensors) const;
    // the compiler will take care of invoke
    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const Tensor& input,
        const ttnn::SmallVector<uint32_t>& repetition_vector,
        const uint32_t m_num_repeats,
        const bool m_is_last_dim,
        const tt::tt_metal::MemoryConfig& output_mem_config);
};
}  // namespace ttnn
