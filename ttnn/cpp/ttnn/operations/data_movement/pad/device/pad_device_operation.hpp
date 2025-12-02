// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/cpp/ttnn/operations/data_movement/pad/device/pad_device_operation_types.hpp"

namespace ttnn::operations::data_movement {

struct PadDeviceOperation {
    using operation_attributes_t = pad::operation_attributes_t;
    using tensor_args_t = pad::tensor_args_t;
    using spec_return_value_t = pad::spec_return_value_t;
    using tensor_return_value_t = pad::tensor_return_value_t;
    using program_factory_t = std::variant<program::PadProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const Tensor& input,
        const ttnn::Shape& output_logical_shape,
        const ttnn::Shape& output_padded_shape,
        const ttnn::Shape& input_tensor_start,
        const float pad_value,
        const tt::tt_metal::MemoryConfig& output_mem_config,
        const bool use_multicore,
        const std::optional<Tensor>& preallocated_output);
};

namespace ttnn::prim {
constexpr auto pad = ttnn::register_operation<"ttnn::prim::pad", ttnn::operations::data_movement::PadDeviceOperation>();
}  // namespace ttnn::prim
