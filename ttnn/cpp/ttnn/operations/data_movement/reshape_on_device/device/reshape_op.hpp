// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/decorators.hpp"
#include "reshape_device_operation_types.hpp"
#include "reshape_tile_program_factory.hpp"
#include "reshape_rm_program_factory.hpp"

namespace ttnn::prim {

struct ReshapeDeviceOperation {
    using operation_attributes_t = ReshapeOnDeviceParams;
    using tensor_args_t = ReshapeOnDeviceInputs;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = tt::tt_metal::Tensor;
    using program_factory_t = std::variant<ttnn::prim::ReshapeTileProgramFactory, ttnn::prim::ReshapeRMProgramFactory>;

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
};

tt::tt_metal::Tensor reshape_on_device(
    const Tensor& input_tensor,
    const tt::tt_metal::Shape& logical_output_shape,
    const tt::tt_metal::Shape& padded_output_shape,
    const tt::tt_metal::MemoryConfig& output_mem_config);

}  // namespace ttnn::prim
