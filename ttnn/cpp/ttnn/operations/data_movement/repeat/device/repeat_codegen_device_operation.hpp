// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operation.hpp"
#include "ttnn/operations/data_movement/repeat/device/repeat_codegen_device_operation_types.hpp"
#include "ttnn/operations/data_movement/repeat/device/repeat_codegen_program_factory.hpp"
#include "ttnn/types.hpp"

namespace ttnn::prim {

struct RepeatCodegenDeviceOperation {
    using operation_attributes_t = RepeatCodegenParams;
    using tensor_args_t = RepeatCodegenInputs;
    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<RepeatCodegenProgramFactory>;

    static program_factory_t select_program_factory(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);
    static void validate_on_program_cache_miss(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);
    static spec_return_value_t compute_output_specs(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);
    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);
};

// Primitive entry: precomputed tile page counts in, launched device op out.
RepeatCodegenDeviceOperation::tensor_return_value_t repeat_codegen(
    const Tensor& input,
    uint32_t num_repeats,
    int32_t repeat_dim,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    uint32_t tile_higher_pages,
    uint32_t tile_rep_dim_pages,
    uint32_t tile_lower_pages,
    uint32_t tile_page_size_bytes);

}  // namespace ttnn::prim
