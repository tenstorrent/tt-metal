// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>
#include "ttnn/decorators.hpp"

#include "ttnn/tensor/types.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "redistribute_to_memory_config_device_operation_types.hpp"
#include "redistribute_to_memory_config_row_major_sharded_program_factory.hpp"
#include "redistribute_to_memory_config_tilized_sharded_program_factory.hpp"

namespace ttnn::prim {

struct RedistributeToMemoryConfigDeviceOperation {
    using operation_attributes_t = operation_attributes_t;
    using tensor_args_t = tensor_args_t;
    using spec_return_value_t = spec_return_value_t;
    using tensor_return_value_t = tensor_return_value_t;

    using program_factory_t =
        std::variant<RedistributeToMemoryConfigRowMajorProgramFactory, RedistributeToMemoryConfigTilizedProgramFactory>;
    static program_factory_t select_program_factory(
        const ::ttnn::prim::operation_attributes_t&, const ::ttnn::prim::tensor_args_t&);

    static void validate_on_program_cache_miss(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static spec_return_value_t compute_output_specs(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static ttsl::hash::hash_t compute_program_hash(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);
};

tensor_return_value_t redistribute_to_memory_config(
    const Tensor& input_tensor,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const tt::tt_metal::DataType& output_dtype,
    const std::optional<Tensor>& preallocated_output = std::nullopt);
}  // namespace ttnn::prim
