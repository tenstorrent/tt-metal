// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>
#include "ttnn/decorators.hpp"

#include "ttnn/tensor/types.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "copy_to_memory_config_device_operation_types.hpp"
#include "copy_to_memory_config_row_major_default_program_factory.hpp"
#include "copy_to_memory_config_tilized_default_program_factory.hpp"

namespace ttnn::prim {

struct CopyToMemoryConfigDeviceOperation {
    using operation_attributes_t = CopyToMemoryConfigOperationAttributes;
    using tensor_args_t = CopyToMemoryConfigTensorArgs;
    using spec_return_value_t = CopyToMemoryConfigSpecReturnValue;
    using tensor_return_value_t = CopyToMemoryConfigTensorReturnValue;

    using program_factory_t =
        std::variant<CopyToMemoryConfigRowMajorDefaultProgramFactory, CopyToMemoryConfigTilizedDefaultProgramFactory>;
    static program_factory_t select_program_factory(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static void validate_on_program_cache_miss(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static spec_return_value_t compute_output_specs(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static ttsl::hash::hash_t compute_program_hash(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);
};

Tensor copy_to_memory_config(
    const Tensor& input_tensor,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const tt::tt_metal::DataType& output_dtype,
    const std::optional<Tensor>& preallocated_output = std::nullopt);
}  // namespace ttnn::prim
