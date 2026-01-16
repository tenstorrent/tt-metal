// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <variant>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/decorators.hpp"

#include "group_attn_matmul_device_operation_types.hpp"
#include "group_attn_matmul_program_factory.hpp"

namespace ttnn::operations::experimental::matmul::group_attn_matmul {

struct GroupAttnMatmulDeviceOperation {
    using operation_attributes_t = group_attn_matmul::operation_attributes_t;
    using tensor_args_t = group_attn_matmul::tensor_args_t;
    using spec_return_value_t = group_attn_matmul::spec_return_value_t;
    using tensor_return_value_t = group_attn_matmul::tensor_return_value_t;
    using program_factory_t = std::variant<program::GroupAttnMatmulProgramFactory>;

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
};

}  // namespace ttnn::operations::experimental::matmul::group_attn_matmul

namespace ttnn::prim {

ttnn::operations::experimental::matmul::group_attn_matmul::GroupAttnMatmulDeviceOperation::tensor_return_value_t
group_attn_matmul(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const CoreCoord& compute_with_storage_grid_size,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<const DataType> output_dtype,
    std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config,
    std::optional<const uint32_t> num_tokens,
    std::optional<const bool> transpose_hw,
    uint32_t out_subblock_w,
    bool row_major,
    std::optional<Tensor> preallocated_output);

}  // namespace ttnn::prim
