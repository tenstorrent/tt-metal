// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <variant>

#include "ttnn/device_operation.hpp"
#include "ttnn/metal_v2_artifacts.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/types.hpp"

namespace ttnn::experimental::llama31_8b {

struct RotaryBf8ProgramFactory;

struct RotaryBf8Operation {
    struct operation_attributes_t {
        DataType output_dtype = DataType::BFLOAT8_B;
        MemoryConfig output_memory_config;
        DeviceComputeKernelConfig compute_kernel_config;
        std::uint32_t table_row = 0;
    };

    struct tensor_args_t {
        const Tensor& input;
        const Tensor& cos_cache;
        const Tensor& sin_cache;
        const Tensor& trans_mat;
    };

    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<RotaryBf8ProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

struct RotaryBf8ProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const RotaryBf8Operation::operation_attributes_t& operation_attributes,
        const RotaryBf8Operation::tensor_args_t& tensor_args,
        RotaryBf8Operation::tensor_return_value_t& output);
};

}  // namespace ttnn::experimental::llama31_8b

namespace ttnn::prim {
ttnn::Tensor llama31_8b_rotary_bf8(
    const Tensor& input,
    const Tensor& cos_cache,
    const Tensor& sin_cache,
    const Tensor& trans_mat,
    DataType output_dtype,
    const MemoryConfig& output_memory_config,
    const DeviceComputeKernelConfig& compute_kernel_config,
    std::uint32_t table_row);
}  // namespace ttnn::prim
