// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include "deepseek_moe_fast_reduce_nc_fused_device_operation_types.hpp"
#include "deepseek_moe_fast_reduce_nc_fused_program_factory.hpp"

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

struct DeepseekMoEFastReduceNCFusedDeviceOperation {
    using operation_attributes_t = DeepseekMoEFastReduceNCFusedParams;
    using tensor_args_t = DeepseekMoEFastReduceNCFusedInputs;
    using spec_return_value_t = ttnn::TensorSpec;
    using tensor_return_value_t = std::vector<ttnn::Tensor>;
    using program_factory_t = std::variant<DeepseekMoEFastReduceNCFusedProgramFactory>;

    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

std::vector<ttnn::Tensor> deepseek_moe_fast_reduce_nc_fused(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& scores_tensor,
    uint32_t reduce_dim,
    uint64_t split_size,
    const tt::tt_metal::MemoryConfig& output_memory_config,
    const ttnn::DeviceComputeKernelConfig& compute_kernel_config);

}  // namespace ttnn::prim
