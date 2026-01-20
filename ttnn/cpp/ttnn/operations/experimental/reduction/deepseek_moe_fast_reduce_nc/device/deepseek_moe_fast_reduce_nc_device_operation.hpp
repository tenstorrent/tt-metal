// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include "deepseek_moe_fast_reduce_nc_device_operation_types.hpp"
#include "deepseek_moe_fast_reduce_nc_program_factory.hpp"

#include "ttnn/decorators.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::reduction::deepseek_moe_fast_reduce_nc::detail {

struct DeepseekMoEFastReduceNCDeviceOperation {
    using operation_attributes_t = deepseek_moe_fast_reduce_nc::detail::operation_attributes_t;
    using tensor_args_t = deepseek_moe_fast_reduce_nc::detail::tensor_args_t;
    using spec_return_value_t = deepseek_moe_fast_reduce_nc::detail::spec_return_value_t;
    using tensor_return_value_t = deepseek_moe_fast_reduce_nc::detail::tensor_return_value_t;
    using program_factory_t = std::variant<DeepseekMoEFastReduceNCProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::operations::experimental::reduction::deepseek_moe_fast_reduce_nc::detail

namespace ttnn::prim {

ttnn::operations::experimental::reduction::deepseek_moe_fast_reduce_nc::detail::DeepseekMoEFastReduceNCDeviceOperation::
    tensor_return_value_t
    deepseek_moe_fast_reduce_nc(
        const ttnn::Tensor& input_tensor,
        uint32_t reduction_dim,
        uint32_t split_dim,
        const ttnn::MemoryConfig& output_memory_config,
        const ttnn::DeviceComputeKernelConfig& compute_kernel_config);

}  // namespace ttnn::prim
