// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/decorators.hpp"
#include "fast_reduce_nc_device_operation_types.hpp"
#include "fast_reduce_nc_program_factory.hpp"

namespace ttnn::operations::experimental::reduction::detail {

struct FastReduceNCDeviceOperation {
    using operation_attributes_t = detail::operation_attributes_t;
    using tensor_args_t = detail::tensor_args_t;
    using spec_return_value_t = detail::spec_return_value_t;
    using tensor_return_value_t = detail::tensor_return_value_t;
    using program_factory_t = std::variant<program::FastReduceNCProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const Tensor& input,
        const int32_t& dim,
        const std::optional<const Tensor>& output,
        const MemoryConfig& output_mem_config,
        const DeviceComputeKernelConfig& compute_kernel_config);
};

}  // namespace ttnn::operations::experimental::reduction::detail

namespace ttnn::prim {
constexpr auto fast_reduce_nc = ttnn::register_operation<
    "ttnn::prim::fast_reduce_nc",
    ttnn::operations::experimental::reduction::detail::FastReduceNCDeviceOperation>();
}  // namespace ttnn::prim
