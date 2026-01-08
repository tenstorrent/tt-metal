// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "reduce_avg_w_rm_program_factory.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/decorators.hpp"
#include "reduce_avg_w_rm_device_operation_types.hpp"

namespace ttnn::operations::reduction::reduce_avg_w_rm {

struct ReduceAvgWRmDeviceOperation {
    using operation_attributes_t = reduce_avg_w_rm::operation_attributes_t;
    using tensor_args_t = reduce_avg_w_rm::tensor_args_t;
    using spec_return_value_t = reduce_avg_w_rm::spec_return_value_t;
    using tensor_return_value_t = reduce_avg_w_rm::tensor_return_value_t;
    using program_factory_t = std::variant<program::ReduceAvgWRmProgramFactory>;

    // ALL STATIC FUNCTIONS - This is the modern pattern!
    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes, const tensor_args_t&);
    static tt::stl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::operations::reduction::reduce_avg_w_rm

namespace ttnn::prim {
// Primitive operation - free function that calls launch_on_device
ttnn::operations::reduction::reduce_avg_w_rm::ReduceAvgWRmDeviceOperation::tensor_return_value_t reduce_avg_w_rm(
    const Tensor& input,
    std::optional<MemoryConfig> output_mem_config,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config);
}  // namespace ttnn::prim
