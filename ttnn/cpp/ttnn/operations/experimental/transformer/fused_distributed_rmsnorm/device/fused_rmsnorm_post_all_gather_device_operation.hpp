// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/decorators.hpp"

#include "fused_rmsnorm_post_all_gather_device_operation_types.hpp"
#include "fused_rmsnorm_post_all_gather_program_factory.hpp"

namespace ttnn::operations::experimental::transformer::fused_rmsnorm_post_all_gather {

struct FusedRMSNormPostAllGatherDeviceOperation {
    using operation_attributes_t = FusedRmsnormPostAllGatherParams;
    using tensor_args_t = FusedRmsnormPostAllGatherInputs;
    using spec_return_value_t = fused_rmsnorm_post_all_gather::spec_return_value_t;
    using tensor_return_value_t = fused_rmsnorm_post_all_gather::tensor_return_value_t;
    using program_factory_t = std::variant<program::FusedRMSNormPostAllGatherProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(const operation_attributes_t& args, const tensor_args_t&);
};

}  // namespace ttnn::operations::experimental::transformer::fused_rmsnorm_post_all_gather

namespace ttnn::prim {
ttnn::operations::experimental::transformer::fused_rmsnorm_post_all_gather::tensor_return_value_t
fused_rmsnorm_post_all_gather(
    const Tensor& input_tensor,
    const Tensor& stats_tensor,
    float eps,
    uint32_t num_heads,
    const std::optional<const Tensor>& weight,
    const std::optional<const Tensor>& transformation_mat,
    const std::optional<const Tensor>& rope_cos,
    const std::optional<const Tensor>& rope_sin,
    const MemoryConfig& memory_config,
    const DeviceComputeKernelConfig& compute_kernel_config,
    const std::optional<DataType>& dtype);
}  // namespace ttnn::prim
