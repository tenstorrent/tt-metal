// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/decorators.hpp"

#include "fused_rmsnorm_pre_all_gather_device_operation_types.hpp"
#include "fused_rmsnorm_pre_all_gather_program_factory.hpp"

namespace ttnn::operations::experimental::transformer::fused_rmsnorm_pre_all_gather {

struct FusedRMSNormPreAllGatherDeviceOperation {
    using operation_attributes_t = FusedRmsnormPreAllGatherParams;
    using tensor_args_t = FusedRmsnormPreAllGatherInputs;
    using spec_return_value_t = fused_rmsnorm_pre_all_gather::spec_return_value_t;
    using tensor_return_value_t = fused_rmsnorm_pre_all_gather::tensor_return_value_t;
    using program_factory_t = std::variant<program::FusedRMSNormPreAllGatherProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(const operation_attributes_t& args, const tensor_args_t&);
};

}  // namespace ttnn::operations::experimental::transformer::fused_rmsnorm_pre_all_gather

namespace ttnn::prim {
ttnn::operations::experimental::transformer::fused_rmsnorm_pre_all_gather::tensor_return_value_t
fused_rmsnorm_pre_all_gather(
    const Tensor& input_tensor, tt::tt_metal::DataType dtype, const DeviceComputeKernelConfig& compute_kernel_config);
}  // namespace ttnn::prim
