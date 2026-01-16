// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/device_operation.hpp"

#include "dit_layernorm_pre_all_gather_device_operation_types.hpp"
#include "dit_layernorm_pre_all_gather_welford_program_factory.hpp"

namespace ttnn::operations::experimental::transformer::dit_layernorm {

struct PreAllGatherDeviceOperation {
    using operation_attributes_t = PreAllGatherOperationAttributes;
    using tensor_args_t = PreAllGatherTensorArgs;
    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<program::PreAllGatherWelfordProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::operations::experimental::transformer::dit_layernorm

namespace ttnn::prim {

Tensor dit_layernorm_pre_all_gather(
    const Tensor& input,
    const std::optional<tt::tt_metal::DataType>& dtype,
    const DeviceComputeKernelConfig& compute_kernel_config,
    const tt::tt_metal::MemoryConfig& memory_config);

}  // namespace ttnn::prim
