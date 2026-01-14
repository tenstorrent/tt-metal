// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/device_operation.hpp"

#include "dit_layernorm_post_all_gather_device_operation_types.hpp"
#include "dit_layernorm_post_all_gather_welford_program_factory.hpp"

namespace ttnn::operations::experimental::transformer::dit_layernorm {

struct PostAllGatherDeviceOperation {
    using operation_attributes_t = PostAllGatherOperationAttributes;
    using tensor_args_t = PostAllGatherTensorArgs;
    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<program::PostAllGatherWelfordProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::operations::experimental::transformer::dit_layernorm

namespace ttnn::prim {

Tensor dit_layernorm_post_all_gather(
    const Tensor& input,
    const Tensor& stats,
    float eps,
    const std::optional<const Tensor>& gamma,
    const std::optional<const Tensor>& beta,
    const MemoryConfig& memory_config,
    const DeviceComputeKernelConfig& compute_kernel_config,
    const std::optional<DataType>& dtype);

}  // namespace ttnn::prim
