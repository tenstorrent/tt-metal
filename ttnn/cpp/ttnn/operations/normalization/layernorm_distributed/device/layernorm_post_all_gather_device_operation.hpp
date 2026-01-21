// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "layernorm_post_all_gather_program_factory.hpp"

#include "layernorm_post_all_gather_device_operation_types.hpp"

namespace ttnn::prim {

struct LayerNormPostAllGatherDeviceOperation {
    using operation_attributes_t = LayerNormPostAllGatherParams;
    using tensor_args_t = LayerNormPostAllGatherInputs;
    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t =
        std::variant<LayerNormPostAllGatherProgramFactory, LayerNormPostAllGatherWelfordProgramFactory>;

    static program_factory_t select_program_factory(
        const operation_attributes_t& args, const tensor_args_t& tensor_args);

    static void validate_on_program_cache_hit(const operation_attributes_t& args, const tensor_args_t& tensor_args);
    static void validate_on_program_cache_miss(const operation_attributes_t& args, const tensor_args_t& tensor_args);

    static spec_return_value_t compute_output_specs(
        const operation_attributes_t& args, const tensor_args_t& tensor_args);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& args, const tensor_args_t& tensor_args);
};

}  // namespace ttnn::prim

namespace ttnn::prim {

Tensor layer_norm_post_all_gather(
    const Tensor& input,
    const Tensor& stats,
    LayerNormDistributedType norm_type,
    float eps,
    const std::optional<const Tensor>& gamma,
    const std::optional<const Tensor>& beta,
    const MemoryConfig& memory_config,
    const DeviceComputeKernelConfig& compute_kernel_config,
    const std::optional<DataType>& dtype,
    const std::optional<bool>& use_2d_core_grid,
    const LayerNormProgramConfig& program_config);

}  // namespace ttnn::prim
