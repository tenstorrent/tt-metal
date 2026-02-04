// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "layernorm_pre_all_gather_program_factory.hpp"

#include "layernorm_pre_all_gather_device_operation_types.hpp"

namespace ttnn::prim {

struct LayerNormPreAllGatherDeviceOperation {
    using operation_attributes_t = LayerNormPreAllGatherParams;
    using tensor_args_t = Tensor;
    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<
        LayerNormPreAllGatherProgramFactory,
        LayerNormPreAllGather2DProgramFactory,
        LayerNormPreAllGatherWelfordProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::prim

namespace ttnn::prim {

Tensor layer_norm_pre_all_gather(
    const Tensor& input,
    LayerNormDistributedType norm_type,
    const std::optional<tt::tt_metal::DataType>& dtype,
    const DeviceComputeKernelConfig& compute_kernel_config,
    const LayerNormProgramConfig& program_config,
    const std::optional<bool>& use_2d_core_grid);

}  // namespace ttnn::prim
