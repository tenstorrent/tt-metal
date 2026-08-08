// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>

#include "kda_causal_conv_device_operation_types.hpp"
#include "kda_causal_conv_program_factory.hpp"

namespace ttnn::prim {

struct KdaCausalConvOperation {
    using operation_attributes_t = KdaCausalConvParams;
    using tensor_args_t = KdaCausalConvInputs;
    using spec_return_value_t = std::vector<tt::tt_metal::TensorSpec>;
    using tensor_return_value_t = std::vector<Tensor>;
    using program_factory_t = std::variant<KdaCausalConvProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

std::vector<Tensor> kda_causal_conv1d_split(
    const Tensor&,
    const Tensor&,
    const Tensor&,
    const Tensor&,
    const Tensor&,
    const Tensor&,
    uint32_t,
    uint32_t,
    uint32_t,
    const tt::tt_metal::MemoryConfig&,
    const DeviceComputeKernelConfig&);

}  // namespace ttnn::prim
