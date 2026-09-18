// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>

#include "qkv_causal_conv1d_silu_device_operation_types.hpp"
#include "qkv_causal_conv1d_silu_program_factory.hpp"
#include "ttnn/operation.hpp"

namespace ttnn::experimental::prim {

struct QkvCausalConv1dSiluOperation {
    using operation_attributes_t = QkvCausalConv1dSiluParams;
    using tensor_args_t = QkvCausalConv1dSiluInputs;
    using spec_return_value_t = std::vector<tt::tt_metal::TensorSpec>;
    using tensor_return_value_t = std::vector<Tensor>;
    using program_factory_t = std::variant<QkvCausalConv1dSiluProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static tt::tt_metal::operation::OpPerformanceModelGeneral<tensor_return_value_t> create_op_performance_model(
        const operation_attributes_t&, const tensor_args_t&, tensor_return_value_t&);
};

std::vector<Tensor> qkv_causal_conv1d_silu(
    const Tensor&,
    const Tensor&,
    const Tensor&,
    const Tensor&,
    const Tensor&,
    const Tensor&,
    uint32_t,
    uint32_t,
    uint32_t,
    uint32_t,
    const tt::tt_metal::MemoryConfig&,
    const DeviceComputeKernelConfig&);

}  // namespace ttnn::experimental::prim
