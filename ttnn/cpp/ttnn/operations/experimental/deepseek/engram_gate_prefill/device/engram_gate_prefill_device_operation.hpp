// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "engram_gate_prefill_device_operation_types.hpp"
#include "engram_gate_prefill_program_factory.hpp"

#include "ttnn/device_operation.hpp"

namespace ttnn::operations::experimental::deepseek::engram_gate_prefill {

struct EngramGatePrefillDeviceOperation {
    using operation_attributes_t = engram_gate_prefill::operation_attributes_t;
    using tensor_args_t = engram_gate_prefill::tensor_args_t;
    using spec_return_value_t = engram_gate_prefill::spec_return_value_t;
    using tensor_return_value_t = engram_gate_prefill::tensor_return_value_t;
    using program_factory_t = std::variant<EngramGatePrefillProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::operations::experimental::deepseek::engram_gate_prefill

namespace ttnn::prim {

ttnn::Tensor engram_gate_prefill(
    const ttnn::Tensor& streams,
    const ttnn::Tensor& kv,
    const ttnn::Tensor& gate_weight,
    float eps,
    float clamp_value,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    const std::optional<const ttnn::DeviceComputeKernelConfig>& compute_kernel_config);

}  // namespace ttnn::prim
