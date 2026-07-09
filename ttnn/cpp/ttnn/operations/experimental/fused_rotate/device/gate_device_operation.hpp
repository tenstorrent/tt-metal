// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "gate_program_factory.hpp"
#include "gate_device_operation_types.hpp"

namespace ttnn::experimental::prim {

struct GateDeviceOperation {
    using operation_attributes_t = GateParams;
    using tensor_args_t = GateInputs;
    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<GateProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {
Tensor fused_gate(
    const Tensor& a, const Tensor& gate, const Tensor& b, uint32_t Wt, uint32_t Gt, uint32_t Ht, uint32_t mode);
}  // namespace ttnn::prim
