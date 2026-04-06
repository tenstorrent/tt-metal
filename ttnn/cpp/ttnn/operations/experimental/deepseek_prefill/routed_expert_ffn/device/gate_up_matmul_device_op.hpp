// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>

#include "gate_up_matmul_types.hpp"
#include "gate_up_matmul_program_factory.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::routed_expert_ffn {

struct GateUpMatmulDeviceOperation {
    using operation_attributes_t = GateUpMatmulParams;
    using tensor_args_t = GateUpMatmulInputs;
    using spec_return_value_t = TensorSpec;
    using topology_return_value_t = std::vector<tt::tt_metal::TensorTopology>;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<GateUpMatmulProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static topology_return_value_t compute_output_topologies(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static tt::stl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::operations::experimental::deepseek_prefill::routed_expert_ffn

namespace ttnn::prim {
// Returns act_out = silu(x @ gate_proj) * (x @ up_proj) (fused SiLU).
ttnn::Tensor gate_up_matmul(
    const ttnn::Tensor& x,
    const ttnn::Tensor& gate_proj,
    const ttnn::Tensor& up_proj,
    const std::optional<const ttnn::DeviceComputeKernelConfig>& compute_kernel_config);
}  // namespace ttnn::prim
