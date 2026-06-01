// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <variant>

#include "ttnn/core.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"
#include <tt-metalium/program_descriptors.hpp>

#include "fused_experts_device_operation_types.hpp"

namespace ttnn::operations::experimental::deepseek::moe::fused_experts {

// Fuses the per-expert routed-FFN loop
//   gate_up = matmul(x, gate_up_w[e]); act = swiglu(gate_up);
//   down = matmul(act, down_w[e]); acc += down * routing_weights[:, e]
// for all selected experts into a single device operation.
//
// Uses the descriptor-based program factory API (returns a ProgramDescriptor); the framework
// handles program construction, caching and runtime-arg patching -- no shared_variables_t or
// override_runtime_arguments required.
struct FusedExpertsDeviceOperation {
    using operation_attributes_t = fused_experts::operation_attributes_t;
    using tensor_args_t = fused_experts::tensor_args_t;
    using spec_return_value_t = fused_experts::spec_return_value_t;
    using tensor_return_value_t = fused_experts::tensor_return_value_t;

    // Distributes output tiles across the compute grid.
    struct MultiCore {
        static tt::tt_metal::ProgramDescriptor create_descriptor(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);
    };

    using program_factory_t = std::variant<MultiCore>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);

    // The per-expert weight DRAM addresses are baked into the kernels as compile-time args, so the
    // default (spec-only) program hash would reuse a stale program when only the weight tensors
    // change. Fold the weight addresses into the hash so different weights miss the program cache.
    static tt::tt_metal::operation::Hash compute_program_hash(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const Tensor& input_tensor,
        const Tensor& routing_weights,
        const std::vector<Tensor>& gate_up_weights,
        const std::vector<Tensor>& down_weights,
        uint32_t num_experts,
        uint32_t intermediate_size,
        float swiglu_limit,
        const std::optional<MemoryConfig>& memory_config);
};

}  // namespace ttnn::operations::experimental::deepseek::moe::fused_experts

namespace ttnn::prim {
ttnn::operations::experimental::deepseek::moe::fused_experts::FusedExpertsDeviceOperation::tensor_return_value_t
fused_experts(
    const Tensor& input_tensor,
    const Tensor& routing_weights,
    const std::vector<Tensor>& gate_up_weights,
    const std::vector<Tensor>& down_weights,
    uint32_t num_experts,
    uint32_t intermediate_size,
    float swiglu_limit,
    const std::optional<MemoryConfig>& memory_config = std::nullopt);
}  // namespace ttnn::prim
