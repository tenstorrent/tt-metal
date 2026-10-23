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

#include "fused_experts_prefill_types.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::fused_experts_prefill {

// Routed-expert FFN for prefill that reads the DECODE weight layout directly (no second copy of the
// experts in DRAM). Uses the descriptor-based program factory API.
//
// The 12x10 = 120 worker cores are split into 15 groups of 8 (4 columns x 2 rows). Every expert is
// owned by one group (expert e -> group e % 15), and inside a group the 8 cores split each expert's
// work: the I dim for gate/up (+SwiGLU), the H dim for down, exchanging the SwiGLU activation with a
// unicast all-to-all inside the group.
struct FusedExpertsPrefillDeviceOperation {
    using operation_attributes_t = fused_experts_prefill::operation_attributes_t;
    using tensor_args_t = fused_experts_prefill::tensor_args_t;
    using spec_return_value_t = fused_experts_prefill::spec_return_value_t;
    using tensor_return_value_t = fused_experts_prefill::tensor_return_value_t;

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

    // Weight DRAM addresses are baked into per-core runtime args as raw values (there are 2 x E of
    // them), so they must key the program cache -- a different set of weight tensors (next layer)
    // must build its own program.
    static tt::tt_metal::operation::Hash compute_program_hash(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const Tensor& x_tok,
        const Tensor& routing_scores,
        const std::vector<Tensor>& gate_up_weights,
        const std::vector<Tensor>& down_weights,
        uint32_t intermediate_size,
        float swiglu_limit,
        uint32_t top_k,
        float routed_scaling_factor,
        float routing_eps,
        const std::optional<MemoryConfig>& memory_config,
        const std::optional<Tensor>& routing_indices,
        const std::optional<Tensor>& ranking_scores);
};

}  // namespace ttnn::operations::experimental::deepseek_prefill::fused_experts_prefill

namespace ttnn::prim {
// Returns the per-(slot, token) weighted FFN output, [1, top_k, T, H] ROW_MAJOR bfloat16.
ttnn::operations::experimental::deepseek_prefill::fused_experts_prefill::FusedExpertsPrefillDeviceOperation::
    tensor_return_value_t
    fused_experts_prefill(
        const Tensor& x_tok,
        const Tensor& routing_scores,
        const std::vector<Tensor>& gate_up_weights,
        const std::vector<Tensor>& down_weights,
        uint32_t intermediate_size,
        float swiglu_limit,
        uint32_t top_k,
        float routed_scaling_factor,
        float routing_eps,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& routing_indices = std::nullopt,
        const std::optional<Tensor>& ranking_scores = std::nullopt);
}  // namespace ttnn::prim
