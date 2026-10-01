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

#include <tt-metalium/constants.hpp>
#include <tt-metalium/tile.hpp>

namespace ttnn::operations::experimental::deepseek::moe::fused_experts {

// Decode ROW_MAJOR [1,1,1,H] is physically a sequence of 1x32 faces (same packing as
// rms_norm / matmul_decode). Compute uses that tile instead of tilize/untilize.
inline bool fused_experts_rm_as_1x32(const Tensor& x) { return x.layout() == tt::tt_metal::Layout::ROW_MAJOR; }

inline tt::tt_metal::Tile fused_experts_compute_tile(const Tensor& x) {
    if (fused_experts_rm_as_1x32(x)) {
        return tt::tt_metal::Tile({1, tt::constants::TILE_WIDTH});
    }
    return x.tensor_spec().tile();
}

// ROW_MAJOR + L1 + HEIGHT_SHARDED with a shard spec: the replicated activation row that
// `all_gather_for_matmul` multicasts onto every matmul core and that `matmul_decode` consumes in
// place as its "replicated-A" input. Each core of the shard grid holds a full [rows, H] row, so the
// op reads it locally and neither reads it from DRAM nor broadcasts it.
inline bool fused_experts_input_is_replicated(const Tensor& x) {
    if (x.layout() != tt::tt_metal::Layout::ROW_MAJOR) {
        return false;
    }
    const auto& mem = x.memory_config();
    return mem.memory_layout() == tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED &&
           mem.buffer_type() == tt::tt_metal::BufferType::L1 && mem.shard_spec().has_value();
}

// Token rows carried by the input. A replicated row has `num_cores * rows` in dim -2 (that is what
// all_gather_for_matmul's output spec reports, so the framework's volume accounting stays honest),
// so the row count has to come from the shard height -- the same derivation matmul_decode makes in
// `m_from_replicated_shard`.
inline uint32_t fused_experts_input_rows(const Tensor& x) {
    if (fused_experts_input_is_replicated(x)) {
        return static_cast<uint32_t>(x.memory_config().shard_spec()->shape[0]);
    }
    return static_cast<uint32_t>(x.logical_shape()[-2]);
}

// Fuses the per-expert routed-FFN loop
//   gate_up = matmul(x, gate_up_w[e]); act = swiglu(gate_up);
//   down = matmul(act, down_w[e]); acc += down * w[:, e]
// for all selected experts into a single device operation, where the selection comes either from the
// router's precomputed ids or from a top-k the leader kernel computes over an E-wide score row, and
// the per-token weights w are always derived on device (see tensor_args_t).
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
        const Tensor& routing_scores,
        const std::vector<Tensor>& gate_up_weights,
        const std::vector<Tensor>& down_weights,
        uint32_t num_experts,
        uint32_t intermediate_size,
        float swiglu_limit,
        uint32_t top_k,
        float routed_scaling_factor,
        float routing_eps,
        uint32_t experts_block_size,
        bool two_hub_gather,
        const std::optional<MemoryConfig>& memory_config,
        const std::optional<Tensor>& routing_indices,
        const std::optional<Tensor>& ranking_scores);
};

}  // namespace ttnn::operations::experimental::deepseek::moe::fused_experts

namespace ttnn::prim {
ttnn::operations::experimental::deepseek::moe::fused_experts::FusedExpertsDeviceOperation::tensor_return_value_t
fused_experts(
    const Tensor& input_tensor,
    const Tensor& routing_scores,
    const std::vector<Tensor>& gate_up_weights,
    const std::vector<Tensor>& down_weights,
    uint32_t num_experts,
    uint32_t intermediate_size,
    float swiglu_limit,
    uint32_t top_k = 0,
    float routed_scaling_factor = 1.0F,
    float routing_eps = 0.0F,
    uint32_t experts_block_size = 0,
    bool two_hub_gather = true,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& routing_indices = std::nullopt,
    const std::optional<Tensor>& ranking_scores = std::nullopt);
}  // namespace ttnn::prim
