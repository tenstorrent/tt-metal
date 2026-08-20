// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "fused_experts_device_operation.hpp"

#include <algorithm>

#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::operations::experimental::deepseek::moe::fused_experts {

FusedExpertsDeviceOperation::program_factory_t FusedExpertsDeviceOperation::select_program_factory(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& /*tensor_args*/) {
    return MultiCore{};
}

void FusedExpertsDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    const auto& x = tensor_args.input_tensor;

    TT_FATAL(x.storage_type() == StorageType::DEVICE, "fused_experts: input_tensor must be on device");
    // The matmul consumes the activation as tiles, so the input must be TILE layout ([1,1,1,H] -> Kt tiles).
    TT_FATAL(
        x.layout() == tt::tt_metal::Layout::TILE, "fused_experts: input_tensor must be TILE layout for the matmul");

    const uint32_t batch = static_cast<uint32_t>(x.logical_shape()[-2]);
    const uint32_t num_weights_arg = static_cast<uint32_t>(tensor_args.gate_up_weights.size());

    {
        const auto& idx = tensor_args.routing_indices;
        const auto& scores = tensor_args.routing_scores;
        TT_FATAL(idx.storage_type() == StorageType::DEVICE, "fused_experts: routing_indices must be on device");
        TT_FATAL(scores.storage_type() == StorageType::DEVICE, "fused_experts: routing_scores must be on device");

        // Both are consumed in their native tile layout -- the kernel walks 16x16 faces to reach a
        // token's row -- which is what lets the router's topk output be passed through untouched.
        TT_FATAL(
            idx.layout() == tt::tt_metal::Layout::TILE,
            "fused_experts: routing_indices must be TILE layout (the topk index output, unmodified)");
        TT_FATAL(scores.layout() == tt::tt_metal::Layout::TILE, "fused_experts: routing_scores must be TILE layout");
        // UINT16 is what ttnn.topk emits. BFLOAT16 is accepted because ttnn.embedding only
        // gathers from a bfloat16 table, and that is how a table-driven router (the hash
        // router's frozen tid2eid) hands over its ids; every expert id below 256 is exactly
        // representable in bf16, so nothing is lost.
        TT_FATAL(
            idx.dtype() == tt::tt_metal::DataType::UINT16 || idx.dtype() == tt::tt_metal::DataType::BFLOAT16,
            "fused_experts: routing_indices must be UINT16 (a topk index output) or BFLOAT16 (an "
            "embedding gather), got {}",
            idx.dtype());
        TT_FATAL(
            idx.dtype() != tt::tt_metal::DataType::BFLOAT16 || num_weights_arg <= 256,
            "fused_experts: BFLOAT16 routing_indices need num_experts <= 256 so every id is exact "
            "in bf16, got {}",
            num_weights_arg);
        TT_FATAL(
            scores.dtype() == tt::tt_metal::DataType::BFLOAT16,
            "fused_experts: routing_scores must be BFLOAT16, got {}",
            scores.dtype());
        // Pages are read by page id, so both must be interleaved rather than sharded.
        TT_FATAL(
            !idx.memory_config().is_sharded() && !scores.memory_config().is_sharded(),
            "fused_experts: routing_indices / routing_scores must be interleaved (read by page id)");

        TT_FATAL(
            attributes.top_k > 0 && attributes.top_k <= num_weights_arg,
            "fused_experts: top_k ({}) must be in [1, {}]",
            attributes.top_k,
            num_weights_arg);
        // The ids sit in the first `top_k` columns of a token's row, so they must all land in the
        // first 16-wide face of the tile -- the only face the kernel reads for them.
        TT_FATAL(
            attributes.top_k <= 16,
            "fused_experts: top_k ({}) must be <= 16 (the ids must fit one 16-wide tile face)",
            attributes.top_k);
        TT_FATAL(
            static_cast<uint32_t>(idx.logical_shape()[-1]) == attributes.top_k,
            "fused_experts: routing_indices last dim ({}) must equal top_k ({})",
            idx.logical_shape()[-1],
            attributes.top_k);
        TT_FATAL(
            static_cast<uint32_t>(idx.logical_shape()[-2]) == batch,
            "fused_experts: routing_indices must have one row per token ({}), got {}",
            batch,
            idx.logical_shape()[-2]);
        TT_FATAL(
            static_cast<uint32_t>(scores.logical_shape()[-1]) == num_weights_arg,
            "fused_experts: routing_scores last dim ({}) must equal the number of experts ({})",
            scores.logical_shape()[-1],
            num_weights_arg);
        TT_FATAL(
            static_cast<uint32_t>(scores.logical_shape()[-2]) == batch,
            "fused_experts: routing_scores must have one row per token ({}), got {}",
            batch,
            scores.logical_shape()[-2]);
    }

    TT_FATAL(
        tensor_args.gate_up_weights.size() == tensor_args.down_weights.size(),
        "fused_experts: gate_up_weights ({}) and down_weights ({}) must have the same length",
        tensor_args.gate_up_weights.size(),
        tensor_args.down_weights.size());

    const uint32_t num_weights = num_weights_arg;
    TT_FATAL(num_weights > 0, "fused_experts: need at least one expert");

    // The op takes all `num_weights` experts and runs the gate_up matmul only for the
    // `num_experts` routing-selected ("hit") experts; the caller must pass the actual hit count,
    // i.e. the size of the *union* of the token rows' selections, because every hit expert is
    // evaluated for every row and rows that did not select it contribute through a zero weight.
    // That size is data dependent -- it grows with the batch, and shrinks if a table-driven router
    // names an expert twice for one token -- so it cannot be checked more tightly here than
    // against the expert count. A bound below the actual union drops the surplus (see the kernel).
    TT_FATAL(
        attributes.num_experts > 0 && attributes.num_experts <= num_weights,
        "fused_experts: num_experts ({}) must be in [1, {}]",
        attributes.num_experts,
        num_weights);

    // Experts run in blocks of this many, and only a block's activations are held in L1 at once, so
    // this -- not num_experts -- is what has to fit. `invoke` has already resolved 0 and clamped it
    // to num_experts, so anything outside the range means it was set directly on the attributes.
    TT_FATAL(
        attributes.experts_block_size > 0 && attributes.experts_block_size <= attributes.num_experts,
        "fused_experts: experts_block_size ({}) must be in [1, num_experts ({})]",
        attributes.experts_block_size,
        attributes.num_experts);

    // gate_up weights must be DRAM ND-sharded so that each shard is exactly one core's column
    // slice (read in a single NoC read by the dataflow kernels). The SwiGLU I dim is spread
    // over all kNumCores cores -- 32 columns each once I >= 32*kNumCores -- so a shard holds
    // that core's gate columns plus their paired up columns, and the weight is permuted on the
    // host into matching per-core [gate | up] blocks. See swiglu_tiles_per_core_for().
    constexpr uint32_t kNumCores = 64;  // 8x8 compute grid
    constexpr uint32_t TILE_DIM = 32;
    const uint32_t i_tiles = attributes.intermediate_size / TILE_DIM;
    const uint32_t swiglu_tiles_per_core = std::max<uint32_t>(1u, i_tiles / kNumCores);
    const uint32_t kColsPerCore = 2u * TILE_DIM * swiglu_tiles_per_core;
    // Phase 1 holds gate and up for this core's slice in DST at once, which fits 4 fp32 tiles.
    TT_FATAL(
        2u * swiglu_tiles_per_core <= 4u,
        "fused_experts: intermediate_size ({}) gives {} SwiGLU tiles per core, but the gate+up "
        "SwiGLU pass holds 2x that in DST (max 4 fp32 tiles)",
        attributes.intermediate_size,
        swiglu_tiles_per_core);
    for (uint32_t e = 0; e < num_weights; ++e) {
        const auto& w = tensor_args.gate_up_weights[e];
        TT_FATAL(
            w.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM,
            "fused_experts: gate_up_weights[{}] must be in DRAM",
            e);
        const auto& nd = w.memory_config().nd_shard_spec();
        TT_FATAL(nd.has_value(), "fused_experts: gate_up_weights[{}] must be ND-sharded (one shard per core)", e);
        const auto& shard_shape = nd->shard_shape;
        TT_FATAL(
            static_cast<uint32_t>(shard_shape[-1]) == kColsPerCore,
            "fused_experts: gate_up_weights[{}] shard last dim ({}) must be {} (one core's [gate | up] slice)",
            e,
            shard_shape[-1],
            kColsPerCore);
        TT_FATAL(
            static_cast<uint32_t>(shard_shape[-2]) == static_cast<uint32_t>(w.logical_shape()[-2]),
            "fused_experts: gate_up_weights[{}] shard must span the full K dim ({} rows), got {}",
            e,
            w.logical_shape()[-2],
            shard_shape[-2]);
    }

    // down weights must be DRAM ND-sharded so each shard is exactly one core's [I, H/64]
    // column slice (read in a single NoC read). Each shard spans the full I (contraction) dim
    // and one core's 64-column H output slice; H/64 shards cover the output H dim.
    const uint32_t hidden = static_cast<uint32_t>(x.logical_shape()[-1]);
    TT_FATAL(
        hidden % kNumCores == 0,
        "fused_experts: hidden dim ({}) must be divisible by the {}-core grid",
        hidden,
        kNumCores);
    const uint32_t down_shard_cols = hidden / kNumCores;
    for (uint32_t e = 0; e < num_weights; ++e) {
        const auto& w = tensor_args.down_weights[e];
        TT_FATAL(
            w.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM,
            "fused_experts: down_weights[{}] must be in DRAM",
            e);
        const auto& nd = w.memory_config().nd_shard_spec();
        TT_FATAL(nd.has_value(), "fused_experts: down_weights[{}] must be ND-sharded (one shard per core)", e);
        const auto& shard_shape = nd->shard_shape;
        TT_FATAL(
            static_cast<uint32_t>(w.logical_shape()[-2]) == attributes.intermediate_size,
            "fused_experts: down_weights[{}] K dim ({}) must equal intermediate_size ({})",
            e,
            w.logical_shape()[-2],
            attributes.intermediate_size);
        TT_FATAL(
            static_cast<uint32_t>(w.logical_shape()[-1]) == hidden,
            "fused_experts: down_weights[{}] output dim ({}) must equal hidden ({})",
            e,
            w.logical_shape()[-1],
            hidden);
        TT_FATAL(
            static_cast<uint32_t>(shard_shape[-1]) == down_shard_cols,
            "fused_experts: down_weights[{}] shard last dim ({}) must be {} (one core's H/64 slice)",
            e,
            shard_shape[-1],
            down_shard_cols);
        TT_FATAL(
            static_cast<uint32_t>(shard_shape[-2]) == static_cast<uint32_t>(w.logical_shape()[-2]),
            "fused_experts: down_weights[{}] shard must span the full I dim ({} rows), got {}",
            e,
            w.logical_shape()[-2],
            shard_shape[-2]);
    }

    // Token rows (batch): activations are [1, 1, B, H] -- the B tokens of a batched decode step (or
    // of a short prefill chunk) are packed into dim -2, each with its own routing-weight row, and
    // all B are computed in one op.
    //
    // B is capped at a single 32-row tile, and the batch/seq dims of a [1, B, S, H] activation must
    // therefore be folded into dim -2 (B*S tokens) by the caller. That is not an arbitrary limit:
    // batching pays off here only because ONE fetch of an expert's weights serves every token, which
    // requires every token's activation to be resident on every core simultaneously. Tokens packed
    // in dim -2 share a single tile row, so the tile counts, the DST budget, the L1 footprint and
    // the matmul cost are all identical to the single-token case. Tokens spread over separate tile
    // rows (e.g. an unfolded [1, B, 1, H]) would instead multiply both the resident activation
    // (k_tiles tiles) and the gathered activation block (num_experts * i_tiles tiles, already the
    // dominant L1 consumer) by B, and waste 31 of every 32 matmul rows on tile padding.
    TT_FATAL(
        batch >= 1 && batch <= TILE_DIM,
        "fused_experts: batch (input_tensor dim -2) must be in [1, {}], got {}",
        TILE_DIM,
        batch);
    const auto& x_shape = x.logical_shape();
    for (int d = 0; d + 2 < static_cast<int>(x_shape.rank()); ++d) {
        TT_FATAL(
            x_shape[d] == 1,
            "fused_experts: input_tensor dim {} must be 1 (got {}); the tokens of a [1, B, S, H] "
            "activation must be folded into dim -2 as [1, 1, B*S, H] so they share one tile row",
            d,
            x_shape[d]);
    }
    // Matmul contraction dim: input H must match gate_up K (rows).
    TT_FATAL(
        static_cast<uint32_t>(x.logical_shape()[-1]) ==
            static_cast<uint32_t>(tensor_args.gate_up_weights.front().logical_shape()[-2]),
        "fused_experts: input hidden dim ({}) must equal gate_up K ({})",
        x.logical_shape()[-1],
        tensor_args.gate_up_weights.front().logical_shape()[-2]);

    // SwiGLU splits the gate_up output (2I) into gate/up halves of size I each.
    const uint32_t two_intermediate = static_cast<uint32_t>(tensor_args.gate_up_weights.front().logical_shape()[-1]);
    TT_FATAL(
        two_intermediate == 2u * attributes.intermediate_size,
        "fused_experts: gate_up output dim ({}) must equal 2 * intermediate_size ({})",
        two_intermediate,
        2u * attributes.intermediate_size);

    // TODO: validate per-expert weight shapes against H / 2I / I, dtypes, and tile alignment.
}

void FusedExpertsDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(attributes, tensor_args);
}

tt::tt_metal::operation::Hash FusedExpertsDeviceOperation::compute_program_hash(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    // CRITICAL: the per-expert gate_up / down weight DRAM base addresses are baked into the
    // dataflow kernels as COMPILE-TIME args (see `append_addrs_ct` in the program factory), so a
    // program compiled for one set of expert weights is only valid for those exact buffers. The
    // default device-op hash keys solely on tensor specs (shape / dtype / layout / memory-config)
    // plus the scalar attributes -- all identical from one MoE layer to the next -- so it would
    // return a stale cached program holding the *previous* layer's (by then possibly freed) weight
    // addresses, making the matmuls read garbage DRAM (observed as ~1e37 / inf outputs). Fold every
    // weight buffer address into the hash so a different set of weight tensors misses the program
    // cache and recompiles with the correct baked-in addresses.
    std::vector<uint32_t> weight_addresses;
    weight_addresses.reserve(tensor_args.gate_up_weights.size() + tensor_args.down_weights.size());
    for (const auto& w : tensor_args.gate_up_weights) {
        weight_addresses.push_back(static_cast<uint32_t>(w.buffer()->address()));
    }
    for (const auto& w : tensor_args.down_weights) {
        weight_addresses.push_back(static_cast<uint32_t>(w.buffer()->address()));
    }
    // top_k / scaling / eps are compile-time args of the routing kernel, so they have to key the
    // program alongside the weight addresses. The ids' dtype does too (it selects the decode), and
    // it rides along in the routing_indices spec.
    auto hash = tt::tt_metal::operation::hash_operation<FusedExpertsDeviceOperation>(
        attributes.num_experts,
        attributes.intermediate_size,
        attributes.swiglu_limit,
        attributes.top_k,
        attributes.routed_scaling_factor,
        attributes.routing_eps,
        attributes.output_memory_config,
        tensor_args.input_tensor,
        tensor_args.routing_indices,
        tensor_args.routing_scores,
        tensor_args.gate_up_weights.front(),
        tensor_args.down_weights.front(),
        weight_addresses);
    return hash;
}

FusedExpertsDeviceOperation::spec_return_value_t FusedExpertsDeviceOperation::compute_output_specs(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    // The output is, per token row, the routing-weighted sum of every selected expert's down
    // matmul result:
    //   act       = silu(clamp(gate, max=limit)) * clamp(up, -limit, limit),
    //               where [gate, up] = x @ gate_up_w[hit_ids[i]];
    //   output[b] = sum_i routing_weights[b, hit_ids[i]] * (act[b] @ down_w[hit_ids[i]]).
    // Shape [1, B, H] (the B token rows, padded to a 32-row tile in TILE layout), BFLOAT16.
    // H is the hidden size (== down weight output dim == input hidden dim).
    const uint32_t batch = static_cast<uint32_t>(tensor_args.input_tensor.logical_shape()[-2]);
    const uint32_t hidden = static_cast<uint32_t>(tensor_args.input_tensor.logical_shape()[-1]);
    const ttnn::Shape output_shape({1, batch, hidden});
    return tt::tt_metal::TensorSpec(
        output_shape,
        tt::tt_metal::TensorLayout(
            tt::tt_metal::DataType::BFLOAT16,
            tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
            attributes.output_memory_config));
}

FusedExpertsDeviceOperation::tensor_return_value_t FusedExpertsDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    auto output_spec = compute_output_specs(operation_attributes, tensor_args);
    return create_device_tensor(output_spec, tensor_args.input_tensor.device());
}

std::tuple<FusedExpertsDeviceOperation::operation_attributes_t, FusedExpertsDeviceOperation::tensor_args_t>
FusedExpertsDeviceOperation::invoke(
    const Tensor& input_tensor,
    const Tensor& routing_indices,
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
    const std::optional<MemoryConfig>& memory_config) {
    operation_attributes_t attributes{
        .num_experts = num_experts,
        .intermediate_size = intermediate_size,
        .swiglu_limit = swiglu_limit,
        // 0 means "no blocking": resolve it here so the attribute (and the program hash) always
        // carries the block size the kernels are actually compiled for.
        .experts_block_size = experts_block_size == 0 ? num_experts : std::min(experts_block_size, num_experts),
        // k defaults to the index tensor's width, so a caller that already shaped the router's
        // output correctly does not have to restate it.
        .top_k = top_k == 0 ? static_cast<uint32_t>(routing_indices.logical_shape()[-1]) : top_k,
        .routed_scaling_factor = routed_scaling_factor,
        .routing_eps = routing_eps,
        .output_memory_config = memory_config.value_or(input_tensor.memory_config()),
    };
    tensor_args_t tensor_args{
        .input_tensor = input_tensor,
        .routing_indices = routing_indices,
        .routing_scores = routing_scores,
        .gate_up_weights = gate_up_weights,
        .down_weights = down_weights,
    };
    return {std::move(attributes), std::move(tensor_args)};
}

}  // namespace ttnn::operations::experimental::deepseek::moe::fused_experts

namespace ttnn::prim {
ttnn::operations::experimental::deepseek::moe::fused_experts::FusedExpertsDeviceOperation::tensor_return_value_t
fused_experts(
    const Tensor& input_tensor,
    const Tensor& routing_indices,
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
    const std::optional<MemoryConfig>& memory_config) {
    using OperationType = ttnn::operations::experimental::deepseek::moe::fused_experts::FusedExpertsDeviceOperation;
    auto [operation_attributes, tensor_args] = OperationType::invoke(
        input_tensor,
        routing_indices,
        routing_scores,
        gate_up_weights,
        down_weights,
        num_experts,
        intermediate_size,
        swiglu_limit,
        top_k,
        routed_scaling_factor,
        routing_eps,
        experts_block_size,
        memory_config);
    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}
}  // namespace ttnn::prim
