// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "fused_experts_device_operation.hpp"

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
    const auto& rw = tensor_args.routing_weights;

    TT_FATAL(x.storage_type() == StorageType::DEVICE, "fused_experts: input_tensor must be on device");
    TT_FATAL(rw.storage_type() == StorageType::DEVICE, "fused_experts: routing_weights must be on device");
    // The matmul consumes the activation as tiles, so the input must be TILE layout ([1,1,1,H] -> Kt tiles).
    TT_FATAL(
        x.layout() == tt::tt_metal::Layout::TILE, "fused_experts: input_tensor must be TILE layout for the matmul");

    // First version reads routing_weights element-by-element on a single core, so it must be a
    // contiguous ROW_MAJOR bfloat16 row.
    TT_FATAL(rw.layout() == tt::tt_metal::Layout::ROW_MAJOR, "fused_experts: routing_weights must be ROW_MAJOR layout");
    TT_FATAL(
        rw.dtype() == tt::tt_metal::DataType::BFLOAT16,
        "fused_experts: routing_weights must be BFLOAT16, got {}",
        rw.dtype());

    TT_FATAL(
        tensor_args.gate_up_weights.size() == tensor_args.down_weights.size(),
        "fused_experts: gate_up_weights ({}) and down_weights ({}) must have the same length",
        tensor_args.gate_up_weights.size(),
        tensor_args.down_weights.size());

    // Expert selection/scaling is on-device: weight i is scaled by routing_weights column i, so the
    // routing-weight width must match the number of provided weight pairs.
    const uint32_t num_weights = static_cast<uint32_t>(tensor_args.gate_up_weights.size());
    TT_FATAL(num_weights > 0, "fused_experts: need at least one expert");
    TT_FATAL(
        static_cast<uint32_t>(rw.logical_shape()[-1]) == num_weights,
        "fused_experts: routing_weights last dim ({}) must equal the number of experts ({})",
        rw.logical_shape()[-1],
        num_weights);

    // The op takes all `num_weights` experts and runs the gate_up matmul only for the
    // `num_experts` routing-selected ("hit") experts; the caller must pass the actual
    // hit count (the number of nonzero routing-weight columns).
    TT_FATAL(
        attributes.num_experts > 0 && attributes.num_experts <= num_weights,
        "fused_experts: num_experts ({}) must be in [1, {}]",
        attributes.num_experts,
        num_weights);

    // gate_up weights must be DRAM ND-sharded so that each shard is exactly one core's
    // [K, 128] column slice (read in a single NoC read by the dataflow kernels). The
    // weight is permuted on the host into per-core [gate_64 | up_64] blocks, so each
    // shard holds a core's 2 gate tiles plus their paired up tiles (128 cols). I/64
    // shards cover the SwiGLU output I dim.
    constexpr uint32_t kColsPerCore = 128;
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
            "fused_experts: gate_up_weights[{}] shard last dim ({}) must be {} (one core's [gate_64 | up_64] slice)",
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
    constexpr uint32_t kNumCores = 64;  // 8x8 compute grid
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

    // Decode-only: sequence length T == 1.
    TT_FATAL(
        static_cast<uint32_t>(x.logical_shape()[-2]) == 1,
        "fused_experts: decode op expects sequence length 1, got {}",
        x.logical_shape()[-2]);

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
    auto hash = tt::tt_metal::operation::hash_operation<FusedExpertsDeviceOperation>(
        attributes.num_experts,
        attributes.intermediate_size,
        attributes.swiglu_limit,
        attributes.output_memory_config,
        tensor_args.input_tensor,
        tensor_args.routing_weights,
        tensor_args.gate_up_weights.front(),
        tensor_args.down_weights.front(),
        weight_addresses);
    return hash;
}

FusedExpertsDeviceOperation::spec_return_value_t FusedExpertsDeviceOperation::compute_output_specs(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    // The output is the routing-weighted sum of every selected expert's down matmul result:
    //   act    = silu(clamp(gate, max=limit)) * clamp(up, -limit, limit),
    //            where [gate, up] = x @ gate_up_w[hit_ids[i]];
    //   output = sum_i routing_weights[hit_ids[i]] * (act @ down_w[hit_ids[i]]).
    // Shape [1, 1, H] (decode token row, padded to a 32-row tile in TILE layout), BFLOAT16.
    // H is the hidden size (== down weight output dim == input hidden dim).
    const uint32_t hidden = static_cast<uint32_t>(tensor_args.input_tensor.logical_shape()[-1]);
    const ttnn::Shape output_shape({1, 1, hidden});
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
    const Tensor& routing_weights,
    const std::vector<Tensor>& gate_up_weights,
    const std::vector<Tensor>& down_weights,
    uint32_t num_experts,
    uint32_t intermediate_size,
    float swiglu_limit,
    const std::optional<MemoryConfig>& memory_config) {
    operation_attributes_t attributes{
        .num_experts = num_experts,
        .intermediate_size = intermediate_size,
        .swiglu_limit = swiglu_limit,
        .output_memory_config = memory_config.value_or(input_tensor.memory_config()),
    };
    tensor_args_t tensor_args{
        .input_tensor = input_tensor,
        .routing_weights = routing_weights,
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
    const Tensor& routing_weights,
    const std::vector<Tensor>& gate_up_weights,
    const std::vector<Tensor>& down_weights,
    uint32_t num_experts,
    uint32_t intermediate_size,
    float swiglu_limit,
    const std::optional<MemoryConfig>& memory_config) {
    using OperationType = ttnn::operations::experimental::deepseek::moe::fused_experts::FusedExpertsDeviceOperation;
    auto [operation_attributes, tensor_args] = OperationType::invoke(
        input_tensor,
        routing_weights,
        gate_up_weights,
        down_weights,
        num_experts,
        intermediate_size,
        swiglu_limit,
        memory_config);
    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}
}  // namespace ttnn::prim
