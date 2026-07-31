// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "all_gather_regime_a_matmul_async_device_operation.hpp"

#include <cstdlib>

#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/constants.hpp>
#include "ttnn/tensor/tensor_ops.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::experimental::prim {

void AllGatherRegimeAMatmulAsyncDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& act = tensor_args.input_tensor;
    const auto& weight = tensor_args.weight_tensor;

    // Device / storage.
    TT_FATAL(
        act.storage_type() == StorageType::DEVICE && weight.storage_type() == StorageType::DEVICE,
        "all_gather_regime_a_matmul_async operands must be on device");
    TT_FATAL(act.device() == weight.device(), "all_gather_regime_a_matmul_async inputs must reside on the same device");
    TT_FATAL(
        act.buffer() != nullptr && weight.buffer() != nullptr,
        "all_gather_regime_a_matmul_async inputs must be allocated in device buffers");

    // ---- Fused fabric all-gather (Phase 1) contract ----
    const uint32_t tp = operation_attributes.tp;
    if (tp > 1) {
        TT_FATAL(tp % 2 == 0, "fused gather requires an EVEN TP group (bidirectional schedule), got {}", tp);
        const uint32_t k_local = act.logical_shape()[-1];
        const uint32_t k_global = weight.logical_shape()[-2];
        TT_FATAL(
            k_global == k_local * tp,
            "with tp={} in0 must be this device's [M, K/tp] shard: expected in0 K {} to be in1 K {} / tp",
            tp,
            k_local,
            k_global);
        TT_FATAL(
            k_local % tt::constants::TILE_WIDTH == 0,
            "the in0 K shard ({}) must be tile-aligned so shards land on tile boundaries in the staging buffer",
            k_local);
        TT_FATAL(
            tensor_args.gather_staging_buffer.has_value(),
            "fused gather (tp={}) requires gather_staging_buffer (the caller's persistent_output_buffer)",
            tp);
        const auto& stage = *tensor_args.gather_staging_buffer;
        TT_FATAL(
            stage.storage_type() == StorageType::DEVICE && stage.buffer() != nullptr,
            "gather_staging_buffer must be an allocated device tensor");
        TT_FATAL(
            stage.logical_shape()[-1] == k_global && stage.logical_shape()[-2] == act.logical_shape()[-2],
            "gather_staging_buffer must be [M, K] = [{}, {}], got [{}, {}]",
            act.logical_shape()[-2],
            k_global,
            stage.logical_shape()[-2],
            stage.logical_shape()[-1]);
        TT_FATAL(
            stage.dtype() == act.dtype(),
            "gather_staging_buffer dtype must match in0 (remote ranks write raw tiles into it)");
    }

    // Layout.
    TT_FATAL(
        act.layout() == Layout::TILE && weight.layout() == Layout::TILE,
        "all_gather_regime_a_matmul_async requires TILE layout for input and weight");

    // DType: bf16 in/out only (v1).
    TT_FATAL(
        act.dtype() == DataType::BFLOAT16 && weight.dtype() == DataType::BFLOAT16,
        "all_gather_regime_a_matmul_async v1 supports only BFLOAT16 inputs");

    // Memory-layout assumptions (v1): the in0-ring reader + output writer use interleaved-DRAM accessors,
    // and the writer / CBs hardcode bf16 tile size/format. Validate these rather than silently reading or
    // writing a wrongly-formatted tensor. (in1's DRAM 8-bank width-shard is validated separately below.)
    TT_FATAL(
        act.memory_config().buffer_type() == BufferType::DRAM &&
            act.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "all_gather_regime_a_matmul_async input (in0) must be DRAM INTERLEAVED");
    // Output numerics/layout are FIXED (BF16, DRAM-interleaved) — created that way in compute_output_specs;
    // there is no caller-supplied output dtype / memory_config to validate.

    // Shapes: no batching — all leading dims (< -2) must be 1 for both operands.
    const auto& a_logical = act.logical_shape();
    const auto& w_logical = weight.logical_shape();
    TT_FATAL(
        a_logical.rank() >= 2 && w_logical.rank() >= 2, "all_gather_regime_a_matmul_async expects rank >= 2 tensors");
    for (int i = 0; i < static_cast<int>(a_logical.rank()) - 2; ++i) {
        TT_FATAL(
            a_logical[i] == 1, "all_gather_regime_a_matmul_async input must have 1 in all dims < -2 (no batching)");
    }
    for (int i = 0; i < static_cast<int>(w_logical.rank()) - 2; ++i) {
        TT_FATAL(
            w_logical[i] == 1, "all_gather_regime_a_matmul_async weight must have 1 in all dims < -2 (no batching)");
    }

    // in1 is ALWAYS the full-K operand, so it is the authority on global K. With tp>1 in0 carries only this
    // device's K/tp shard (the tp*K_local == K_global relation is checked in the fused-gather block above),
    // so the two only have to match on the single-chip path.
    const uint32_t K = w_logical[-2];
    const uint32_t K_in0 = a_logical[-1];
    if (tp == 1) {
        TT_FATAL(
            K_in0 == K, "all_gather_regime_a_matmul_async inner dimensions must match, got K={} and K_w={}", K_in0, K);
    }
    TT_FATAL(
        a_logical[-2] > 0 && K > 0 && w_logical[-1] > 0,
        "all_gather_regime_a_matmul_async dimensions must be positive");

    // in1 must be DRAM width-sharded (built via create_regime_a_weight_memory_config): exactly 8 banks,
    // ROW_MAJOR, shard width == ceil(Nt/8) tiles, and enough K rows to cover the logical Kt.
    const auto& w_mem = weight.memory_config();
    TT_FATAL(
        w_mem.buffer_type() == BufferType::DRAM && w_mem.memory_layout() == TensorMemoryLayout::WIDTH_SHARDED,
        "all_gather_regime_a_matmul_async weight must be DRAM WIDTH_SHARDED (use "
        "create_regime_a_weight_memory_config)");
    const auto shard = w_mem.shard_spec();
    TT_FATAL(shard.has_value(), "all_gather_regime_a_matmul_async weight must carry a shard spec");
    auto cdiv32 = [](uint32_t v) { return (v + 31u) / 32u; };
    const uint32_t Kt = cdiv32(K);
    const uint32_t Nt = cdiv32(static_cast<uint32_t>(w_logical[-1]));
    const uint32_t exp_shard_cols = ((Nt + 7u) / 8u) * 32u;  // ceil(Nt/8) tiles, in elements
    TT_FATAL(
        shard->grid.num_cores() == 8u,
        "all_gather_regime_a_matmul_async weight shard must span exactly 8 DRAM banks, got {}",
        shard->grid.num_cores());
    TT_FATAL(
        shard->orientation == ShardOrientation::ROW_MAJOR,
        "all_gather_regime_a_matmul_async weight shard must be ROW_MAJOR");
    TT_FATAL(
        shard->shape[1] == exp_shard_cols,
        "all_gather_regime_a_matmul_async weight shard width must be ceil(Nt/8)*32 = {} elements, got {}",
        exp_shard_cols,
        shard->shape[1]);
    TT_FATAL(
        shard->shape[0] >= Kt * 32u,
        "all_gather_regime_a_matmul_async weight shard height must cover K ({} tiles / {} elements), got {} rows",
        Kt,
        Kt * 32u,
        shard->shape[0]);

    // config is optional: nullopt -> the program factory auto-selects via auto_select_config (ported
    // FLUX/LTX picker). An explicit RegimeAMatmulConfig overrides for reproducibility.

    // ---- Fusion validation (bias / activation / addcmul / output-split). ----
    const uint32_t N = w_logical[-1];
    const uint32_t M = a_logical[-2];

    // Output column-split: chunks>=1, dim==-1, N%chunks==0, per-chunk tile-aligned. The writer stores chunk
    // base addresses in a fixed chunk_addr[kMaxChunks] array (kMaxChunks=16 in in0_ring_reduce_writer.cpp);
    // reject chunks beyond that to avoid an out-of-bounds runtime-arg read / write.
    constexpr int32_t kMaxChunks = 16;
    const int32_t chunks = operation_attributes.chunks;
    TT_FATAL(chunks >= 1, "all_gather_regime_a_matmul_async requires chunks >= 1, got chunks={}", chunks);
    TT_FATAL(
        chunks <= kMaxChunks,
        "all_gather_regime_a_matmul_async supports at most {} chunks, got chunks={}",
        kMaxChunks,
        chunks);
    // (split `dim` is validated == -1 in the public wrapper; only `chunks` reaches the device op.)
    if (chunks > 1) {
        TT_FATAL(N % static_cast<uint32_t>(chunks) == 0, "Output width N={} must be divisible by chunks={}", N, chunks);
        const uint32_t N_per_chunk = N / static_cast<uint32_t>(chunks);
        TT_FATAL(
            N_per_chunk % TILE_WIDTH == 0,
            "Each chunk N/chunks={} must be a multiple of TILE_WIDTH={}",
            N_per_chunk,
            TILE_WIDTH);
    }

    const bool has_bias = tensor_args.bias_tensor.has_value();
    if (has_bias) {
        const auto& bias = *tensor_args.bias_tensor;
        TT_FATAL(bias.storage_type() == StorageType::DEVICE, "all_gather_regime_a_matmul_async bias must be on device");
        TT_FATAL(
            bias.buffer() != nullptr, "all_gather_regime_a_matmul_async bias must be allocated in a device buffer");
        TT_FATAL(bias.device() == act.device(), "all_gather_regime_a_matmul_async bias must be on the same device");
        TT_FATAL(bias.layout() == Layout::TILE, "all_gather_regime_a_matmul_async bias must be TILE layout");
        // The writer's feed_fused reads the bias via an interleaved-DRAM TensorAccessor (page = global N tile).
        TT_FATAL(
            bias.memory_config().buffer_type() == BufferType::DRAM &&
                bias.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
            "all_gather_regime_a_matmul_async bias must be DRAM INTERLEAVED");
        // Bias CB (c_4) is hardcoded Float16_b in the program factory; only BFLOAT16 is implemented.
        TT_FATAL(
            bias.dtype() == DataType::BFLOAT16,
            "all_gather_regime_a_matmul_async bias must be BFLOAT16 (only bf16 bias is implemented), got {}",
            bias.dtype());
        const auto& b_logical = bias.logical_shape();
        TT_FATAL(b_logical.rank() >= 1, "all_gather_regime_a_matmul_async bias must have rank >= 1");
        for (int i = 0; i < static_cast<int>(b_logical.rank()) - 1; ++i) {
            TT_FATAL(b_logical[i] == 1, "all_gather_regime_a_matmul_async bias must be 1 in all dims except the last");
        }
        TT_FATAL(
            b_logical[-1] == N,
            "all_gather_regime_a_matmul_async bias last dim must equal N ({}), got {}",
            N,
            b_logical[-1]);
    }

    // addcmul is all-or-nothing: {scalar, residual (input_a), gate (input_b)} must be supplied together or
    // not at all — otherwise a caller that passes residual/gate WITHOUT a scalar would have those tensors
    // silently ignored (no fusion applied).
    const bool has_scalar = operation_attributes.fused_ternary_scalar.has_value();
    const bool has_resid = tensor_args.fused_ternary_input_a.has_value();
    const bool has_gate = tensor_args.fused_ternary_input_b.has_value();
    TT_FATAL(
        (has_scalar && has_resid && has_gate) || (!has_scalar && !has_resid && !has_gate),
        "all_gather_regime_a_matmul_async addcmul requires all of {{scalar, residual (input_a), gate (input_b)}} "
        "together or "
        "none (got scalar={}, residual={}, gate={})",
        has_scalar,
        has_resid,
        has_gate);
    const bool has_ternary = has_scalar;
    if (has_ternary) {
        TT_FATAL(
            !operation_attributes.fused_activation.has_value(),
            "all_gather_regime_a_matmul_async does not support fused_activation together with addcmul; use one or the "
            "other");
        const auto& ta = *tensor_args.fused_ternary_input_a;  // residual [M, N]
        const auto& tb = *tensor_args.fused_ternary_input_b;  // gate [1, N] or [M, N]
        for (const auto* t : {&ta, &tb}) {
            TT_FATAL(
                t->storage_type() == StorageType::DEVICE,
                "all_gather_regime_a_matmul_async addcmul operands must be on device");
            TT_FATAL(
                t->buffer() != nullptr,
                "all_gather_regime_a_matmul_async addcmul operands must be allocated in device buffers");
            TT_FATAL(
                t->device() == act.device(),
                "all_gather_regime_a_matmul_async addcmul operands must be on the same device");
            TT_FATAL(
                t->layout() == Layout::TILE, "all_gather_regime_a_matmul_async addcmul operands must be TILE layout");
            // The writer's feed_fused reads residual/gate via interleaved-DRAM TensorAccessors.
            TT_FATAL(
                t->memory_config().buffer_type() == BufferType::DRAM &&
                    t->memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
                "all_gather_regime_a_matmul_async addcmul operands (residual/gate) must be DRAM INTERLEAVED");
            // Only bf16 (residual+gate) and fp32 (gate) CB formats are implemented; residual is further
            // pinned to BFLOAT16 below. BFLOAT8_B/BFLOAT4_B would silently map to the bf16 format => wrong.
            TT_FATAL(
                t->dtype() == DataType::BFLOAT16 || t->dtype() == DataType::FLOAT32,
                "all_gather_regime_a_matmul_async addcmul operand must be BFLOAT16 or FLOAT32, got {}",
                t->dtype());
            // No batching: all dims < -2 must be 1 (only the trailing [.., M|1, N] matrix is addressed).
            const auto& tl = t->logical_shape();
            TT_FATAL(tl.rank() >= 2, "all_gather_regime_a_matmul_async addcmul operand must have rank >= 2");
            for (int i = 0; i < static_cast<int>(tl.rank()) - 2; ++i) {
                TT_FATAL(
                    tl[i] == 1,
                    "all_gather_regime_a_matmul_async addcmul operand must be 1 in all dims < -2 (no batching)");
            }
        }
        // Residual (ternary_a) must share in1's bf16 tile format (CB c_5 is bf16).
        TT_FATAL(
            ta.dtype() == DataType::BFLOAT16,
            "all_gather_regime_a_matmul_async addcmul residual (fused_ternary_input_a) must be BFLOAT16");
        const auto& ta_l = ta.logical_shape();
        const auto& tb_l = tb.logical_shape();
        TT_FATAL(
            ta_l[-2] == M && ta_l[-1] == N,
            "all_gather_regime_a_matmul_async addcmul residual shape must be [M={}, N={}], got [{}, {}]",
            M,
            N,
            ta_l[-2],
            ta_l[-1]);
        TT_FATAL(
            (tb_l[-2] == 1 || tb_l[-2] == M) && tb_l[-1] == N,
            "all_gather_regime_a_matmul_async addcmul gate shape must be [1, N={}] (broadcast) or [M={}, N={}] (full), "
            "got [{}, {}]",
            N,
            M,
            N,
            tb_l[-2],
            tb_l[-1]);
    }
}

AllGatherRegimeAMatmulAsyncDeviceOperation::spec_return_value_t
AllGatherRegimeAMatmulAsyncDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& act = tensor_args.input_tensor;
    const auto& weight = tensor_args.weight_tensor;
    const uint32_t N = weight.logical_shape()[-1];

    // Output numerics/layout are FIXED: BF16, DRAM-interleaved (MemoryConfig{} default). Not caller-tunable.
    const auto dtype = DataType::BFLOAT16;
    const auto memory_config = MemoryConfig{};

    // Output column-split (all_gather_regime_a_matmul_async_split): `chunks` equal-width [.., M, N/chunks] tensors
    // written directly by the writer. chunks==1 (public all_gather_regime_a_matmul_async) => a single full-width
    // output.
    const int32_t chunks = operation_attributes.chunks < 1 ? 1 : operation_attributes.chunks;
    const uint32_t N_per_chunk = N / static_cast<uint32_t>(chunks);
    std::vector<TensorSpec> specs;
    specs.reserve(chunks);
    for (int32_t c = 0; c < chunks; ++c) {
        ttnn::Shape output_shape(act.logical_shape());
        output_shape[-1] = N_per_chunk;
        specs.emplace_back(output_shape, TensorLayout(dtype, PageConfig(Layout::TILE), memory_config));
    }
    return specs;
}

AllGatherRegimeAMatmulAsyncDeviceOperation::tensor_return_value_t
AllGatherRegimeAMatmulAsyncDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto specs = compute_output_specs(operation_attributes, tensor_args);
    auto* device = tensor_args.input_tensor.device();
    std::vector<Tensor> outs;
    outs.reserve(specs.size());
    for (const auto& spec : specs) {
        outs.push_back(create_device_tensor(spec, device));
    }
    return outs;
}

std::tuple<
    AllGatherRegimeAMatmulAsyncDeviceOperation::operation_attributes_t,
    AllGatherRegimeAMatmulAsyncDeviceOperation::tensor_args_t>
AllGatherRegimeAMatmulAsyncDeviceOperation::invoke(
    const Tensor& input_tensor,
    const Tensor& weight_tensor,
    const std::optional<const RegimeAMatmulConfig>& config,
    const std::optional<Tensor>& bias_tensor,
    std::optional<operations::unary::UnaryWithParam> fused_activation,
    std::optional<float> fused_ternary_scalar,
    const std::optional<Tensor>& fused_ternary_input_a,
    const std::optional<Tensor>& fused_ternary_input_b,
    int32_t chunks) {
    return {
        operation_attributes_t{
            .config = config,
            .fused_activation = std::move(fused_activation),
            .fused_ternary_scalar = fused_ternary_scalar,
            .chunks = chunks},
        tensor_args_t{
            .input_tensor = input_tensor,
            .weight_tensor = weight_tensor,
            .bias_tensor = bias_tensor,
            .fused_ternary_input_a = fused_ternary_input_a,
            .fused_ternary_input_b = fused_ternary_input_b}};
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

std::vector<Tensor> all_gather_regime_a_matmul_async(
    const Tensor& input_tensor,
    const Tensor& weight_tensor,
    const std::optional<const experimental::prim::RegimeAMatmulConfig>& config,
    const std::optional<Tensor>& bias_tensor,
    std::optional<operations::unary::UnaryWithParam> fused_activation,
    std::optional<float> fused_ternary_scalar,
    const std::optional<Tensor>& fused_ternary_input_a,
    const std::optional<Tensor>& fused_ternary_input_b,
    int32_t chunks) {
    using OperationType = experimental::prim::AllGatherRegimeAMatmulAsyncDeviceOperation;
    auto [attributes, tensor_args] = OperationType::invoke(
        input_tensor,
        weight_tensor,
        config,
        bias_tensor,
        std::move(fused_activation),
        fused_ternary_scalar,
        fused_ternary_input_a,
        fused_ternary_input_b,
        chunks);
    return ttnn::device_operation::launch<OperationType>(attributes, tensor_args);
}

}  // namespace ttnn::prim
