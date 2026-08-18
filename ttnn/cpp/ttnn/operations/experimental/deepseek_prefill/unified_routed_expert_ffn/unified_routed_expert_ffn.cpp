// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "unified_routed_expert_ffn.hpp"

#include "device/unified_routed_expert_ffn_device_operation.hpp"
#include "tt-metalium/math.hpp"
#include "ttnn/operations/creation/creation.hpp"
#include "ttnn/operations/experimental/deepseek_prefill/moe_fused_swiglu/moe_fused_swiglu.hpp"
#include "ttnn/operations/experimental/deepseek_prefill/routed_expert_ffn/routed_expert_ffn.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::unified_routed_expert_ffn {

ttnn::Tensor unified_routed_expert_ffn(
    const ttnn::Tensor& x,
    const ttnn::Tensor& gate_proj,
    const ttnn::Tensor& up_proj,
    const ttnn::Tensor& down_proj,
    const ttnn::Tensor& counts,
    const ttnn::Tensor& global_expert_idx_table,
    uint32_t local_expert_id,
    const std::optional<const ttnn::DeviceComputeKernelConfig>& compute_kernel_config,
    const std::optional<ttnn::Tensor>& output,
    const std::optional<ttnn::Tensor>& expert_region_offsets,
    const std::optional<uint32_t>& input_m_tiles,
    bool read_x_at_offset,
    bool x_is_row_major,
    RoutedExpertActivation activation,
    const std::optional<ttnn::Tensor>& gate_bias,
    const std::optional<ttnn::Tensor>& up_bias,
    const std::optional<ttnn::Tensor>& down_bias) {
    // Single-op fused per-expert FFN. One device Program runs gate matmul,
    // up matmul, silu, multiply, down matmul as four phases inside the same
    // kernel. The kernel reads counts[global_expert_idx_table[local_expert_id]]
    // device-side at entry and, from that runtime count, PICKS chunk_M_tiles /
    // per_core_M / num_chunks itself (adaptive_chunk.hpp) — sizing the per-core
    // work to the actual token count with no expected-token argument. chunks
    // past the count are skipped entirely (no matmul, no mcast).
    //
    // The host only sets the CB-sized MAXIMUM chunk (kMaxChunkMTiles => per_core_M
    // 8). The program factory's L1 guard may lower it for large models; the
    // device picker never exceeds whatever max the CBs were sized to.
    constexpr uint32_t kMaxChunkMTiles = 64;  // per_core_M_max = 8 (L1 cap)
    // This expert's M in tiles. Defaults to x's allocated M; a caller passing a
    // shared x buffer (wider than one region) supplies the per-expert value.
    const uint32_t M_tiles_full = input_m_tiles.value_or(x.padded_shape()[-2] / 32);

    return ttnn::prim::unified_routed_expert_ffn(
        x,
        gate_proj,
        up_proj,
        down_proj,
        counts,
        global_expert_idx_table,
        local_expert_id,
        kMaxChunkMTiles,
        M_tiles_full,
        read_x_at_offset,
        x_is_row_major,
        compute_kernel_config.has_value() ? std::optional<ttnn::DeviceComputeKernelConfig>(*compute_kernel_config)
                                          : std::nullopt,
        output,
        expert_region_offsets,
        activation,
        gate_bias,
        up_bias,
        down_bias);
}

ttnn::Tensor unified_routed_expert_moe(
    const ttnn::Tensor& dispatched_buffer,
    const ttnn::Tensor& expert_region_offsets,
    const ttnn::Tensor& expert_token_counts,
    const ttnn::Tensor& global_expert_idx_table,
    const std::vector<ttnn::Tensor>& gate_projs,
    const std::vector<ttnn::Tensor>& up_projs,
    const std::vector<ttnn::Tensor>& down_projs,
    uint32_t max_dispatched_tokens_per_expert,
    const std::optional<const ttnn::DeviceComputeKernelConfig>& compute_kernel_config,
    RoutedExpertActivation activation,
    RoutedExpertImplementation implementation,
    const std::optional<std::vector<ttnn::Tensor>>& gate_biases,
    const std::optional<std::vector<ttnn::Tensor>>& up_biases,
    const std::optional<std::vector<ttnn::Tensor>>& down_biases) {
    TT_FATAL(
        gate_projs.size() == up_projs.size() && gate_projs.size() == down_projs.size(),
        "gate/up/down projection lists must have the same length (got {}, {}, {})",
        gate_projs.size(),
        up_projs.size(),
        down_projs.size());
    const uint32_t experts_per_chip = static_cast<uint32_t>(gate_projs.size());
    TT_FATAL(experts_per_chip > 0, "Need at least one expert per chip");

    // Optional per-expert biases (gpt-oss): all three lists together or none,
    // each the same length as the weight lists (one bias per local expert).
    const int bias_lists = static_cast<int>(gate_biases.has_value()) + static_cast<int>(up_biases.has_value()) +
                           static_cast<int>(down_biases.has_value());
    TT_FATAL(
        bias_lists == 0 || bias_lists == 3,
        "gate/up/down bias lists must all be provided together or all omitted (got {} of 3)",
        bias_lists);
    const bool has_bias = bias_lists == 3;
    if (has_bias) {
        TT_FATAL(
            gate_biases->size() == experts_per_chip && up_biases->size() == experts_per_chip &&
                down_biases->size() == experts_per_chip,
            "bias lists must have one entry per local expert ({}), got ({}, {}, {})",
            experts_per_chip,
            gate_biases->size(),
            up_biases->size(),
            down_biases->size());
    }

    // Per-expert composite: run the selected kernel on each expert's slice of
    // the dispatched buffer at that expert's region offset. This fuses the
    // old ttnn::extract (input slice) + ttnn::insert (output placement) pair
    // into the reader and writer — no per-expert temporary buffer or extra DRAM
    // round trip. The fused implementation is intentionally explicit: callers
    // with SwiGluOai or projection biases must select the unified path.
    //
    // x is the whole shared buffer, so pass this expert's row count
    // (max_dispatched_tokens_per_expert in tiles) as input_m_tiles — the op sizes
    // its grid/chunks to one expert, not the buffer.
    //
    const bool use_moe_fused_swiglu = implementation == RoutedExpertImplementation::MoeFusedSwiGlu;
    if (use_moe_fused_swiglu) {
        TT_FATAL(
            activation == RoutedExpertActivation::Silu,
            "RoutedExpertImplementation::MoeFusedSwiGlu supports only RoutedExpertActivation::Silu");
        TT_FATAL(
            !has_bias, "RoutedExpertImplementation::MoeFusedSwiGlu does not support routed-expert projection biases");
    }

    // moe_fused_swiglu readers prefetch the next M block, so its direct-write
    // output may not alias the shared input. Allocate one output for all
    // experts; combine reads only count-bounded rows, so inactive rows need not
    // be initialized. The legacy fallback retains its TILE in-place behavior.
    const bool x_is_row_major = dispatched_buffer.layout() == tt::tt_metal::Layout::ROW_MAJOR;
    const ttnn::Tensor output =
        (use_moe_fused_swiglu || x_is_row_major)
            ? ttnn::empty(
                  dispatched_buffer.logical_shape(),
                  tt::tt_metal::DataType::BFLOAT8_B,
                  tt::tt_metal::Layout::TILE,
                  dispatched_buffer.device(),
                  tt::tt_metal::MemoryConfig{
                      tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM})
            : dispatched_buffer;
    const uint32_t m_tiles = (max_dispatched_tokens_per_expert + 31) / 32;

    // This kernel manages these flags itself. Preserve the numerical settings
    // the caller chose while normalizing the implementation-only settings.
    auto fused_compute_kernel_config = compute_kernel_config.value_or(ttnn::DeviceComputeKernelConfig{});
    fused_compute_kernel_config.fp32_dest_acc_en = false;
    fused_compute_kernel_config.packer_l1_acc = false;
    fused_compute_kernel_config.dst_full_sync_en = false;
    // This is the tuned rectangular launch shape for the routed-expert kernel;
    // the complete worker grid has a different geometry and is not equivalent.
    const auto fused_core_grid = tt::tt_metal::CoreCoord{11, 8};
    for (uint32_t local_expert = 0; local_expert < experts_per_chip; ++local_expert) {
        if (use_moe_fused_swiglu) {
            ttnn::operations::experimental::deepseek_prefill::moe_fused_swiglu::moe_fused_swiglu(
                dispatched_buffer,
                gate_projs[local_expert],
                up_projs[local_expert],
                down_projs[local_expert],
                expert_token_counts,
                global_expert_idx_table,
                local_expert,
                m_tiles,
                std::nullopt,
                std::nullopt,
                fused_compute_kernel_config,
                fused_core_grid,
                output,
                expert_region_offsets,
                /*read_x_at_offset=*/true);
            continue;
        }
        unified_routed_expert_ffn(
            dispatched_buffer,
            gate_projs[local_expert],
            up_projs[local_expert],
            down_projs[local_expert],
            expert_token_counts,
            global_expert_idx_table,
            local_expert,
            compute_kernel_config,
            output,
            expert_region_offsets,
            m_tiles,
            /*read_x_at_offset=*/true,
            x_is_row_major,
            activation,
            has_bias ? std::optional<ttnn::Tensor>((*gate_biases)[local_expert]) : std::nullopt,
            has_bias ? std::optional<ttnn::Tensor>((*up_biases)[local_expert]) : std::nullopt,
            has_bias ? std::optional<ttnn::Tensor>((*down_biases)[local_expert]) : std::nullopt);
    }
    return output;
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::unified_routed_expert_ffn
