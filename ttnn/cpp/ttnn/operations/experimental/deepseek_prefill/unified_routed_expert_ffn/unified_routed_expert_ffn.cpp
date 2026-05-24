// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "unified_routed_expert_ffn.hpp"

#include "device/unified_routed_expert_ffn_device_operation.hpp"
#include "tt-metalium/math.hpp"
#include "ttnn/operations/data_movement/pad/pad.hpp"
#include "ttnn/operations/data_movement/slice/slice.hpp"
#include "ttnn/operations/experimental/deepseek_prefill/extract/extract.hpp"
#include "ttnn/operations/experimental/deepseek_prefill/insert/insert.hpp"
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
    const std::optional<ttnn::Tensor>& output) {
    // Block-sharded path: delegate to production's routed_expert_ffn which
    // does the 4-op chain (gate matmul -> block-sharded L1, up matmul ->
    // block-sharded L1, silu * up -> L1 interleaved, down matmul) plus
    // chunked-M handling for M_tiles > 64. Matches production perf (529 μs
    // for 2k summed across the 4 inner ops on DS-V3 dims).
    //
    // The counts / global_expert_idx_table / local_expert_id args are unused
    // on this path — they were the device-side count-skipping interface for
    // the custom Program path (now defunct). All M tokens are processed.
    (void)counts;
    (void)global_expert_idx_table;
    (void)local_expert_id;
    return ttnn::routed_expert_ffn(
        x,
        gate_proj,
        up_proj,
        down_proj,
        compute_kernel_config,
        output.has_value() ? std::optional<ttnn::Tensor>(*output) : std::nullopt);
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
    const std::optional<const ttnn::DeviceComputeKernelConfig>& compute_kernel_config) {
    TT_FATAL(
        gate_projs.size() == up_projs.size() && gate_projs.size() == down_projs.size(),
        "gate/up/down projection lists must have the same length (got {}, {}, {})",
        gate_projs.size(),
        up_projs.size(),
        down_projs.size());
    const uint32_t experts_per_chip = static_cast<uint32_t>(gate_projs.size());
    TT_FATAL(experts_per_chip > 0, "Need at least one expert per chip");

    // The per-expert loop runs entirely C++-side. For each local expert we:
    //   1. extract its tokens out of the dispatched buffer (returns the full
    //      max_dispatched_tokens_per_expert-sized buffer; the first count
    //      rows hold this expert's valid tokens, the rest is dispatch
    //      padding).
    //   2. run the unified routed-expert FFN. The op reads counts and the
    //      local->global idx mapping on-device and bounds its chunk loop to
    //      ceil(count / chunk_M_tiles) — no host-side sync, no per-expert
    //      narrow.
    //   3. insert the FFN output back into the expert_outputs buffer at this
    //      expert's region.
    // The unified FFN's chunk loop uses chunk_M_tiles in {16, 24, 32, 40, 48,
    // 56, 64} (per_core_M >= 2 across the 8-row grid). For M_tiles_full to
    // divide some chunk_M_tiles >= 16, the input must have M_tiles_full % 16
    // == 0 (i.e. M rows must be a multiple of 16*TILE_H = 512). When the
    // dispatched buffer size isn't already aligned (e.g. 1600/3200 rows), pad
    // the extracted tokens up to the next 512-row boundary and slice back
    // after the FFN. The on-device count-bounded chunk loop ensures the
    // padded chunks do no real work.
    constexpr uint32_t kRowsAlignment = 512;  // 16 tiles * 32 rows/tile
    const uint32_t padded_rows =
        ((max_dispatched_tokens_per_expert + kRowsAlignment - 1) / kRowsAlignment) * kRowsAlignment;
    const bool needs_pad = (padded_rows != max_dispatched_tokens_per_expert);

    auto expert_outputs = dispatched_buffer;
    for (uint32_t local_expert = 0; local_expert < experts_per_chip; ++local_expert) {
        auto tokens = ttnn::extract(
            dispatched_buffer,
            expert_region_offsets,
            expert_token_counts,
            global_expert_idx_table,
            local_expert,
            max_dispatched_tokens_per_expert);

        if (needs_pad) {
            tokens = ttnn::pad(
                tokens,
                ttnn::SmallVector<operations::data_movement::PadSpecDim>{
                    {0, padded_rows - max_dispatched_tokens_per_expert}, {0, 0}},
                /*value=*/0.0f,
                /*use_multicore=*/true);
        }

        auto ffn_out = unified_routed_expert_ffn(
            tokens,
            gate_projs[local_expert],
            up_projs[local_expert],
            down_projs[local_expert],
            expert_token_counts,
            global_expert_idx_table,
            local_expert,
            compute_kernel_config,
            std::nullopt);

        if (needs_pad) {
            const ttnn::SmallVector<uint32_t> begins{0, 0};
            const ttnn::SmallVector<uint32_t> ends{
                max_dispatched_tokens_per_expert, static_cast<uint32_t>(ffn_out.padded_shape()[-1])};
            const ttnn::SmallVector<uint32_t> step{1, 1};
            ffn_out = ttnn::slice(ffn_out, begins, ends, step);
        }

        expert_outputs = ttnn::insert(
            expert_outputs, ffn_out, expert_region_offsets, expert_token_counts, global_expert_idx_table, local_expert);
    }

    return expert_outputs;
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::unified_routed_expert_ffn
