// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

// Builds this chip's [local_real_tokens, pad_side] MoE padding-config row entirely on-device.
//
// The per-chunk values arrive as two 1-element uint32 DRAM tensors (raw addresses in common args 3
// and 4) — `actual_start` (absolute KV position of this chunk's first real token) and `actual_end`
// (one past its last real token). Reading them on-device keeps them off the host dispatch path, so
// this op is trace-safe: one captured program recomputes the correct config on every replay.
//
// Why this exists: the host builder (TtMoEGatePrefill.build_padding_config) ends in a ttnn.from_torch,
// which is an illegal host->device write inside a trace capture. Consumers (moe_grouped_topk, dispatch)
// already read padding_config as a device tensor, so moving only its PRODUCER on-device makes the
// whole path traceable with no change downstream.
//
// Compile args: [0]=cb_id_out, [1]=cb_id_meta, [2]=pad_side, [3..]=config accessor, then ONE metadata
// accessor (the two 1-element tensors share an identical layout, so one accessor serves both reads).
//
// Common runtime args: [0]=my_sp_coord, [1]=sp_factor, [2]=tokens_per_chip,
//                      [3]=actual_start tensor DRAM addr, [4]=actual_end tensor DRAM addr.
// Per-core runtime args: [0]=config output buffer address (Buffer* binding).

namespace {

// Number of positions in [base, base+len) that fall below `valid_end`. All-unsigned (no signed
// underflow): the rotated layout guarantees every position is >= actual_start, so only the upper
// bound matters.
inline uint32_t prefix_count(uint32_t valid_end, uint32_t base, uint32_t len) {
    if (valid_end <= base) {
        return 0;
    }
    const uint32_t n = valid_end - base;
    return n < len ? n : len;
}

}  // namespace

void kernel_main() {
    const uint32_t config_addr = get_arg_val<uint32_t>(0);

    const uint32_t my_sp_coord = get_common_arg_val<uint32_t>(0);
    const uint32_t sp_factor = get_common_arg_val<uint32_t>(1);
    const uint32_t tokens_per_chip = get_common_arg_val<uint32_t>(2);
    const uint32_t actual_start_addr = get_common_arg_val<uint32_t>(3);
    const uint32_t actual_end_addr = get_common_arg_val<uint32_t>(4);

    constexpr uint32_t cb_id_out = get_compile_time_arg_val(0);
    constexpr uint32_t cb_id_meta = get_compile_time_arg_val(1);
    constexpr uint32_t pad_side = get_compile_time_arg_val(2);
    constexpr auto config_args = TensorAccessorArgs<3>();
    constexpr auto meta_args = TensorAccessorArgs<config_args.next_compile_time_args_offset()>();

    constexpr uint32_t kMetadataReadBytes = 4;
    constexpr uint32_t onepage = 1;

    Noc noc;

    // ---- read actual_start / actual_end (element [0] of each 1-element tensor) ----
    // Both reads target dst offset 0 of the same scratch slot (a 4-byte DRAM read into a
    // non-16B-aligned dst offset lands wrong), so read start, extract, then overwrite with end.
    // Single reserve_back/push_back: a second reserve_back on a single-page CB with no intervening
    // pop would deadlock.
    CircularBuffer cb_meta(cb_id_meta);
    cb_meta.reserve_back(onepage);

    const auto s_start = TensorAccessor(meta_args, actual_start_addr);
    noc.async_read(s_start, cb_meta, kMetadataReadBytes, {.page_id = 0}, {.offset_bytes = 0});
    noc.async_read_barrier();
    // The metadata tensors sit at FIXED DRAM addresses reused every chunk, so the RISC data cache may
    // still hold the previous chunk's value for this L1 line (the barrier orders the DMA; volatile
    // still reads through the cache). Force a refetch, else a stale read silently produces the prior
    // chunk's config.
    invalidate_l1_cache();
    const uint32_t actual_start = CoreLocalMem<volatile uint32_t>(cb_meta.get_write_ptr())[0];

    const auto s_end = TensorAccessor(meta_args, actual_end_addr);
    noc.async_read(s_end, cb_meta, kMetadataReadBytes, {.page_id = 0}, {.offset_bytes = 0});
    noc.async_read_barrier();
    invalidate_l1_cache();  // same fresh-metadata refetch as above
    const uint32_t actual_end = CoreLocalMem<volatile uint32_t>(cb_meta.get_write_ptr())[0];
    cb_meta.push_back(onepage);

    // ---- per-chip real-token count under the KV-pad-aware rotation ----
    // Chip c's chunk_local rows carry global positions that are a strictly increasing sequence, so its
    // real rows are a contiguous PREFIX and one count describes the split. The rotation mirrors
    // update_padded_kv_cache's writer exactly (same boundary_slab/boundary_chip/boundary_offset), and
    // the sequence is piecewise-linear in the local row with at most two segments — so the count is
    // closed-form, no per-row loop.
    //
    // valid_end == actual_start + actual_isl == actual_end, so the ISL never has to be formed.
    const uint32_t valid_end = actual_end;
    const uint32_t chunk_global = sp_factor * tokens_per_chip;
    const uint32_t boundary_slab = actual_start / chunk_global;
    const uint32_t boundary_chip = (actual_start / tokens_per_chip) % sp_factor;
    const uint32_t boundary_offset = actual_start % tokens_per_chip;

    uint32_t local_real_tokens;
    if (my_sp_coord != boundary_chip) {
        // One linear run: chips before the boundary have had their pad consumed and start a slab
        // later; chips after it stay at the current slab base.
        const uint32_t slab = (my_sp_coord < boundary_chip) ? (boundary_slab + 1) : boundary_slab;
        const uint32_t base = slab * chunk_global + my_sp_coord * tokens_per_chip;
        local_real_tokens = prefix_count(valid_end, base, tokens_per_chip);
    } else {
        // The boundary chip starts mid-slab at boundary_offset, so its rows wrap into the next slab:
        // two linear runs.
        const uint32_t base_lo = boundary_slab * chunk_global + my_sp_coord * tokens_per_chip + boundary_offset;
        const uint32_t len_lo = tokens_per_chip - boundary_offset;
        const uint32_t base_hi = (boundary_slab + 1) * chunk_global + my_sp_coord * tokens_per_chip;
        const uint32_t len_hi = boundary_offset;
        local_real_tokens = prefix_count(valid_end, base_lo, len_lo) + prefix_count(valid_end, base_hi, len_hi);
    }

    // ---- write this chip's [local_real_tokens, pad_side] row ----
    // The consumers read page 0 and take element [0] as the real-token count and [1] as the pad side.
    const uint32_t out_page_bytes = get_local_cb_interface(cb_id_out).fifo_page_size;
    CircularBuffer cb_out(cb_id_out);
    cb_out.reserve_back(onepage);
    {
        // Zero the whole page first: the row may be padded up to the buffer's aligned page size, and
        // the full page is written below, so trailing bytes must be deterministic rather than stale L1.
        auto out = CoreLocalMem<volatile uint32_t>(cb_out.get_write_ptr());
        for (uint32_t i = 0; i < out_page_bytes / sizeof(uint32_t); ++i) {
            out[i] = 0;
        }
        out[0] = local_real_tokens;
        out[1] = pad_side;
    }
    cb_out.push_back(onepage);

    cb_out.wait_front(onepage);
    const auto s_out = TensorAccessor(config_args, config_addr);
    noc.async_write(cb_out, s_out, out_page_bytes, {}, {.page_id = 0});
    noc.async_write_barrier();
    cb_out.pop_front(onepage);
}
