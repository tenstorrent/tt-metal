// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Vol2col data-movement primitives for the fused NP conv3d reader (conv3d_reader_vol2col.cpp).
//
// The interior (non-halo) gather + vol2col primitives here are kept in lock-step with the upstream
// conv3d reader (ttnn/cpp/ttnn/operations/experimental/conv3d/device/kernels/reader_vol2col.cpp):
// trid-ring pipelining, coalesced bank-major DRAM bursts, and DRAM-read staging.  gather_rows_halo
// is the neighbor-pad addition — boundary blocks read the cross-device halo buffer instead of
// clamp/zero padding — ported to the same experimental::CB / Noc abstractions.

#pragma once

#include <stdint.h>
#include <type_traits>
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>
#include <ttnn/operations/experimental/conv3d/device/kernels/conv3d_gather_tuning.hpp>

// Pre-zero CB pages via NOC DMA from MEM_ZEROS so tile-alignment padding is zero.
// Uses MEM_ZEROS_SIZE-aligned transactions (same pattern as zero_out_tiles in conv_reader_common.hpp).
// padded_page_bytes must be a multiple of 16 to guarantee remainder alignment.
template <uint32_t padded_page_bytes, typename Dst>
FORCE_INLINE void pre_zero_pages(Noc noc, const Dst& dst, uint32_t offset, uint32_t num_pages) {
    static_assert(padded_page_bytes % 16 == 0, "CB page size must be 16-byte aligned for NOC transactions");
    uint32_t total = num_pages * padded_page_bytes;
    experimental::set_read_state<MEM_ZEROS_SIZE>(noc, MEM_ZEROS_BASE);
    while (total >= MEM_ZEROS_SIZE) {
        experimental::read_with_state(noc, dst, MEM_ZEROS_BASE, {.offset_bytes = offset});
        offset += MEM_ZEROS_SIZE;
        total -= MEM_ZEROS_SIZE;
    }
    if (total > 0) {
        UnicastEndpoint self_ep;
        noc.async_read(
            self_ep, dst, total, experimental::local_addr(MEM_ZEROS_BASE, noc.get_noc_id()), {.offset_bytes = offset});
    }
    noc.async_read_barrier();
}

inline int32_t clampIndex(int32_t idx, int32_t lower_bound, int32_t upper_bound) {
    // If we're doing replicate padding, clamp idx into [lower_bound, upper_bound].
    if (idx < lower_bound) {
        return lower_bound;
    }
    if (idx > upper_bound) {
        return upper_bound;
    }
    return idx;
}

template <uint32_t in_row_size_bytes, typename Dst>
inline void zeroPad(Noc noc, const Dst& dst, uint32_t offset) {
    // Zero-fill from MEM_ZEROS
    constexpr uint32_t num_full_reads = in_row_size_bytes / MEM_ZEROS_SIZE;
    constexpr uint32_t partial_read_size = in_row_size_bytes % MEM_ZEROS_SIZE;

    UnicastEndpoint self_ep;
    const auto zeros_src = experimental::local_addr(MEM_ZEROS_BASE, noc.get_noc_id());

    for (uint32_t i = 0; i < num_full_reads; ++i) {
        noc.async_read(self_ep, dst, MEM_ZEROS_SIZE, zeros_src, {.offset_bytes = offset});
        offset += MEM_ZEROS_SIZE;
    }
    if (partial_read_size > 0) {
        noc.async_read(self_ep, dst, partial_read_size, zeros_src, {.offset_bytes = offset});
    }
}

template <typename Reader>
FORCE_INLINE void read_input_row(
    Noc noc,
    const Reader& reader,
    uint32_t page_idx,
    uint32_t c_in_offset_bytes,
    uint32_t in_row_size_bytes,
    const experimental::CB& dst,
    uint32_t dst_offset,
    uint32_t size_bytes) {
    if constexpr (Reader::DSpec::tensor_shape_static) {
        if constexpr ((reader.dspec().rank() > 1) && (reader.dspec().tensor_shape()[1] > 1)) {
            // Width/block sharded RowMajor tensors may split a logical row across multiple pages.
            // Height-sharded inputs keep a single page per row and should use the direct path below.
            constexpr uint32_t width_in_pages = reader.dspec().tensor_shape()[1];
            const uint32_t col_page_idx = c_in_offset_bytes / in_row_size_bytes;
            const uint32_t in_offset_bytes = c_in_offset_bytes - (col_page_idx * in_row_size_bytes);
            ASSERT(col_page_idx < width_in_pages);
            const uint32_t in_page_id = page_idx * width_in_pages + col_page_idx;
            noc.async_read(
                reader,
                dst,
                size_bytes,
                {.page_id = in_page_id, .offset_bytes = in_offset_bytes},
                {.offset_bytes = dst_offset});
            return;
        }
    }

    noc.async_read(
        reader,
        dst,
        size_bytes,
        {.page_id = page_idx, .offset_bytes = c_in_offset_bytes},
        {.offset_bytes = dst_offset});
}

template <uint32_t dram_read_alignment, typename Reader>
FORCE_INLINE void read_input_row_staged_from_dram(
    Noc noc,
    const Reader& reader,
    uint32_t page_idx,
    uint32_t c_in_offset_bytes,
    const experimental::CB& dst,
    uint32_t dst_offset,
    uint32_t size_bytes,
    const experimental::CB& scratch_cb) {
    static_assert(Reader::DSpec::is_interleaved && Reader::DSpec::is_dram);
    static_assert(dram_read_alignment > 0);
    static_assert((dram_read_alignment & (dram_read_alignment - 1)) == 0);
    constexpr uint32_t alignment_mask = dram_read_alignment - 1;

    // The factory enables this helper only when DRAM pages are aligned but the split
    // C-in slice is smaller than the DRAM read alignment.  Always stage through an
    // aligned scratch window so every read follows the same alignment-safe path.
    const uint64_t src_noc_addr = reader.get_noc_addr(page_idx, c_in_offset_bytes, noc.get_noc_id());
    const uint32_t src_align_offset = static_cast<uint32_t>(src_noc_addr) & alignment_mask;
    const uint64_t aligned_src_noc_addr = src_noc_addr - src_align_offset;
    const uint32_t aligned_read_size = (src_align_offset + size_bytes + alignment_mask) & ~alignment_mask;

    const uint32_t scratch_unaligned = scratch_cb.get_write_ptr();
    const uint32_t scratch_l1_addr = (scratch_unaligned + alignment_mask) & ~alignment_mask;

    // Aligning backward gives us a raw full NOC address rather than a reader page/offset pair.
    noc_async_read<NOC_MAX_BURST_SIZE + 1, true>(
        aligned_src_noc_addr, scratch_l1_addr, aligned_read_size, noc.get_noc_id());
    // Scratch must be populated before it is used as the source for the local L1 copy.
    noc.async_read_barrier();

    // The scratch CB is a single staging page reused by each call.  After issuing the
    // local copy, drain it before returning so the next staged read cannot overwrite
    // scratch while it is still the source for an in-flight L1->L1 read.
    UnicastEndpoint self_ep;
    noc.async_read(
        self_ep,
        dst,
        size_bytes,
        experimental::local_addr(scratch_l1_addr + src_align_offset, noc.get_noc_id()),
        {.offset_bytes = dst_offset});
    noc.async_read_barrier();
}

template <bool EnableDramReadStaging, uint32_t dram_read_alignment, typename Reader>
FORCE_INLINE void read_input_row_maybe_staged(
    Noc noc,
    const Reader& reader,
    uint32_t page_idx,
    uint32_t c_in_offset_bytes,
    uint32_t in_row_size_bytes,
    const experimental::CB& dst,
    uint32_t dst_offset,
    uint32_t size_bytes,
    const experimental::CB& scratch_cb) {
    if constexpr (EnableDramReadStaging && Reader::DSpec::is_interleaved && Reader::DSpec::is_dram) {
        read_input_row_staged_from_dram<dram_read_alignment>(
            noc, reader, page_idx, c_in_offset_bytes, dst, dst_offset, size_bytes, scratch_cb);
    } else {
        read_input_row(noc, reader, page_idx, c_in_offset_bytes, in_row_size_bytes, dst, dst_offset, size_bytes);
    }
}

// Manages chunked CB writes: reserves TILE_HEIGHT pages, tracks patches written,
// pushes when full, and flushes remaining at the end of a block.
template <uint32_t cb_id, uint32_t padded_page_bytes, uint32_t patch_pad_bytes>
struct ChunkWriter {
    static constexpr uint32_t chunk_max = 32;  // TILE_HEIGHT
    uint32_t remaining;
    uint32_t chunk_size;
    uint32_t in_chunk;
    uint32_t write_offset;
    Noc noc;
    experimental::CB cb;

    ChunkWriter(Noc n) : noc(n), cb(cb_id) {}

    void init(uint32_t total_patches) {
        remaining = total_patches;
        in_chunk = 0;
        chunk_size = remaining < chunk_max ? remaining : chunk_max;
        cb.reserve_back(chunk_size);
        write_offset = 0;
        if constexpr (patch_pad_bytes > 0) {
            pre_zero_pages<padded_page_bytes>(noc, cb, 0, chunk_size);
        }
    }

    // Call after writing one patch to write_offset.
    // Returns true when a chunk was pushed and a new one reserved — caller must restore NOC packet state.
    bool advance() {
        if constexpr (patch_pad_bytes > 0) {
            write_offset += patch_pad_bytes;
        }
        in_chunk++;
        if (in_chunk == chunk_size) {
            noc.async_read_barrier();
            cb.push_back(chunk_size);
            remaining -= chunk_size;
            in_chunk = 0;
            if (remaining > 0) {
                chunk_size = remaining < chunk_max ? remaining : chunk_max;
                cb.reserve_back(chunk_size);
                write_offset = 0;
                if constexpr (patch_pad_bytes > 0) {
                    pre_zero_pages<padded_page_bytes>(noc, cb, 0, chunk_size);
                }
                return true;  // pushed and re-reserved — caller may need to restore NOC state
            }
        }
        return false;
    }

    void flush() {
        if (remaining > 0) {
            if (in_chunk > 0) {
                noc.async_read_barrier();
                cb.push_back(chunk_size);
                remaining -= chunk_size;
            }
            while (remaining > 0) {
                const uint32_t cur = remaining < chunk_max ? remaining : chunk_max;
                cb.reserve_back(cur);
                cb.push_back(cur);
                remaining -= cur;
            }
        }
    }
};

template <uint32_t C_in_block_bytes>
struct CoalescedRowLayout {
    uint32_t slot_chunk_offset[NUM_DRAM_BANKS];
    uint32_t slot_chunk_pages[NUM_DRAM_BANKS];

    FORCE_INLINE void compute(uint32_t W_shard_cur) {
        uint32_t offset = 0;
        for (uint32_t slot = 0; slot < NUM_DRAM_BANKS; slot++) {
            slot_chunk_offset[slot] = offset;
            const uint32_t pages = slot < W_shard_cur ? ((W_shard_cur - 1 - slot) / NUM_DRAM_BANKS) + 1 : 0;
            slot_chunk_pages[slot] = pages;
            offset += pages * C_in_block_bytes;
        }
    }

    FORCE_INLINE uint32_t offset_for_w(uint32_t w_natural) const {
        const uint32_t slot = w_natural % NUM_DRAM_BANKS;
        const uint32_t local_idx = w_natural / NUM_DRAM_BANKS;
        return slot_chunk_offset[slot] + local_idx * C_in_block_bytes;
    }
};

// Copy one (t, h) row of patches from the L1 shard into the vol2col CB.
// Iterates over w positions in [w_block, w_block_end), extracting kT×kH×kW patches
// via one_packet NOC reads.  Calls chunk.advance() after each patch.
template <
    uint32_t kT,
    uint32_t kH,
    uint32_t kW,
    uint32_t C_in_block_bytes,
    uint32_t H_shard_max_W_shard_max,
    uint32_t W_shard_max,
    uint32_t stride_w,
    uint32_t cb_id,
    uint32_t padded_page_bytes,
    uint32_t patch_pad_bytes>
void vol2col_shard_to_cb(
    Noc noc,
    uint32_t shard_l1_base,
    uint32_t t_base,
    uint32_t h_base,
    uint32_t w_block,
    uint32_t w_block_end,
    ChunkWriter<cb_id, padded_page_bytes, patch_pad_bytes>& chunk) {
    constexpr uint32_t kW_bytes = kW * C_in_block_bytes;
    experimental::set_read_state<kW_bytes>(noc, shard_l1_base);
    for (uint32_t w = w_block; w < w_block_end; w++) {
        const uint32_t w_base = (w - w_block) * stride_w;
        for (uint32_t kt = 0; kt < kT; kt++) {
            const uint32_t t_local = t_base + kt;
            for (uint32_t kh = 0; kh < kH; kh++) {
                const uint32_t h_local = h_base + kh;
                const uint32_t shard_offset =
                    (t_local * H_shard_max_W_shard_max + h_local * W_shard_max + w_base) * C_in_block_bytes;
                experimental::read_with_state(
                    noc, chunk.cb, shard_l1_base + shard_offset, {.offset_bytes = chunk.write_offset});
                chunk.write_offset += kW_bytes;
            }
        }
        if (chunk.advance()) {
            experimental::set_read_state<kW_bytes>(noc, shard_l1_base);
        }
    }
}

// Shift retained columns to the start of each shard row for sliding-window W reuse.
// With stride_w, adjacent w_blocks overlap by max(0, kW - stride_w) columns, not kW-1.
// After the shift, only (W_shard_cur - overlap) new columns need to be gathered from DRAM.
template <
    uint32_t C_in_block_bytes,
    uint32_t H_shard_max_W_shard_max,
    uint32_t W_shard_max,
    uint32_t kW,
    uint32_t stride_w>
void shift_retained_w_columns(
    Noc noc, const experimental::CB& shard_cb, uint32_t T_shard_cur, uint32_t h_rows_gathered) {
    constexpr uint32_t overlap_w = kW > stride_w ? kW - stride_w : 0;
    constexpr uint32_t shift_bytes = overlap_w * C_in_block_bytes;
    constexpr uint32_t src_off = (W_shard_max - overlap_w) * C_in_block_bytes;
    UnicastEndpoint self_ep;
    const uint32_t shard_l1_base = shard_cb.get_write_ptr();
    for (uint32_t t_local = 0; t_local < T_shard_cur; t_local++) {
        for (uint32_t h_local = 0; h_local < h_rows_gathered; h_local++) {
            const uint32_t row_base = (t_local * H_shard_max_W_shard_max + h_local * W_shard_max) * C_in_block_bytes;
            noc.async_read(
                self_ep,
                shard_cb,
                shift_bytes,
                experimental::local_addr(shard_l1_base + row_base + src_off, noc.get_noc_id()),
                {.offset_bytes = row_base});
        }
    }
    noc.async_read_barrier();
}

// Gather rows from DRAM into the L1 shard buffer.
// When check_padding=false, all positions are known to be in-bounds — skip per-position
// boundary checks and clamp/zeroPad logic (~3-6 RISC-V cycles saved per position).
//
// Trid pipeline: each issued read is tagged with `trid = (issued % N_TRIDS) + 1`.  Once
// `issued >= N_TRIDS`, issuing read `i` first blocks on trid `((i - N_TRIDS) % N_TRIDS) + 1`
// to free that slot.  After the loop, drain by barriering on each in-flight trid.  This
// bounds NoC outstanding reads to N_TRIDS via per-trid waits so later reads continue while
// earlier ones drain, instead of a single trailing barrier that would serialize the drain.
//
// Per-call fast path: when this gather call has fewer than `kGatherFastPathReadCutoff`
// reads it issues untagged + a single trailing barrier, since at that count the ring
// overhead would dominate.  Note this is independent from the host gate:
// the host decides whether N_TRIDS is non-zero for the whole shape; this decides whether
// any individual call (e.g. a small slice along the h boundary) skips the ring locally.
// trid is reset to 0 before returning so callers using untagged reads see clean cmd_buf
// state.
template <
    uint32_t C_in_block_bytes,
    bool is_padding_zeros,
    uint32_t H_shard_max_W_shard_max,
    uint32_t W_shard_max,
    uint32_t T_in,
    uint32_t H_in,
    uint32_t W_in,
    uint32_t H_in_W_in,
    uint32_t in_row_size_bytes,
    bool check_padding,
    uint32_t N_TRIDS,
    bool EnableDramReadStaging,
    uint32_t dram_read_alignment,
    typename Reader>
void gather_rows_to_shard(
    Noc noc,
    const Reader& in_reader,
    const experimental::CB& shard_cb,
    const experimental::CB& dram_read_scratch_cb,
    uint32_t batch_page_base,
    uint32_t c_in_offset_bytes,
    int32_t t_shard_start,
    uint32_t T_shard_cur,
    int32_t h_shard_start,
    uint32_t h_start,
    uint32_t h_end,
    int32_t w_shard_start,
    uint32_t w_col_start,
    uint32_t w_count) {
    // N_TRIDS == 0 is a compile-time sentinel: the host-side classifier disabled the trid
    // ring for this shape (compute-bound or scratch-backed reader mode — see
    // conv3d_program_factory.cpp).  In that case all ring code is constexpr-elided and the
    // function reduces to issue-all + single trailing barrier.  When non-zero, N_TRIDS is
    // always one of the conv3d_gather_tuning depths; the upper bound matches the underlying
    // NoC trid-id range.
    static_assert(
        N_TRIDS == 0 || N_TRIDS == conv3d_gather_tuning::kGatherTridDepthLow ||
        N_TRIDS == conv3d_gather_tuning::kGatherTridDepthHigh);
    static_assert(conv3d_gather_tuning::kGatherTridDepthHigh <= 8, "trid id range exceeded");
    auto trid_for = [](uint32_t i) -> uint32_t {
        if constexpr (N_TRIDS == 0) {
            return 0;  // unused
        } else {
            return (i % N_TRIDS) + 1;
        }
    };
    // Per-call fast path: if this call won't fill at least two ring cycles, the
    // post-loop drain of N_TRIDS barriers is wasted.  Floor at 2 * N_TRIDS reads.
    constexpr uint32_t kGatherFastPathReadCutoff = 2 * N_TRIDS;
    const uint32_t total_reads_estimate = T_shard_cur * (h_end - h_start) * w_count;
    [[maybe_unused]] const bool use_ring = (N_TRIDS != 0) && (total_reads_estimate >= kGatherFastPathReadCutoff);
    [[maybe_unused]] uint32_t issued = 0;
    for (uint32_t t_local = 0; t_local < T_shard_cur; t_local++) {
        const int32_t t_in = t_shard_start + static_cast<int32_t>(t_local);
        [[maybe_unused]] const bool t_outside = check_padding && (t_in < 0 || t_in >= static_cast<int32_t>(T_in));
        [[maybe_unused]] const int32_t t_clamped =
            check_padding ? clampIndex(t_in, 0, static_cast<int32_t>(T_in) - 1) : t_in;
        for (uint32_t h_local = h_start; h_local < h_end; h_local++) {
            const int32_t h_in = h_shard_start + static_cast<int32_t>(h_local);
            [[maybe_unused]] const bool h_outside = check_padding && (h_in < 0 || h_in >= static_cast<int32_t>(H_in));
            [[maybe_unused]] const int32_t h_clamped =
                check_padding ? clampIndex(h_in, 0, static_cast<int32_t>(H_in) - 1) : h_in;
            uint32_t shard_offset =
                (t_local * H_shard_max_W_shard_max + h_local * W_shard_max + w_col_start) * C_in_block_bytes;
            for (uint32_t w_idx = 0; w_idx < w_count; w_idx++) {
                const int32_t w_in = w_shard_start + static_cast<int32_t>(w_col_start + w_idx);
                // Trid ring: free this slot if it's been used N_TRIDS reads ago, then tag the
                // upcoming read with this iteration's trid.  Both N_TRIDS==0 (host-disabled)
                // and use_ring=false (small-burst fallback) bypass; the constexpr branch keeps
                // the disabled binary clean.
                if constexpr (N_TRIDS != 0) {
                    if (use_ring) {
                        if (issued >= N_TRIDS) {
                            experimental::async_read_barrier_with_trid(noc, trid_for(issued - N_TRIDS));
                        }
                        experimental::set_read_trid(noc, trid_for(issued));
                    }
                }
                if constexpr (check_padding) {
                    const bool w_outside = (w_in < 0 || w_in >= static_cast<int32_t>(W_in));
                    const bool in_padding = t_outside || h_outside || w_outside;
                    if (in_padding) {
                        if constexpr (is_padding_zeros) {
                            zeroPad<C_in_block_bytes>(noc, shard_cb, shard_offset);
                        } else {
                            const int32_t w_clamped = clampIndex(w_in, 0, static_cast<int32_t>(W_in) - 1);
                            const uint32_t page_idx = batch_page_base + static_cast<uint32_t>(t_clamped) * H_in_W_in +
                                                      static_cast<uint32_t>(h_clamped) * W_in +
                                                      static_cast<uint32_t>(w_clamped);
                            read_input_row_maybe_staged<EnableDramReadStaging, dram_read_alignment>(
                                noc,
                                in_reader,
                                page_idx,
                                c_in_offset_bytes,
                                in_row_size_bytes,
                                shard_cb,
                                shard_offset,
                                C_in_block_bytes,
                                dram_read_scratch_cb);
                        }
                    } else {
                        const uint32_t page_idx = batch_page_base + static_cast<uint32_t>(t_in) * H_in_W_in +
                                                  static_cast<uint32_t>(h_in) * W_in + static_cast<uint32_t>(w_in);
                        read_input_row_maybe_staged<EnableDramReadStaging, dram_read_alignment>(
                            noc,
                            in_reader,
                            page_idx,
                            c_in_offset_bytes,
                            in_row_size_bytes,
                            shard_cb,
                            shard_offset,
                            C_in_block_bytes,
                            dram_read_scratch_cb);
                    }
                } else {
                    // Fast path: no padding checks
                    const uint32_t page_idx = batch_page_base + static_cast<uint32_t>(t_in) * H_in_W_in +
                                              static_cast<uint32_t>(h_in) * W_in + static_cast<uint32_t>(w_in);
                    read_input_row_maybe_staged<EnableDramReadStaging, dram_read_alignment>(
                        noc,
                        in_reader,
                        page_idx,
                        c_in_offset_bytes,
                        in_row_size_bytes,
                        shard_cb,
                        shard_offset,
                        C_in_block_bytes,
                        dram_read_scratch_cb);
                }
                if constexpr (N_TRIDS != 0) {
                    if (use_ring) {
                        issued++;
                    }
                }
                shard_offset += C_in_block_bytes;
            }
        }
    }
    if constexpr (N_TRIDS != 0) {
        if (use_ring) {
            // Drain the in-flight trids in issue order.  After the loop, the most recently
            // issued read used trid_for(issued-1), and the oldest still in flight used
            // trid_for(issued - to_drain).
            const uint32_t to_drain = issued < N_TRIDS ? issued : N_TRIDS;
            for (uint32_t k = 0; k < to_drain; k++) {
                experimental::async_read_barrier_with_trid(noc, trid_for(issued - to_drain + k));
            }
            // Restore untagged state so vol2col / pre_zero / shift reads (which use trid 0)
            // don't get accounted against a stale per-trid counter.
            experimental::set_read_trid(noc, 0);
        } else {
            // Small-burst fallback: ring code never set a trid this call, so no reset needed.
            noc.async_read_barrier();
        }
    } else {
        // Host-disabled: ring code is fully constexpr-elided, trid was never touched.
        noc.async_read_barrier();
    }
}

// Coalesced shard gather reads each logical row into scratch in bank-major chunks, then
// deinterleaves scratch back into the natural shard layout with local L1->L1 copies.  The
// extra L1 traffic is intentional: perf sweeps showed the win comes from replacing
// many small DRAM reads with larger per-bank bursts and then paying a local reorder pass.
template <
    uint32_t C_in_block_bytes,
    uint32_t H_shard_max_W_shard_max,
    uint32_t W_shard_max,
    uint32_t W_in,
    uint32_t H_in_W_in,
    uint32_t GatherTrids,
    typename Reader>
void gather_rows_to_shard_coalesced(
    Noc noc,
    const Reader& in_reader,
    const experimental::CB& shard_cb,
    uint32_t shard_l1_base,
    uint32_t scratch_row_offset,
    uint32_t batch_page_base,
    [[maybe_unused]] uint32_t c_in_offset_bytes,
    int32_t t_shard_start,
    uint32_t T_shard_cur,
    int32_t h_shard_start,
    uint32_t h_start,
    uint32_t h_end,
    int32_t w_shard_start,
    uint32_t w_col_start,
    uint32_t w_count,
    uint32_t scratch_rows) {
    static_assert(Reader::DSpec::is_interleaved && Reader::DSpec::is_dram);
    static_assert(GatherTrids == 0, "Coalesced shard reads require gather trid ring disabled");
    ASSERT(c_in_offset_bytes == 0);
    ASSERT(h_end > h_start);
    ASSERT(w_count > NUM_DRAM_BANKS);
    ASSERT(scratch_rows > 0);

    CoalescedRowLayout<C_in_block_bytes> row_layout;
    row_layout.compute(w_count);

    const uint32_t h_count = h_end - h_start;
    const uint32_t total_rows = T_shard_cur * h_count;
    const uint32_t scratch_row_bytes = W_shard_max * C_in_block_bytes;

    for (uint32_t row_start = 0; row_start < total_rows; row_start += scratch_rows) {
        const uint32_t rows_cur = std::min(scratch_rows, total_rows - row_start);

        for (uint32_t scratch_row = 0; scratch_row < rows_cur; scratch_row++) {
            const uint32_t row_linear = row_start + scratch_row;
            const uint32_t t_local = row_linear / h_count;
            const uint32_t h_local = h_start + row_linear - t_local * h_count;
            const int32_t t_in = t_shard_start + static_cast<int32_t>(t_local);
            const int32_t h_in = h_shard_start + static_cast<int32_t>(h_local);
            const uint32_t page_base = batch_page_base + static_cast<uint32_t>(t_in) * H_in_W_in +
                                       static_cast<uint32_t>(h_in) * W_in + static_cast<uint32_t>(w_shard_start);
            const uint32_t scratch_base = scratch_row_offset + scratch_row * scratch_row_bytes;

            for (uint32_t slot = 0; slot < NUM_DRAM_BANKS; slot++) {
                const uint32_t pages = row_layout.slot_chunk_pages[slot];
                if (pages == 0) {
                    continue;
                }
                noc.async_read(
                    in_reader,
                    shard_cb,
                    pages * C_in_block_bytes,
                    {.page_id = page_base + w_col_start + slot, .offset_bytes = 0},
                    {.offset_bytes = scratch_base + row_layout.slot_chunk_offset[slot]});
            }
        }
        noc.async_read_barrier();

        if constexpr (C_in_block_bytes <= NOC_MAX_BURST_SIZE) {
            experimental::set_read_state<C_in_block_bytes>(noc, shard_l1_base + scratch_row_offset);
        }

        for (uint32_t scratch_row = 0; scratch_row < rows_cur; scratch_row++) {
            const uint32_t row_linear = row_start + scratch_row;
            const uint32_t t_local = row_linear / h_count;
            const uint32_t h_local = h_start + row_linear - t_local * h_count;
            const uint32_t row_base = (t_local * H_shard_max_W_shard_max + h_local * W_shard_max) * C_in_block_bytes;
            const uint32_t scratch_base = scratch_row_offset + scratch_row * scratch_row_bytes;
            for (uint32_t w_idx = 0; w_idx < w_count; w_idx++) {
                const uint32_t src_l1_addr = shard_l1_base + scratch_base + row_layout.offset_for_w(w_idx);
                const uint32_t dst_offset = row_base + (w_col_start + w_idx) * C_in_block_bytes;
                if constexpr (C_in_block_bytes <= NOC_MAX_BURST_SIZE) {
                    experimental::read_with_state(noc, shard_cb, src_l1_addr, {.offset_bytes = dst_offset});
                } else {
                    UnicastEndpoint self_ep;
                    noc.async_read(
                        self_ep,
                        shard_cb,
                        C_in_block_bytes,
                        experimental::local_addr(src_l1_addr, noc.get_noc_id()),
                        {.offset_bytes = dst_offset});
                }
            }
        }
        noc.async_read_barrier();
    }
}

// Dispatch to coalesced, in-bounds, or padded gather from one call site shape. This keeps the
// first-W incremental gather and the later H-incremental gather wired identically.
template <
    uint32_t C_in_block_bytes,
    bool is_padding_zeros,
    uint32_t H_shard_max_W_shard_max,
    uint32_t W_shard_max,
    uint32_t T_in,
    uint32_t H_in,
    uint32_t W_in,
    uint32_t H_in_W_in,
    uint32_t in_row_size_bytes,
    uint32_t GatherTrids,
    bool EnableCoalescedShardReads,
    bool EnableDramReadStaging,
    uint32_t dram_read_alignment,
    typename Reader>
void gather_rows_to_shard_selected(
    Noc noc,
    const Reader& in_reader,
    const experimental::CB& shard_cb,
    const experimental::CB& dram_read_scratch_cb,
    [[maybe_unused]] uint32_t shard_l1_base,
    [[maybe_unused]] uint32_t coalesced_scratch_offset,
    bool all_in_bounds,
    [[maybe_unused]] bool use_coalesced,
    uint32_t batch_page_base,
    uint32_t c_in_offset_bytes,
    int32_t t_shard_start,
    uint32_t T_shard_cur,
    int32_t h_shard_start,
    uint32_t h_start,
    uint32_t h_end,
    int32_t w_shard_start,
    uint32_t w_col_start,
    uint32_t w_count,
    [[maybe_unused]] uint32_t coalesced_scratch_rows) {
    if constexpr (EnableCoalescedShardReads) {
        if (use_coalesced) {
            ASSERT(all_in_bounds);
            gather_rows_to_shard_coalesced<
                C_in_block_bytes,
                H_shard_max_W_shard_max,
                W_shard_max,
                W_in,
                H_in_W_in,
                GatherTrids>(
                noc,
                in_reader,
                shard_cb,
                shard_l1_base,
                coalesced_scratch_offset,
                batch_page_base,
                c_in_offset_bytes,
                t_shard_start,
                T_shard_cur,
                h_shard_start,
                h_start,
                h_end,
                w_shard_start,
                w_col_start,
                w_count,
                coalesced_scratch_rows);
            return;
        }
    }

    // check_padding is a template arg on gather_rows_to_shard (compile-time elision of the
    // padding bounds-check + clamp/zero-pad branch in the hot inner loop), so it must be
    // dispatched as a constant. Generic lambda + bool-constant lifts the runtime
    // `all_in_bounds` to a compile-time value once and shares the call site.
    const auto do_gather = [&](auto check_padding_v) {
        gather_rows_to_shard<
            C_in_block_bytes,
            is_padding_zeros,
            H_shard_max_W_shard_max,
            W_shard_max,
            T_in,
            H_in,
            W_in,
            H_in_W_in,
            in_row_size_bytes,
            decltype(check_padding_v)::value,
            GatherTrids,
            EnableDramReadStaging,
            dram_read_alignment>(
            noc,
            in_reader,
            shard_cb,
            dram_read_scratch_cb,
            batch_page_base,
            c_in_offset_bytes,
            t_shard_start,
            T_shard_cur,
            h_shard_start,
            h_start,
            h_end,
            w_shard_start,
            w_col_start,
            w_count);
    };
    if (all_in_bounds) {
        do_gather(std::false_type{});
    } else {
        do_gather(std::true_type{});
    }
}

// Halo-aware gather: for H/W-boundary positions, read the cross-device NP halo buffer instead of
// zero-padding.  T-padding is still zero-filled (no temporal neighbor exchange).  Interior positions
// read the original unpadded input.  This is the neighbor-pad fusion delta; it shares the
// experimental::CB / Noc abstractions with the interior gather above.  No trid ring here: boundary
// blocks are a small fraction of the work and mix halo/zero/interior reads per position.
template <
    uint32_t C_in_block_bytes,
    uint32_t H_shard_max_W_shard_max,
    uint32_t W_shard_max,
    uint32_t T_in,
    uint32_t H_in,
    uint32_t W_in,
    uint32_t H_in_W_in,
    uint32_t in_row_size_bytes,
    typename Reader>
void gather_rows_halo(
    Noc noc,
    const Reader& in_reader,
    const Reader& halo_reader,
    const experimental::CB& shard_cb,
    uint32_t batch_page_base,
    uint32_t batch_idx,
    uint32_t c_in_offset_bytes,
    int32_t t_shard_start,
    uint32_t T_shard_cur,
    int32_t h_shard_start,
    uint32_t h_start,
    uint32_t h_end,
    int32_t w_shard_start,
    uint32_t w_col_start,
    uint32_t w_count,
    uint32_t h_halo_outer_dim_size,
    uint32_t h_halo_H,
    uint32_t h_halo_W,
    uint32_t h_halo_padding_h,
    uint32_t h_halo_padding_w,
    uint32_t h_halo_hbot_base,
    uint32_t h_halo_wleft_base,
    uint32_t h_halo_wright_base) {
    (void)h_halo_outer_dim_size;
    for (uint32_t t_local = 0; t_local < T_shard_cur; t_local++) {
        const int32_t t_in = t_shard_start + static_cast<int32_t>(t_local);
        const bool t_outside = (t_in < 0 || t_in >= static_cast<int32_t>(T_in));
        const int32_t t_clamped = t_outside ? clampIndex(t_in, 0, static_cast<int32_t>(T_in) - 1) : t_in;
        for (uint32_t h_local = h_start; h_local < h_end; h_local++) {
            const int32_t h_in = h_shard_start + static_cast<int32_t>(h_local);
            const bool h_outside = (h_in < 0 || h_in >= static_cast<int32_t>(H_in));
            const int32_t h_clamped = h_outside ? clampIndex(h_in, 0, static_cast<int32_t>(H_in) - 1) : h_in;
            uint32_t shard_offset =
                (t_local * H_shard_max_W_shard_max + h_local * W_shard_max + w_col_start) * C_in_block_bytes;
            for (uint32_t w_idx = 0; w_idx < w_count; w_idx++) {
                const int32_t w_in = w_shard_start + static_cast<int32_t>(w_col_start + w_idx);
                const bool w_outside = (w_in < 0 || w_in >= static_cast<int32_t>(W_in));
                const int32_t w_clamped = w_outside ? clampIndex(w_in, 0, static_cast<int32_t>(W_in) - 1) : w_in;

                if (t_outside) {
                    zeroPad<C_in_block_bytes>(noc, shard_cb, shard_offset);
                } else if (w_outside && h_halo_padding_w > 0) {
                    // W boundary (including corners): read from W halo buffer.
                    // W-halo stores h_total = H_dev + 2*ph positions per (t, pad_col).
                    // h_padded = padding_h + h_in maps all positions (interior + H-boundary)
                    // to the correct W-halo index.  Unsigned wrap handles negative h_in.
                    const uint32_t t_global = batch_idx * T_in + static_cast<uint32_t>(t_clamped);
                    const uint32_t h_padded = h_halo_padding_h + static_cast<uint32_t>(h_in);
                    uint32_t halo_page;
                    if (w_in < 0) {
                        const uint32_t pad_col = h_halo_padding_w + static_cast<uint32_t>(w_in);
                        halo_page =
                            h_halo_wleft_base + t_global * h_halo_padding_w * h_halo_H + pad_col * h_halo_H + h_padded;
                    } else {
                        const uint32_t pad_col = static_cast<uint32_t>(w_in) - W_in;
                        halo_page =
                            h_halo_wright_base + t_global * h_halo_padding_w * h_halo_H + pad_col * h_halo_H + h_padded;
                    }
                    read_input_row(
                        noc,
                        halo_reader,
                        halo_page,
                        c_in_offset_bytes,
                        in_row_size_bytes,
                        shard_cb,
                        shard_offset,
                        C_in_block_bytes);
                } else if (h_outside) {
                    // H boundary (not a corner since w_outside was checked first)
                    const uint32_t t_global = batch_idx * T_in + static_cast<uint32_t>(t_clamped);
                    uint32_t halo_page;
                    if (h_in < 0) {
                        const uint32_t pad_row = h_halo_padding_h + static_cast<uint32_t>(h_in);
                        halo_page = t_global * h_halo_padding_h * h_halo_W + pad_row * h_halo_W +
                                    static_cast<uint32_t>(w_clamped);
                    } else {
                        const uint32_t pad_row = static_cast<uint32_t>(h_in) - H_in;
                        halo_page = h_halo_hbot_base + t_global * h_halo_padding_h * h_halo_W + pad_row * h_halo_W +
                                    static_cast<uint32_t>(w_clamped);
                    }
                    read_input_row(
                        noc,
                        halo_reader,
                        halo_page,
                        c_in_offset_bytes,
                        in_row_size_bytes,
                        shard_cb,
                        shard_offset,
                        C_in_block_bytes);
                } else if (w_outside) {
                    zeroPad<C_in_block_bytes>(noc, shard_cb, shard_offset);
                } else {
                    // Interior: read from original unpadded input tensor
                    const uint32_t page_idx = batch_page_base + static_cast<uint32_t>(t_in) * H_in_W_in +
                                              static_cast<uint32_t>(h_in) * W_in + static_cast<uint32_t>(w_in);
                    read_input_row(
                        noc,
                        in_reader,
                        page_idx,
                        c_in_offset_bytes,
                        in_row_size_bytes,
                        shard_cb,
                        shard_offset,
                        C_in_block_bytes);
                }
                shard_offset += C_in_block_bytes;
            }
        }
    }
    noc.async_read_barrier();
}
