// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// EXPERIMENT ONLY (perf bake-off "pad_stamp"). Not part of the op.
//
// The op's writer (ttnn/ttnn/operations/tilize/kernels/tilize_writer.cpp) with
// ONE thing parameterized: how the OUTPUT-format pad stamp materializes the pad
// region of a finished tile. Every other line — placement regimes, work modes,
// levers, zones, index maps — is carried over verbatim so the arms differ only
// in the concept under test.
//
//   PS_FILL      0 = the shipped fill loop, 1 = the same volatile word stores
//                unrolled 8x (loop-overhead only; identical bytes).
//   PS_PADTILE   how a WHOLE-pad tile (valid_rows == 0 || valid_cols == 0) is
//                produced. A partly-valid tile is ALWAYS stamped in place by
//                fill_tile_pad — those are the ragged W/H tails and no variant
//                here changes them.
//                0 = stamp it in the output CB, element-by-element (SHIPPED).
//                1 = REPLICATE: stamp ONE tile into an L1 scratch page once per
//                    kernel, then copy it into the CB slot with a local
//                    L1->L1 transfer (one ~4 KB move instead of ~1024 stores).
//                2 = SOURCE-SUBSTITUTE: don't touch the CB at all — issue the
//                    outgoing page write straight FROM the scratch tile. Every
//                    byte of a whole-pad tile is the pad word, so the CB's
//                    contents for that page are irrelevant. Needs an outgoing
//                    write to exist, so it degrades to (1) on P_LOCAL_SHARD,
//                    where the CB *is* the output tensor.
//                3 = SLOT CACHE (option 3 of the assignment, kept as EVIDENCE):
//                    skip the restamp when this CB slot already held the same
//                    whole-pad pattern. Expected to be WRONG — compute repacks
//                    every byte of the slot each block — and the correctness
//                    gate is what proves it.
//
// RAW-DATAFLOW NOTE (mode 1): the L1->L1 replicate is issued as a local
// `noc_async_read` (from `get_noc_addr(scratch)` into the CB slot) rather than a
// `noc_async_write`. Both express the copy, but the writer's WRITE command
// buffer is already carrying the outgoing page writes — and, on the B8 arm, a
// transaction-id'd write that must stay in flight across the block boundary.
// Putting the copy on the READ command buffer (idle in this kernel) gives it an
// INDEPENDENT barrier: `noc_async_read_barrier()` orders the copy against the
// subsequent page write without touching the write trid state at all.

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
#include "ttnn/ttnn/operations/tilize/perf_experiments/pad_stamp/pad_stamp_fill.hpp"
#include "ttnn/ttnn/operations/tilize/kernels/tilize_noc.hpp"

#ifndef PS_FILL
#define PS_FILL 0
#endif
#ifndef PS_PADTILE
#define PS_PADTILE 0
#endif

void kernel_main() {
    constexpr uint32_t cb_output_tiles = 16;
    // Experiment-only scratch page (one output tile), appended to the program by
    // the harness. Only touched when PS_PADTILE != 0.
    constexpr uint32_t cb_pad_scratch = 2;
    constexpr uint32_t TILE_W = 32;

    constexpr uint32_t placement = get_compile_time_arg_val(0);
    constexpr uint32_t work_mode = get_compile_time_arg_val(1);
    constexpr uint32_t wt_chunk = get_compile_time_arg_val(2);
    constexpr uint32_t nt_h = get_compile_time_arg_val(3);
    constexpr uint32_t wt = get_compile_time_arg_val(4);
    constexpr uint32_t n_chunks = get_compile_time_arg_val(5);
    constexpr uint32_t out_tile_bytes = get_compile_time_arg_val(6);
    constexpr uint32_t block_write = get_compile_time_arg_val(7);
    constexpr uint32_t ablate_dm = get_compile_time_arg_val(8);
    constexpr uint32_t page_write = get_compile_time_arg_val(9);
    constexpr uint32_t write_trid = get_compile_time_arg_val(10);
    constexpr uint32_t out_fill = get_compile_time_arg_val(11);
    constexpr uint32_t out_elem_bytes = get_compile_time_arg_val(12);
    constexpr uint32_t tile_h = get_compile_time_arg_val(13);
    constexpr uint32_t h_in = get_compile_time_arg_val(14);
    constexpr uint32_t w_in_elems = get_compile_time_arg_val(15);
    constexpr uint32_t nth_per_img = get_compile_time_arg_val(16);
    constexpr uint32_t n_img_in = get_compile_time_arg_val(17);
    constexpr uint32_t write_state = get_compile_time_arg_val(18);
    constexpr uint32_t precomp_index = get_compile_time_arg_val(19);
    constexpr auto dst_args = TensorAccessorArgs<20>();

    // A whole-pad tile can only be sourced from scratch when there IS an
    // outgoing write; the zero-copy shard placement falls back to the replicate.
    constexpr uint32_t pad_mode = (PS_PADTILE == 2 && placement == 1) ? 1 : PS_PADTILE;
    constexpr bool uses_scratch = (out_fill != 0) && (pad_mode == 1 || pad_mode == 2);

    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_block = get_arg_val<uint32_t>(1);
    const uint32_t num_blocks = get_arg_val<uint32_t>(2);
    const uint32_t tile_row0 = get_arg_val<uint32_t>(3);
    const uint32_t tile_col0 = get_arg_val<uint32_t>(4);
    const uint32_t pad_word_out = get_arg_val<uint32_t>(5);
    const uint32_t block_row0 = get_arg_val<uint32_t>(6);
    const uint32_t block_wc0 = get_arg_val<uint32_t>(7);

    if (num_blocks == 0) {
        return;
    }

    tilize_kernels::BlockIndex<precomp_index != 0, nt_h> idx;
    idx.init(start_block, block_row0, block_wc0);

    auto tile_row_of = [&](uint32_t i) -> uint32_t {
        if constexpr (work_mode == 1 /* W_REGION */) {
            return tile_row0 + i / n_chunks;
        } else {
            return idx.row;
        }
    };
    auto tile_col_of = [&](uint32_t i) -> uint32_t {
        if constexpr (work_mode == 1 /* W_REGION */) {
            const uint32_t r = i / n_chunks;
            return tile_col0 + (i - r * n_chunks) * wt_chunk;
        } else {
            return idx.wc * wt_chunk;
        }
    };
    auto first_page_of = [&](uint32_t i) -> uint32_t { return tile_row_of(i) * wt + tile_col_of(i); };

    // --- the pad geometry, split out of the shipped `stamp_pad` unchanged -----
    auto valid_rows_of = [&](uint32_t i) -> uint32_t {
        const uint32_t tile_row = tile_row_of(i);
        const uint32_t img = tile_row / nth_per_img;
        const uint32_t row_in_img = (tile_row % nth_per_img) * tile_h;
        uint32_t valid_rows = 0;
        if (img < n_img_in && row_in_img < h_in) {
            valid_rows = h_in - row_in_img;
            if (valid_rows > tile_h) {
                valid_rows = tile_h;
            }
        }
        return valid_rows;
    };
    auto valid_cols_of = [&](uint32_t i, uint32_t k) -> uint32_t {
        const uint32_t col0 = (tile_col_of(i) + k) * TILE_W;
        uint32_t valid_cols = 0;
        if (col0 < w_in_elems) {
            valid_cols = w_in_elems - col0;
            if (valid_cols > TILE_W) {
                valid_cols = TILE_W;
            }
        }
        return valid_cols;
    };

    // --- the pre-stamped pad tile (modes 1 / 2) ------------------------------
    // ONE fill of ONE tile for the whole kernel — and only on a core that
    // actually meets a whole-pad tile. Measured on the `ragged` geometry
    // ([1,1,50,50] -> [1,1,64,64], which has W and H tails but NO whole-pad
    // tile): stamping it unconditionally at kernel start added ~4.3 us to a
    // 6.7 us op, a pure regression on every geometry the pattern cannot help.
    // Hence LAZY — the first whole-pad tile pays it, a geometry without one
    // never does — under a compile-time gate that deletes the code outright
    // when the target's geometry cannot contain a whole-pad tile at all.
    constexpr bool geometry_has_pad_tile =
        (h_in + tile_h <= nth_per_img * tile_h) || (w_in_elems + TILE_W <= wt * TILE_W) ||
        (n_img_in * nth_per_img < nt_h);
    constexpr bool scratch_possible = uses_scratch && geometry_has_pad_tile;
    uint32_t scratch_addr = 0;
    uint64_t scratch_noc = 0;
    bool scratch_ready = false;
    if constexpr (scratch_possible) {
        scratch_addr = get_write_ptr(cb_pad_scratch);
        scratch_noc = get_noc_addr(scratch_addr);
    }
    auto ensure_scratch = [&]() {
        if constexpr (scratch_possible) {
            if (!scratch_ready) {
                pad_stamp::fill_l1_with_val<out_elem_bytes, PS_FILL>(
                    scratch_addr, tile_h * TILE_W * out_elem_bytes, pad_word_out);
                scratch_ready = true;
            }
        }
    };
    // Mode 3 only: a per-CB-SLOT record of which of the slot's wt_chunk tile
    // positions were already stamped as a whole-pad tile. Two slots are enough
    // for every arm that can reach here (the output CB is at most a couple of
    // blocks deep), and one bit per tile position is what makes the cache
    // actually HIT — a "last address stamped" cache never does, because a block
    // walks wt_chunk distinct addresses before coming back.
    static_assert(PS_PADTILE != 3 || wt_chunk <= 32, "slot-cache probe needs wt_chunk <= 32");
    uint32_t slot_base_a = 0xFFFFFFFFu, slot_base_b = 0xFFFFFFFFu;
    uint32_t slot_mask_a = 0, slot_mask_b = 0;

    auto stamp_pad = [&](uint32_t i, uint32_t l1_addr) {
        if constexpr (out_fill) {
            const uint32_t valid_rows = valid_rows_of(i);
            bool copied = false;
            for (uint32_t k = 0; k < wt_chunk; ++k) {
                const uint32_t valid_cols = valid_cols_of(i, k);
                const uint32_t addr = l1_addr + k * out_tile_bytes;
                if (valid_rows == 0 || valid_cols == 0) {
                    constexpr bool stamp_in_place = (pad_mode == 0) || ((pad_mode == 1 || pad_mode == 2) && !scratch_possible);
                    if constexpr (stamp_in_place) {
                        // Baseline, and the compile-time-impossible case: stamp in place.
                        pad_stamp::fill_tile_pad<tile_h, TILE_W, out_elem_bytes, PS_FILL>(
                            addr, valid_rows, valid_cols, pad_word_out);
                    } else if constexpr (pad_mode == 1) {
                        ensure_scratch();
                        noc_async_read(scratch_noc, addr, out_tile_bytes);
                        copied = true;
                    } else if constexpr (pad_mode == 2) {
                        ensure_scratch();  // the outgoing write is sourced from `scratch_addr`
                    } else {  // pad_mode == 3, the UNSOUND slot cache
                        uint32_t* mask;
                        if (l1_addr == slot_base_a) {
                            mask = &slot_mask_a;
                        } else if (l1_addr == slot_base_b) {
                            mask = &slot_mask_b;
                        } else {  // evict the older entry
                            slot_base_b = slot_base_a;
                            slot_mask_b = slot_mask_a;
                            slot_base_a = l1_addr;
                            slot_mask_a = 0;
                            mask = &slot_mask_a;
                        }
                        if ((*mask & (1u << k)) == 0) {
                            pad_stamp::fill_tile_pad<tile_h, TILE_W, out_elem_bytes, PS_FILL>(
                                addr, valid_rows, valid_cols, pad_word_out);
                            *mask |= (1u << k);
                        }
                    }
                } else {
                    // Ragged W/H tail — identical in every arm.
                    pad_stamp::fill_tile_pad<tile_h, TILE_W, out_elem_bytes, PS_FILL>(
                        addr, valid_rows, valid_cols, pad_word_out);
                }
            }
            if constexpr (pad_mode == 1) {
                // The copy must have LANDED before anything reads the slot (the
                // outgoing page write, or the host on the aliased-shard path).
                // Independent of the write barrier / write trid by construction.
                if (copied) {
                    noc_async_read_barrier();
                }
            }
        }
    };

    // Mode 2: source a whole-pad page from the pre-stamped scratch tile.
    auto src_of = [&](uint32_t i, uint32_t k, uint32_t l1_addr) -> uint32_t {
        if constexpr (out_fill && pad_mode == 2 && scratch_possible) {
            if (valid_rows_of(i) == 0 || valid_cols_of(i, k) == 0) {
                return scratch_addr;
            }
        }
        return l1_addr;
    };

    if constexpr (placement == 1 /* P_LOCAL_SHARD */) {
        for (uint32_t i = 0; i < num_blocks; ++i) {
            idx.seek(i);
            {
                MaybeDeviceZoneScope("writer_wait");
                cb_wait_front(cb_output_tiles, wt_chunk);
            }
            {
                MaybeDeviceZoneScope("writer_stamp");
                stamp_pad(i, get_read_ptr(cb_output_tiles));
            }
            cb_pop_front(cb_output_tiles, wt_chunk);
            idx.advance();
        }
        return;
    }

    const auto accessor = TensorAccessor(dst_args, dst_addr);
    tilize_kernels::StatefulWrite<write_state != 0, out_tile_bytes> put;

    if constexpr (write_trid) {
        constexpr uint32_t slot_bytes = wt_chunk * out_tile_bytes;
        const uint32_t slot_base = get_read_ptr(cb_output_tiles);
        uint32_t slot = 0, trid_issue = 1, trid_wait = 1;
        bool in_flight = false;

        for (uint32_t i = 0; i < num_blocks; ++i) {
            idx.seek(i);
            const uint32_t first_page = first_page_of(i);
            {
                MaybeDeviceZoneScope("writer_wait");
                cb_wait_front(cb_output_tiles, in_flight ? 2 * wt_chunk : wt_chunk);
            }
            const uint32_t slot_addr = slot_base + slot * slot_bytes;

            {
                MaybeDeviceZoneScope("writer_stamp");
                stamp_pad(i, slot_addr);
            }
            {
                MaybeDeviceZoneScope("writer_issue");
                noc_async_write_set_trid(trid_issue);
                uint32_t l1_addr = slot_addr;
                for (uint32_t k = 0; k < wt_chunk; ++k) {
                    noc_async_write(src_of(i, k, l1_addr), accessor.get_noc_addr(first_page + k), out_tile_bytes);
                    l1_addr += out_tile_bytes;
                }
            }
            slot ^= 1;
            trid_issue ^= 3;

            if (in_flight) {
                {
                    MaybeDeviceZoneScope("writer_barrier");
                    noc_async_write_barrier_with_trid(trid_wait);
                }
                cb_pop_front(cb_output_tiles, wt_chunk);
                trid_wait ^= 3;
            }
            in_flight = true;
            idx.advance();
        }
        noc_async_write_barrier_with_trid(trid_wait);
        cb_pop_front(cb_output_tiles, wt_chunk);
        noc_async_write_set_trid(0);
        return;
    }

    for (uint32_t i = 0; i < num_blocks; ++i) {
        idx.seek(i);
        const uint32_t first_page = first_page_of(i);

        {
            MaybeDeviceZoneScope("writer_wait");
            cb_wait_front(cb_output_tiles, wt_chunk);
        }
        uint32_t l1_addr = get_read_ptr(cb_output_tiles);

        {
            MaybeDeviceZoneScope("writer_stamp");
            stamp_pad(i, l1_addr);
        }

        {
            MaybeDeviceZoneScope("writer_issue");
            for (uint32_t k = 0; k < wt_chunk; ++k) {
                if constexpr (!ablate_dm) {
                    const uint32_t src = src_of(i, k, l1_addr);
                    if constexpr (page_write) {
                        put.write(src, accessor.get_noc_addr(first_page + k));
                    } else {
                        constexpr uint32_t half = out_tile_bytes / 2;
                        noc_async_write(src, accessor.get_noc_addr(first_page + k), half);
                        noc_async_write(
                            src + half, accessor.get_noc_addr(first_page + k, half), out_tile_bytes - half);
                    }
                }
                l1_addr += out_tile_bytes;
                if constexpr (!block_write) {
                    noc_async_write_barrier();
                }
            }
        }

        {
            MaybeDeviceZoneScope("writer_barrier");
            noc_async_write_barrier();
        }
        cb_pop_front(cb_output_tiles, wt_chunk);
        idx.advance();
    }
}
