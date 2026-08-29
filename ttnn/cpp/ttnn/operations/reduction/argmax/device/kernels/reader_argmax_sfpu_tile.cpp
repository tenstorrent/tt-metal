// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// =============================================================================
// reader_argmax_sfpu_tile.cpp — dataflow side of the SFPU TILE-layout argmax.
//
// Per 32-row tile-row pass, every core:
//   1. Streams its `w_count` input tiles (slice [w_start, w_start + w_count)
//      of the tile-row) into the double-buffered input CB in chunks. The CB
//      is treated as a ring of pages addressed by GLOBAL page index
//      (slot = t % num_pages) on both sides, so chunk batches may wrap
//      mid-batch without any linear-placement assumption (same scheme as
//      reader_argmax_rvv_tile.cpp).
//   2. Collects phase 1's two candidate tiles (max values -> bf16 CB,
//      winning LOCAL tile indices -> UInt32 CB) and finishes its slice with
//      PHASE 2: for each valid row r, scan the 32 per-column candidates in
//      ascending column order with the lexicographic rule
//          take c  iff  val_c > best_val                 (IEEE, on bf16 bits)
//                   or (val_c == best_val && idx_c < best_idx)
//      where idx_c = (w_start + win_tile[r][c]) * 32 + c. The IEEE compare
//      is emulated bit-exactly on bf16 patterns (NaN never compares
//      greater/equal; +0 == -0), matching gt_binary_tile's fp32 semantics so
//      the whole op implements ONE documented order (phase 1 already mapped
//      NaN -> same-signed inf and flushed denormals, so no NaN/denormal bit
//      pattern can appear among the candidates anyway).
//
// MULTICORE (num_cores > 1): each core deposits its per-row (index, value)
// candidates — 32 rows x 8 bytes = 256 B — into its slot of the exchange
// buffer on the gather core (core_id 0) and bumps the done semaphore; the
// gather core merges the num_cores candidates per row with the same
// lexicographic rule (a per-row scalar merge — NOT a cross-core tile
// reduce) and stages the final results into output pages. Slot reuse across
// passes is flow-controlled by a cumulative credit semaphore: workers may
// send pass p only once the gather core has consumed pass p-1. Both
// semaphores count cumulatively (wait_min, no mid-run resets) and are
// restored to 0 at kernel end so trace replay — which does not re-run the
// dispatcher's semaphore init — starts clean.
//
// The exchange buffer is a CB allocated identically on every core, so a
// worker's local cb_xchg write pointer equals the gather core's address.
//
// Compile-time args: see the factory (argmax_sfpu_tile_program_factory.cpp).
// Runtime args:
//   [0] src base address, [1] dst base address, [2, optional] maxval base
//   address; then core_id, w_start, w_count; then — gather core only —
//   num_cores (x, y) physical coord pairs of all cores (slot order), used to
//   return per-pass credits.
// =============================================================================

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "internal/tt-1xx/risc_common.h"  // invalidate_l1_cache()

namespace {

// IEEE order on bf16 bit patterns — bit-exact emulation of an fp32 compare
// after the exact bf16->fp32 widening the unpacker performs.
inline bool ieee_isnan(uint16_t x) { return static_cast<uint16_t>(x & 0x7FFFu) > 0x7F80u; }

inline bool ieee_gt(uint16_t a, uint16_t b) {
    if (ieee_isnan(a) || ieee_isnan(b)) {
        return false;
    }
    const bool az = static_cast<uint16_t>(a & 0x7FFFu) == 0;
    const bool bz = static_cast<uint16_t>(b & 0x7FFFu) == 0;
    if (az && bz) {
        return false;  // +0 == -0
    }
    if ((a ^ b) & 0x8000u) {
        return (a & 0x8000u) == 0;  // signs differ (not both zero): positive wins
    }
    return (a & 0x8000u) ? (a < b) : (a > b);  // sign-magnitude within a sign
}

inline bool ieee_eq(uint16_t a, uint16_t b) {
    if (ieee_isnan(a) || ieee_isnan(b)) {
        return false;
    }
    const bool az = static_cast<uint16_t>(a & 0x7FFFu) == 0;
    const bool bz = static_cast<uint16_t>(b & 0x7FFFu) == 0;
    if (az && bz) {
        return true;
    }
    return a == b;
}

// (row, col) -> element offset inside a 32x32 tile stored as faces 0..3
// (16x16 row-major each): face = (row<16 ? 0 : 2) + (col<16 ? 0 : 1).
inline uint32_t face_elem(uint32_t r, uint32_t c) {
    const uint32_t face = ((r >> 4) << 1) + (c >> 4);
    return face * 256u + (r & 15u) * 16u + (c & 15u);
}

}  // namespace

void kernel_main() {
    constexpr uint32_t cb_in = get_compile_time_arg_val(0);
    constexpr uint32_t cb_res_val = get_compile_time_arg_val(1);
    constexpr uint32_t cb_res_idx = get_compile_time_arg_val(2);
    constexpr uint32_t cb_xchg = get_compile_time_arg_val(3);
    constexpr uint32_t cb_stage_idx = get_compile_time_arg_val(4);
    constexpr uint32_t cb_stage_val = get_compile_time_arg_val(5);
    constexpr uint32_t src_page_size = get_compile_time_arg_val(6);
    constexpr uint32_t chunk_pages = get_compile_time_arg_val(7);
    constexpr uint32_t in_cb_pages = get_compile_time_arg_val(8);
    constexpr uint32_t h_tiles = get_compile_time_arg_val(9);
    constexpr uint32_t logical_height = get_compile_time_arg_val(10);
    constexpr uint32_t outer_dim_units = get_compile_time_arg_val(11);
    constexpr uint32_t out_page_elems = get_compile_time_arg_val(12);
    constexpr uint32_t dst_page_size = get_compile_time_arg_val(13);
    constexpr uint32_t val_page_size = get_compile_time_arg_val(14);
    constexpr bool has_maxval = (bool)get_compile_time_arg_val(15);
    constexpr uint32_t num_cores = get_compile_time_arg_val(16);
    constexpr uint32_t gather_noc_x = get_compile_time_arg_val(17);
    constexpr uint32_t gather_noc_y = get_compile_time_arg_val(18);
    constexpr uint32_t done_sem_id = get_compile_time_arg_val(19);
    constexpr uint32_t start_sem_id = get_compile_time_arg_val(20);
    constexpr uint32_t w_tiles_total = get_compile_time_arg_val(21);
    constexpr uint32_t num_c_time_args = 22;

    // Exchange slot layout: u32 idx[32] then u32 val[32] (bf16 bits in the
    // low half-word) — 256 B per core.
    constexpr uint32_t xchg_slot_words = 64;
    constexpr uint32_t xchg_slot_bytes = xchg_slot_words * sizeof(uint32_t);

    const uint32_t src_base_addr = get_arg_val<uint32_t>(0);
    const uint32_t dst_base_addr = get_arg_val<uint32_t>(1);
    const uint32_t val_base_addr = has_maxval ? get_arg_val<uint32_t>(2) : 0;
    uint32_t argi = has_maxval ? 3 : 2;
    const uint32_t core_id = get_arg_val<uint32_t>(argi++);
    const uint32_t w_start = get_arg_val<uint32_t>(argi++);
    const uint32_t w_count = get_arg_val<uint32_t>(argi++);
    const uint32_t coord_args_base = argi;  // gather core only: (x, y) pairs

    const bool is_gather = (core_id == 0);

    // Contract with the factory: exactly THREE TensorAccessorArgs blocks are
    // appended (src, dst, val) — when no maxval tensor is supplied the dst
    // block is duplicated as a placeholder so the constexpr offset chain
    // lines up; has_maxval guards every use of the third accessor.
    constexpr auto s_src_args = TensorAccessorArgs<num_c_time_args>();
    constexpr auto s_dst_args = TensorAccessorArgs<s_src_args.next_compile_time_args_offset()>();
    const auto s_src = TensorAccessor(s_src_args, src_base_addr, src_page_size);
    const auto s_dst = TensorAccessor(s_dst_args, dst_base_addr, dst_page_size);
    constexpr auto s_val_args = TensorAccessorArgs<s_dst_args.next_compile_time_args_offset()>();
    const auto s_val = TensorAccessor(s_val_args, val_base_addr, val_page_size);

    Noc noc;
    CircularBuffer in_cb(cb_in);
    CircularBuffer res_val_cb(cb_res_val);
    CircularBuffer res_idx_cb(cb_res_idx);
    CircularBuffer xchg_cb(cb_xchg);
    CircularBuffer stage_idx_cb(cb_stage_idx);
    CircularBuffer stage_val_cb(cb_stage_val);

    // Input CB ring base (write pointer sits at base before any push). The ring
    // is addressed by GLOBAL page index, so the base is captured ONCE here: the
    // CB's own write pointer advances on every push_back and cannot be used as
    // the NoC destination, hence a CoreLocalMem over the fixed base + offset.
    const CoreLocalMem<uint32_t> in_ring(in_cb.get_write_ptr());

    // Exchange buffer: allocated identically on every core, so the local
    // address doubles as the gather core's address for remote deposits.
    const uint32_t xchg_base = xchg_cb.get_write_ptr();
    const uint32_t my_slot_addr = xchg_base + core_id * xchg_slot_bytes;
    const CoreLocalMem<volatile uint32_t> my_slot(my_slot_addr);

    // Output staging buffers (gather core only; plain L1 scratch, no FIFO
    // semantics).
    const CoreLocalMem<uint32_t> stage_idx(stage_idx_cb.get_write_ptr());
    const CoreLocalMem<uint16_t> stage_val(stage_val_cb.get_write_ptr());

    // Semaphore handles resolve to a local L1 offset only — constructing them
    // has no side effect, so they are unconditional even though every use is
    // guarded by `if constexpr (num_cores > 1)` (the factory only allocates
    // the semaphores in the multicore case).
    Semaphore<> done_sem(done_sem_id);
    Semaphore<> start_sem(start_sem_id);

    constexpr uint32_t num_passes = outer_dim_units * h_tiles;

    uint32_t t_global = 0;   // global input page counter (matches compute side)
    uint32_t collected = 0;  // elements accumulated toward the current output page
    uint32_t out_page_id = 0;
    uint32_t pass = 0;

    for (uint32_t outer = 0; outer < outer_dim_units; outer++) {
        for (uint32_t i = 0; i < h_tiles; i++) {
            const uint32_t row_base = i * 32u;
            const uint32_t units = (logical_height - row_base < 32u) ? (logical_height - row_base) : 32u;
            const uint32_t slice_first = (outer * h_tiles + i) * w_tiles_total + w_start;

            // ---- stream this core's slice of the tile-row ------------------
            uint32_t done = 0;
            while (done < w_count) {
                const uint32_t chunk = (w_count - done < chunk_pages) ? (w_count - done) : chunk_pages;
                in_cb.reserve_back(chunk);
                for (uint32_t k = 0; k < chunk; k++) {
                    const uint32_t slot = (t_global + k) % in_cb_pages;
                    noc.async_read(
                        s_src,
                        in_ring,
                        src_page_size,
                        {.page_id = slice_first + done + k},
                        {.offset_bytes = slot * src_page_size});
                    if ((k & 31u) == 31u) {
                        // Bound the outstanding NOC read count within a chunk.
                        noc.async_read_barrier();
                    }
                }
                noc.async_read_barrier();
                in_cb.push_back(chunk);
                t_global += chunk;
                done += chunk;
            }

            // ---- phase 2: 32 lexicographic compares per valid row ----------
            res_val_cb.wait_front(1);
            res_idx_cb.wait_front(1);
            invalidate_l1_cache();
            const CoreLocalMem<volatile uint16_t> vals(res_val_cb.get_read_ptr());
            const CoreLocalMem<volatile uint32_t> tidx(res_idx_cb.get_read_ptr());

            for (uint32_t r = 0; r < units; r++) {
                uint16_t best_v = vals[face_elem(r, 0)];
                uint32_t best_i = (w_start + tidx[face_elem(r, 0)]) * 32u;
                for (uint32_t c = 1; c < 32u; ++c) {
                    const uint32_t e = face_elem(r, c);
                    const uint16_t vc = vals[e];
                    const uint32_t ic = (w_start + tidx[e]) * 32u + c;
                    if (ieee_gt(vc, best_v) || (ieee_eq(vc, best_v) && ic < best_i)) {
                        best_v = vc;
                        best_i = ic;
                    }
                }
                my_slot[r] = best_i;
                my_slot[32u + r] = best_v;
            }
            res_val_cb.pop_front(1);
            res_idx_cb.pop_front(1);

            if constexpr (num_cores > 1) {
                if (!is_gather) {
                    // Slot-reuse credit: the gather core has consumed pass
                    // p-1 once start_sem >= p (cumulative).
                    if (pass > 0) {
                        start_sem.wait_min(pass);
                    }
                    noc.async_write(
                        my_slot,
                        UnicastEndpoint{},
                        xchg_slot_bytes,
                        {},
                        {.noc_x = gather_noc_x, .noc_y = gather_noc_y, .addr = my_slot_addr});
                    noc.async_write_barrier();
                    done_sem.up(noc, gather_noc_x, gather_noc_y, 1);
                    noc.async_atomic_barrier();
                }
            }

            if (is_gather) {
                if constexpr (num_cores > 1) {
                    done_sem.wait_min((pass + 1) * (num_cores - 1));
                    invalidate_l1_cache();
                }
                // ---- merge the per-core candidates, stage output rows ------
                for (uint32_t r = 0; r < units; r++) {
                    const CoreLocalMem<volatile uint32_t> slot0(xchg_base);
                    uint32_t best_i = slot0[r];
                    uint16_t best_v = (uint16_t)slot0[32u + r];
                    for (uint32_t j = 1; j < num_cores; j++) {
                        const CoreLocalMem<volatile uint32_t> slot(xchg_base + j * xchg_slot_bytes);
                        const uint32_t ic = slot[r];
                        const uint16_t vc = (uint16_t)slot[32u + r];
                        if (ieee_gt(vc, best_v) || (ieee_eq(vc, best_v) && ic < best_i)) {
                            best_v = vc;
                            best_i = ic;
                        }
                    }
                    stage_idx[collected] = best_i;
                    if constexpr (has_maxval) {
                        stage_val[collected] = best_v;
                    }
                    collected++;
                    if (collected == out_page_elems) {
                        noc.async_write(stage_idx, s_dst, dst_page_size, {}, {.page_id = out_page_id});
                        if constexpr (has_maxval) {
                            noc.async_write(stage_val, s_val, val_page_size, {}, {.page_id = out_page_id});
                        }
                        noc.async_write_barrier();
                        collected = 0;
                        out_page_id++;
                    }
                }
                if constexpr (num_cores > 1) {
                    // Return slot-reuse credits — only if another pass will
                    // need them, so no increment can race the workers' final
                    // start_sem reset below.
                    if (pass + 1 < num_passes) {
                        for (uint32_t j = 1; j < num_cores; j++) {
                            const uint32_t wx = get_arg_val<uint32_t>(coord_args_base + 2 * j);
                            const uint32_t wy = get_arg_val<uint32_t>(coord_args_base + 2 * j + 1);
                            start_sem.up(noc, wx, wy, 1);
                        }
                        noc.async_atomic_barrier();
                    }
                }
            }
            pass++;
        }
    }

    // Restore semaphores to 0 for trace replay (the dispatcher's semaphore
    // init does not re-run on replay). Race-free: the gather core's final
    // done_sem wait has observed every worker increment, and workers receive
    // no credit after their final wait (none is sent for the last pass).
    if constexpr (num_cores > 1) {
        if (is_gather) {
            done_sem.set(0);
        } else {
            start_sem.set(0);
        }
    }
}
