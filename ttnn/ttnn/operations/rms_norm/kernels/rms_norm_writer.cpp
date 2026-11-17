// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// rms_norm writer (BRISC). TILE: writes tile pages (Rb x Sb window per barrier).
// RM: writes only valid sticks / valid bytes of each stick segment (32 sticks per barrier);
// H/W pad is never written.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/semaphore.h"

void kernel_main() {
    const uint32_t o_addr = get_arg_val<uint32_t>(0);
    const uint32_t first_tile_row = get_arg_val<uint32_t>(1);
    const uint32_t rows_core = get_arg_val<uint32_t>(2);
    const uint32_t slice_tile0 = get_arg_val<uint32_t>(3);
    const uint32_t my_j = get_arg_val<uint32_t>(4);  // this core's slice index within its row group
    constexpr uint32_t peers_base = 5;               // N_W (x, y) physical coords, in slice order

    constexpr bool IS_RM = get_compile_time_arg_val(0) != 0;
    constexpr uint32_t ES = get_compile_time_arg_val(1);
    constexpr uint32_t Wt = get_compile_time_arg_val(2);
    constexpr uint32_t Sb = get_compile_time_arg_val(3);
    constexpr uint32_t Rb = get_compile_time_arg_val(4);
    constexpr uint32_t W = get_compile_time_arg_val(5);
    constexpr uint32_t H = get_compile_time_arg_val(6);
    constexpr uint32_t Ht_img = get_compile_time_arg_val(7);
    constexpr uint32_t cb_out_tiles = get_compile_time_arg_val(8);
    constexpr uint32_t cb_out_rm = get_compile_time_arg_val(9);
    constexpr uint32_t Wt_full = get_compile_time_arg_val(10);  // Wt is the per-core slice width
    constexpr uint32_t N_W = get_compile_time_arg_val(11);      // cores sharing a row group (W-split); 1 = off
    constexpr uint32_t cb_psend = get_compile_time_arg_val(12);
    constexpr uint32_t cb_part = get_compile_time_arg_val(13);
    constexpr uint32_t sem_id0 = get_compile_time_arg_val(14);
    constexpr uint32_t sem_id1 = get_compile_time_arg_val(15);
    constexpr bool OUT_SHARDED = get_compile_time_arg_val(16) != 0;  // compute packs straight into the out shard
    constexpr bool ROW_FLAT = get_compile_time_arg_val(17) != 0;     // RM shard: tile-row = 32 consecutive sticks
    constexpr bool OUT_SHARD_RM = get_compile_time_arg_val(18) != 0;  // RM out sticks live in the local shard
    constexpr uint32_t OUT_STRIDE = get_compile_time_arg_val(19);     // shard stick pitch in bytes
    constexpr bool W_PAD = get_compile_time_arg_val(20) != 0;         // slices cover more W tiles/elements than W
    constexpr auto o_args = TensorAccessorArgs<21>();
    constexpr uint32_t tail_base = peers_base + 2 * (N_W > 1 ? N_W : 0);
    const uint32_t first_stick = get_arg_val<uint32_t>(tail_base);
    const uint32_t core_sticks = get_arg_val<uint32_t>(tail_base + 1);
    const uint32_t slice_valid_elems = get_arg_val<uint32_t>(tail_base + 2);  // valid W elements in the slice
    const uint32_t slice_elem0 = get_arg_val<uint32_t>(tail_base + 3);        // global W element offset

    constexpr uint32_t TILE_DIM = 32;
    constexpr uint32_t num_seg = Wt / Sb;
    constexpr uint32_t SEG_BYTES = Sb * TILE_DIM * ES;

    DataflowBuffer dfb_tiles(cb_out_tiles);
    DataflowBuffer dfb_rm(cb_out_rm);

    const uint32_t tile_bytes = dfb_tiles.get_tile_size();
    const auto acc = TensorAccessor(o_args, o_addr, IS_RM ? W * ES : tile_bytes);
    const uint32_t slice_off = slice_elem0 * ES;  // interleaved RM: byte offset of the slice in a stick
    const uint32_t slice_bytes = slice_valid_elems * ES;

    // ---- W-split partial exchange (all-gather inside a row group) ----
    // Row block bi: wait the local partial tiles (cb_psend), unicast each to every peer's cb_part at
    // slot [row][my_j] of that block's parity half, publish via the parity semaphore, wait for all N_W
    // partials, then push the block (Rb * N_W tiles) to compute. Two parity halves + two cumulative
    // semaphores make a faster peer's block bi+1 partials unable to clobber / be confused with block bi.
    DataflowBuffer dfb_psend(cb_psend);
    DataflowBuffer dfb_part(cb_part);
    Noc noc;
    Semaphore<> sem0(sem_id0);
    Semaphore<> sem1(sem_id1);
    const uint32_t part_base = N_W > 1 ? dfb_part.get_write_ptr() : 0;  // identical on every core of the group
    auto exchange_partials = [&](uint32_t bi, uint32_t rb) {
        const uint32_t ptile = dfb_psend.get_tile_size();
        const uint32_t parity = bi & 1;
        const uint32_t slot_base = part_base + parity * (Rb * N_W * ptile);
        dfb_psend.wait_front(rb);
        const uint32_t src = dfb_psend.get_read_ptr();
        for (uint32_t p = 0; p < N_W; ++p) {
            const uint32_t px = get_arg_val<uint32_t>(peers_base + 2 * p);
            const uint32_t py = get_arg_val<uint32_t>(peers_base + 2 * p + 1);
            for (uint32_t rr = 0; rr < rb; ++rr) {
                noc_async_write(src + rr * ptile, get_noc_addr(px, py, slot_base + (rr * N_W + my_j) * ptile), ptile);
            }
        }
        noc_async_write_barrier();
        for (uint32_t p = 0; p < N_W; ++p) {
            const uint32_t px = get_arg_val<uint32_t>(peers_base + 2 * p);
            const uint32_t py = get_arg_val<uint32_t>(peers_base + 2 * p + 1);
            if (parity == 0) {
                sem0.up(noc, px, py, 1);
            } else {
                sem1.up(noc, px, py, 1);
            }
        }
        const uint32_t target = ((bi >> 1) + 1) * N_W;
        if (parity == 0) {
            sem0.wait_min(target);
        } else {
            sem1.wait_min(target);
        }
        dfb_part.reserve_back(Rb * N_W);
        dfb_part.push_back(Rb * N_W);
        dfb_psend.pop_front(rb);
    };

    for (uint32_t b = 0; b < rows_core; b += Rb) {
        const uint32_t rb = (rows_core - b) < Rb ? (rows_core - b) : Rb;
        const uint32_t tr0 = first_tile_row + b;
        if constexpr (N_W > 1) {
            exchange_partials(b / Rb, rb);
        }
        for (uint32_t seg = 0; seg < num_seg; ++seg) {
            if constexpr (IS_RM) {
                for (uint32_t rr = 0; rr < rb; ++rr) {
                    const uint32_t tr = tr0 + rr;
                    uint32_t valid_rows, stick0;
                    if constexpr (ROW_FLAT) {
                        const uint32_t left = core_sticks - tr * TILE_DIM;
                        valid_rows = left < TILE_DIM ? left : TILE_DIM;
                        stick0 = first_stick + tr * TILE_DIM;
                    } else {
                        const uint32_t img = tr / Ht_img;
                        const uint32_t r = tr % Ht_img;
                        valid_rows = (H - r * TILE_DIM) < TILE_DIM ? (H - r * TILE_DIM) : TILE_DIM;
                        stick0 = img * H + r * TILE_DIM;
                    }
                    dfb_rm.wait_front(Sb);
                    const uint32_t base = dfb_rm.get_read_ptr();
                    const uint32_t seg_off = seg * SEG_BYTES;  // byte offset inside the slice
                    const uint32_t avail = slice_bytes > seg_off ? slice_bytes - seg_off : 0;
                    const uint32_t bytes = avail < SEG_BYTES ? avail : SEG_BYTES;
                    for (uint32_t k = 0; bytes > 0 && k < valid_rows; ++k) {
                        if constexpr (OUT_SHARD_RM) {
                            noc_async_write(
                                base + k * SEG_BYTES,
                                get_noc_addr(o_addr + (stick0 - first_stick + k) * OUT_STRIDE + seg_off),
                                bytes);
                        } else {
                            noc_async_write(
                                base + k * SEG_BYTES, acc.get_noc_addr(stick0 + k, slice_off + seg_off), bytes);
                        }
                    }
                    noc_async_write_barrier();
                    dfb_rm.pop_front(Sb);
                }
            } else if constexpr (!OUT_SHARDED) {
                dfb_tiles.wait_front(rb * Sb);
                const uint32_t base = dfb_tiles.get_read_ptr();
                for (uint32_t rr = 0; rr < rb; ++rr) {
                    for (uint32_t c = 0; c < Sb; ++c) {
                        const uint32_t g = slice_tile0 + seg * Sb + c;  // global tile column
                        if constexpr (W_PAD) {
                            if (g >= Wt_full) {
                                continue;  // pad tile of a ragged last slice: nothing to store
                            }
                        }
                        noc_async_write_tile((tr0 + rr) * Wt_full + g, acc, base + (rr * Sb + c) * tile_bytes);
                    }
                }
                noc_async_write_barrier();
                dfb_tiles.pop_front(rb * Sb);
            }
        }
    }
}
