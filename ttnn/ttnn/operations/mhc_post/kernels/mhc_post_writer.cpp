// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// mhc_post writer (BRISC / NoC1) — plain-2d and stats-2d-tree regimes.
//
// store_block(row, b): the block's output window is n * block_c pages laid out [j][c]
// (compute pushes per tile; a ragged tail is padded to the nominal count by compute).
// Each valid tile is written as soon as it is pushed (cumulative wait on the window),
// one write barrier per block, then the whole nominal window is popped.
//
// fused (stats), after row r (see mhc_post_combine.hpp):
//   ship_partial(r): patch the row's ss (col 0 of cb_ss_row) into column 24 of cb_partial,
//       wait for the leader's grant, write it into my fixed gather slot, bump the leader's
//       per-set arrival semaphore.
//   leader: ship row r - LAG's combined tile to the row's root (two-level) or write it to the
//       stats tensor (single level).
//   root: write row r - 2*LAG's stats tile.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/ttnn/operations/mhc_post/kernels/mhc_post_combine.hpp"

namespace {

constexpr uint32_t cb_out_tiles = get_compile_time_arg_val(0);
constexpr uint32_t n = get_compile_time_arg_val(1);
constexpr uint32_t block_c = get_compile_time_arg_val(2);
constexpr uint32_t tensor_c_tiles = get_compile_time_arg_val(3);
constexpr uint32_t FB = 4;  // fused block (descriptor fused_ct)
constexpr bool fused = get_compile_time_arg_val(FB + 0) != 0;
constexpr uint32_t cb_ss_row = get_compile_time_arg_val(FB + 6);
constexpr uint32_t cb_partial = get_compile_time_arg_val(FB + 7);
constexpr uint32_t cb_gather = get_compile_time_arg_val(FB + 8);
constexpr uint32_t cb_root_gather = get_compile_time_arg_val(FB + 9);
constexpr uint32_t cb_lead_out = get_compile_time_arg_val(FB + 10);
constexpr uint32_t cb_stats_row = get_compile_time_arg_val(FB + 12);
constexpr uint32_t Q = get_compile_time_arg_val(FB + 13);
constexpr uint32_t COMBINE_FANIN = get_compile_time_arg_val(FB + 14);
constexpr uint32_t COMBINE_DEPTH = get_compile_time_arg_val(FB + 15);
constexpr uint32_t COMBINE_LAG = get_compile_time_arg_val(FB + 16);
constexpr uint32_t SEM_GATHER = get_compile_time_arg_val(FB + 17);
constexpr uint32_t SEM_ROOT_GATHER = get_compile_time_arg_val(FB + 18);
constexpr uint32_t SEM_LEAD_CREDIT = get_compile_time_arg_val(FB + 19);
constexpr uint32_t SEM_ROOT_CREDIT = get_compile_time_arg_val(FB + 20);
constexpr uint32_t FUSED_CT_LEN = 25;  // len(fused_ct) in the descriptor
constexpr auto out_args = TensorAccessorArgs<FB + FUSED_CT_LEN>();
constexpr auto stats_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();

constexpr uint32_t SS_COL = 24;  // n (n + 2)
constexpr uint32_t FACE_WORDS = 256;
constexpr uint32_t FACE_DIM = 16;

FORCE_INLINE uint32_t tile_word(uint32_t t, uint32_t c) {
    return ((t / FACE_DIM) * 2 + (c / FACE_DIM)) * FACE_WORDS + (t % FACE_DIM) * FACE_DIM + (c % FACE_DIM);
}

FORCE_INLINE volatile tt_l1_ptr uint32_t* sem_ptr(uint32_t id) {
    return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(id));
}

}  // namespace

void kernel_main() {
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t row_start = get_arg_val<uint32_t>(1);
    const uint32_t num_rows = get_arg_val<uint32_t>(2);
    const uint32_t c_start = get_arg_val<uint32_t>(3);
    const uint32_t num_c = get_arg_val<uint32_t>(4);
    // arg 5: g_r (unused here)
    const uint32_t g_c = get_arg_val<uint32_t>(6);
    const uint32_t stats_addr = get_arg_val<uint32_t>(7);
    constexpr uint32_t RT_LEADERS = 8;  // (x, y) of every group leader of my row group

    const uint32_t tile_bytes = get_tile_size(cb_out_tiles);
    const auto out_acc = TensorAccessor(out_args, out_addr, tile_bytes);
    const uint32_t num_blocks = (num_c + block_c - 1) / block_c;
    constexpr uint32_t window = n * block_c;

    const auto geo = mhc_post_combine::Geometry::of(g_c, Q, COMBINE_FANIN);
    constexpr uint32_t STAT_TILE = 4096;  // fp32 partial / stats tile
    const auto stats_acc = TensorAccessor(stats_args, stats_addr, STAT_TILE);
    // This RISC never pushes the gather CBs, so its write pointer is their L1 base (identical on
    // every core of the program).
    uint32_t gather_base = 0, root_gather_base = 0;
    if constexpr (fused) {
        gather_base = get_write_ptr(cb_gather);
        if (geo.two_level()) {
            root_gather_base = get_write_ptr(cb_root_gather);
        }
    }

    auto send_tile = [&](uint32_t src_l1, uint32_t g, uint32_t dst_l1, uint32_t sem_id) {
        const uint32_t px = get_arg_val<uint32_t>(RT_LEADERS + 2 * g);
        const uint32_t py = get_arg_val<uint32_t>(RT_LEADERS + 2 * g + 1);
        noc_async_write(src_l1, get_noc_addr(px, py, dst_l1), STAT_TILE);
        noc_async_write_barrier();
        noc_semaphore_inc(get_noc_addr(px, py, get_semaphore(sem_id)), 1);
    };
    auto write_stats = [&](uint32_t cb, uint32_t row) {
        cb_wait_front(cb, 1);
        noc_async_write_page(row, stats_acc, get_read_ptr(cb));
        noc_async_write_barrier();
        cb_pop_front(cb, 1);
    };

    auto post_row = [&](uint32_t r) {
        if (r < num_rows) {
            // ship_partial(r)
            cb_wait_front(cb_ss_row, 1);
            cb_wait_front(cb_partial, 1);
            volatile tt_l1_ptr uint32_t* part =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_partial));
            volatile tt_l1_ptr uint32_t* ss = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_ss_row));
            for (uint32_t t = 0; t < 32; ++t) {
                part[tile_word(t, SS_COL)] = ss[tile_word(t, 0)];
            }
            noc_semaphore_wait_min(sem_ptr(SEM_LEAD_CREDIT), r + 1);
            const uint32_t set = r % COMBINE_DEPTH;
            send_tile(
                get_read_ptr(cb_partial),
                geo.group,
                gather_base + (set * geo.slots + geo.member) * STAT_TILE,
                SEM_GATHER + set);
            cb_pop_front(cb_partial, 1);
            cb_pop_front(cb_ss_row, 1);
        }
        if (geo.leader && r >= COMBINE_LAG && r - COMBINE_LAG < num_rows) {
            const uint32_t rl = r - COMBINE_LAG;
            if (geo.two_level()) {
                const uint32_t gr = geo.root_group(rl);
                const uint32_t k = geo.root_ordinal(rl);
                const uint32_t set = k % COMBINE_DEPTH;
                cb_wait_front(cb_lead_out, 1);
                noc_semaphore_wait_min(sem_ptr(SEM_ROOT_CREDIT + gr), k + 1);
                send_tile(
                    get_read_ptr(cb_lead_out),
                    gr,
                    root_gather_base + (set * geo.groups + geo.group) * STAT_TILE,
                    SEM_ROOT_GATHER + set);
                cb_pop_front(cb_lead_out, 1);
            } else {
                write_stats(cb_stats_row, row_start + rl);
            }
        }
        if (r >= 2 * COMBINE_LAG && r - 2 * COMBINE_LAG < num_rows && geo.roots(r - 2 * COMBINE_LAG)) {
            write_stats(cb_stats_row, row_start + r - 2 * COMBINE_LAG);
        }
    };

    for (uint32_t r = 0; r < num_rows; ++r) {
        const uint32_t row = row_start + r;
        for (uint32_t b = 0; b < num_blocks; ++b) {
            const uint32_t c0 = c_start + b * block_c;
            const uint32_t extent = (num_c - b * block_c) < block_c ? (num_c - b * block_c) : block_c;
            uint32_t k = 0;  // position in the compute-push order (j-major, c-minor)
            for (uint32_t j = 0; j < n; ++j) {
                const uint32_t page0 = (row * n + j) * tensor_c_tiles + c0;
                for (uint32_t c = 0; c < extent; ++c, ++k) {
                    cb_wait_front(cb_out_tiles, k + 1);
                    noc_async_write_tile(page0 + c, out_acc, get_read_ptr(cb_out_tiles) + k * tile_bytes);
                }
            }
            cb_wait_front(cb_out_tiles, window);  // padding pages of a ragged tail
            noc_async_write_barrier();
            cb_pop_front(cb_out_tiles, window);
        }
        if constexpr (fused) {
            post_row(r);
        }
    }
    if constexpr (fused) {
        for (uint32_t r = num_rows; r < num_rows + 2 * COMBINE_LAG; ++r) {
            post_row(r);
        }
        noc_async_atomic_barrier();
    }
}
