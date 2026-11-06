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
// collapse, additionally:
//   store_block: the block's y tiles (block_c window, pushed per tile) after its X' window.
//   hc_row, on the core that writes a row's stats: permute the stats row into the hc lane layout
//       (cb_hc_lanes, see mhc_post_hc_sfpu.hpp), let compute run the hc program, inverse-permute
//       the result lanes into cb_hc_row and write the hc page.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc.h"
#include "ttnn/ttnn/operations/mhc_post/kernels/mhc_post_combine.hpp"
#include "ttnn/ttnn/operations/mhc_post/kernels/mhc_post_hc_sfpu.hpp"

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
constexpr bool collapse = get_compile_time_arg_val(FB + 25) != 0;
constexpr uint32_t cb_y_tiles = get_compile_time_arg_val(FB + 26);
constexpr uint32_t cb_hc_lanes = get_compile_time_arg_val(FB + 27);
constexpr uint32_t cb_hc_out = get_compile_time_arg_val(FB + 28);
constexpr uint32_t cb_hc_row = get_compile_time_arg_val(FB + 29);
constexpr uint32_t FUSED_CT_LEN = 30;  // len(fused_ct) in the descriptor
constexpr auto out_args = TensorAccessorArgs<FB + FUSED_CT_LEN>();
constexpr auto stats_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();
constexpr auto y_args = TensorAccessorArgs<stats_args.next_compile_time_args_offset()>();
constexpr auto hc_args = TensorAccessorArgs<y_args.next_compile_time_args_offset()>();

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
    const uint32_t y_addr = get_arg_val<uint32_t>(8);
    const uint32_t hc_addr = get_arg_val<uint32_t>(9);
    constexpr uint32_t RT_LEADERS = 10;  // (x, y) of every group leader of my row group

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
    // collapse: y / hc tensors and the hc landing tile (columns NG..31 of the hc page stay 0).
    const uint32_t y_tile_bytes = collapse ? get_tile_size(cb_y_tiles) : 0;
    const auto y_acc = TensorAccessor(y_args, y_addr, y_tile_bytes);
    const auto hc_acc = TensorAccessor(hc_args, hc_addr, STAT_TILE);
    if constexpr (collapse) {
        Noc noc;
        CircularBuffer hc_row_cb(cb_hc_row);
        noc.async_write_zeros(hc_row_cb, STAT_TILE);
        noc.write_zeros_l1_barrier();
    }

    // hc_row: stats tile (row t, col k) -> lane word (k, t) of lane tile k / 16 (+ its duplicate);
    // compute runs the hc program; result lanes -> hc page (row t, col k).
    auto hc_row = [&](uint32_t stats_l1, uint32_t row) {
        using namespace mhc_post_hc;
        cb_reserve_back(cb_hc_lanes, LANE_TILES);
        volatile tt_l1_ptr uint32_t* lanes = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_hc_lanes));
        volatile tt_l1_ptr uint32_t* st = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stats_l1);
        for (uint32_t k = 0; k <= SS_Q; ++k) {
            volatile tt_l1_ptr uint32_t* lt = lanes + (k / 16) * 1024;
            for (uint32_t t = 0; t < 32; ++t) {
                const uint32_t v = st[tile_word(t, k)];
                lt[lane_word(k, t)] = v;
                lt[lane_word(k, t) + 32] = v;
            }
        }
        cb_push_back(cb_hc_lanes, LANE_TILES);
        cb_wait_front(cb_hc_out, LANE_TILES);
        volatile tt_l1_ptr uint32_t* res = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_hc_out));
        volatile tt_l1_ptr uint32_t* hrow = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_hc_row));
        for (uint32_t k = 0; k < NG; ++k) {
            volatile tt_l1_ptr uint32_t* lt = res + (k / 16) * 1024;
            for (uint32_t t = 0; t < 32; ++t) {
                hrow[tile_word(t, k)] = lt[lane_word(k, t)];
            }
        }
        cb_pop_front(cb_hc_out, LANE_TILES);
        noc_async_write_page(row, hc_acc, get_write_ptr(cb_hc_row));
    };
    auto write_stats = [&](uint32_t cb, uint32_t row) {
        cb_wait_front(cb, 1);
        noc_async_write_page(row, stats_acc, get_read_ptr(cb));
        if constexpr (collapse) {
            hc_row(get_read_ptr(cb), row);
        }
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
            if constexpr (collapse) {
                const uint32_t y_page0 = row * tensor_c_tiles + c0;
                for (uint32_t c = 0; c < extent; ++c) {
                    cb_wait_front(cb_y_tiles, c + 1);
                    noc_async_write_tile(y_page0 + c, y_acc, get_read_ptr(cb_y_tiles) + c * y_tile_bytes);
                }
                cb_wait_front(cb_y_tiles, block_c);  // padding pages of a ragged tail
            }
            noc_async_write_barrier();
            cb_pop_front(cb_out_tiles, window);
            if constexpr (collapse) {
                cb_pop_front(cb_y_tiles, block_c);
            }
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
