// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// mhc_post reader (NCRISC / NoC0) — plain-2d and stats-2d-tree regimes.
//
// Per core: rows [row_start, row_start + num_rows) × c-tiles [c_start, c_start + num_c).
//   load_coefficients(row): read the post / comb source tiles (one page each) into the
//       reader-private scratch cb_coef_src, then expand every coefficient q into column 0
//       of its own tile in cb_coef_cols (order: post_0..n-1, comb_{i,j} at n + i*n + j).
//       The first F/X block of the row is issued before blocking on cb_coef_cols space
//       (stall shadow: the DRAM latency hides under the previous row's last blocks).
//   load_block(row, b): block_c F tiles + n*block_c X tiles (layout [i][c]), reads issued
//       back-to-back, one barrier per block, nominal block_c counts pushed (ragged tail
//       reads only its valid tiles).
//   fused (stats): load the resident W slice once (block-major [b][j][c], compact ragged tail);
//       as a combine consumer (leader / root) GRANT each row's gather slot set at the row start
//       (after reserving it locally) and COLLECT it (arrival semaphore -> push to compute)
//       after the mix of row + LAG (root: + 2*LAG). See mhc_post_combine.hpp.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/ttnn/operations/mhc_post/kernels/mhc_post_combine.hpp"

namespace {

constexpr uint32_t cb_coef_src = get_compile_time_arg_val(0);
constexpr uint32_t cb_coef_cols = get_compile_time_arg_val(1);
constexpr uint32_t cb_f_tiles = get_compile_time_arg_val(2);
constexpr uint32_t cb_x_tiles = get_compile_time_arg_val(3);
constexpr uint32_t n = get_compile_time_arg_val(4);
constexpr uint32_t block_c = get_compile_time_arg_val(5);
constexpr uint32_t tensor_c_tiles = get_compile_time_arg_val(6);
constexpr uint32_t n_coef_cols = get_compile_time_arg_val(7);
constexpr uint32_t post_width_tiles = get_compile_time_arg_val(8);
constexpr uint32_t comb_width_tiles = get_compile_time_arg_val(9);
constexpr uint32_t n_src = get_compile_time_arg_val(10);  // 1: post and comb share one page
constexpr uint32_t FB = 11;                               // fused block (descriptor fused_ct)
constexpr bool fused = get_compile_time_arg_val(FB + 0) != 0;
constexpr uint32_t cb_w_slice = get_compile_time_arg_val(FB + 2);
constexpr uint32_t cb_gather = get_compile_time_arg_val(FB + 8);
constexpr uint32_t cb_root_gather = get_compile_time_arg_val(FB + 9);
constexpr uint32_t Q = get_compile_time_arg_val(FB + 13);
constexpr uint32_t COMBINE_FANIN = get_compile_time_arg_val(FB + 14);
constexpr uint32_t COMBINE_DEPTH = get_compile_time_arg_val(FB + 15);
constexpr uint32_t COMBINE_LAG = get_compile_time_arg_val(FB + 16);
constexpr uint32_t SEM_GATHER = get_compile_time_arg_val(FB + 17);
constexpr uint32_t SEM_ROOT_GATHER = get_compile_time_arg_val(FB + 18);
constexpr uint32_t SEM_LEAD_CREDIT = get_compile_time_arg_val(FB + 19);
constexpr uint32_t SEM_ROOT_CREDIT = get_compile_time_arg_val(FB + 20);
static_assert(!fused || COMBINE_LAG + 1 <= COMBINE_DEPTH, "grants run LAG rows ahead of collects");

constexpr auto f_args = TensorAccessorArgs<FB + 22>();
constexpr auto x_args = TensorAccessorArgs<f_args.next_compile_time_args_offset()>();
constexpr auto post_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
constexpr auto comb_args = TensorAccessorArgs<post_args.next_compile_time_args_offset()>();
constexpr auto w_args = TensorAccessorArgs<comb_args.next_compile_time_args_offset()>();

constexpr uint32_t FACE_WORDS = 256;  // 16x16 fp32 face
constexpr uint32_t FACE_DIM = 16;

// Word offset of element (row t, col c) inside a 32x32 fp32 tile (faces 0..3 row-major).
FORCE_INLINE uint32_t tile_word(uint32_t t, uint32_t c) {
    return ((t / FACE_DIM) * 2 + (c / FACE_DIM)) * FACE_WORDS + (t % FACE_DIM) * FACE_DIM + (c % FACE_DIM);
}

// Copy `count` coefficients from source column `src_col` onward into column 0 of consecutive
// destination tiles starting at `dst_base` (32 tokens each). Only column 0 is meaningful.
FORCE_INLINE void expand_coefficients(uint32_t src_addr, uint32_t src_col, uint32_t dst_base, uint32_t count) {
    constexpr uint32_t tile_words = 1024;
    volatile tt_l1_ptr uint32_t* src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src_addr);
    volatile tt_l1_ptr uint32_t* dst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst_base);
    for (uint32_t q = 0; q < count; ++q) {
        volatile tt_l1_ptr uint32_t* dq = dst + q * tile_words;
        for (uint32_t t = 0; t < 32; ++t) {
            dq[tile_word(t, 0)] = src[tile_word(t, src_col + q)];
        }
    }
}

}  // namespace

void kernel_main() {
    const uint32_t f_addr = get_arg_val<uint32_t>(0);
    const uint32_t x_addr = get_arg_val<uint32_t>(1);
    const uint32_t post_addr = get_arg_val<uint32_t>(2);
    const uint32_t comb_addr = get_arg_val<uint32_t>(3);
    const uint32_t row_start = get_arg_val<uint32_t>(4);
    const uint32_t num_rows = get_arg_val<uint32_t>(5);
    const uint32_t c_start = get_arg_val<uint32_t>(6);
    const uint32_t num_c = get_arg_val<uint32_t>(7);
    const uint32_t post_col = get_arg_val<uint32_t>(8);
    const uint32_t comb_col = get_arg_val<uint32_t>(9);
    // arg 10: g_r (unused here)
    const uint32_t g_c = get_arg_val<uint32_t>(11);
    const uint32_t w_addr = get_arg_val<uint32_t>(12);

    const uint32_t f_tile_bytes = get_tile_size(cb_f_tiles);
    const uint32_t x_tile_bytes = get_tile_size(cb_x_tiles);
    const uint32_t coef_tile_bytes = get_tile_size(cb_coef_src);

    const auto f_acc = TensorAccessor(f_args, f_addr, f_tile_bytes);
    const auto x_acc = TensorAccessor(x_args, x_addr, x_tile_bytes);
    const auto post_acc = TensorAccessor(post_args, post_addr, coef_tile_bytes);
    const auto comb_acc = TensorAccessor(comb_args, comb_addr, coef_tile_bytes);

    // Reader-private scratch: slot 0 = post source tile, slot n_src-1 = comb source tile (the
    // same tile when one packed mixes page holds both windows). Never pushed.
    const uint32_t src_post_l1 = get_write_ptr(cb_coef_src);
    const uint32_t src_comb_l1 = src_post_l1 + (n_src - 1) * coef_tile_bytes;

    const uint32_t num_blocks = (num_c + block_c - 1) / block_c;

    // --- fused: resident W slice, block-major [b][j][c] (compact K = n * extent per block) ---
    const auto geo = mhc_post_combine::Geometry::of(g_c, Q, COMBINE_FANIN);
    uint32_t lead_granted = 0, lead_pushed = 0, root_granted = 0, root_pushed = 0;
    if constexpr (fused) {
        const uint32_t w_tile_bytes = get_tile_size(cb_w_slice);
        const auto w_acc = TensorAccessor(w_args, w_addr, w_tile_bytes);
        cb_reserve_back(cb_w_slice, n * num_c);
        uint32_t w_l1 = get_write_ptr(cb_w_slice);
        for (uint32_t b = 0; b < num_blocks; ++b) {
            const uint32_t c0 = c_start + b * block_c;
            const uint32_t extent = (num_c - b * block_c) < block_c ? (num_c - b * block_c) : block_c;
            for (uint32_t j = 0; j < n; ++j) {
                for (uint32_t c = 0; c < extent; ++c) {
                    noc_async_read_tile(j * tensor_c_tiles + c0 + c, w_acc, w_l1);
                    w_l1 += w_tile_bytes;
                }
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb_w_slice, n * num_c);
    }
    constexpr uint32_t RT_MEMBERS = 13;  // (x, y) of my group's members, then of every group leader
    const uint32_t rt_leaders = RT_MEMBERS + 2 * geo.size;

    // GRANT row r's slot sets: reserve them locally (compute has popped their previous
    // occupants), then credit every sender.
    auto grant = [&](uint32_t r) {
        if (geo.leader) {
            cb_reserve_back(cb_gather, geo.slots * (lead_granted - lead_pushed + 1));
            ++lead_granted;
            const uint32_t sem = get_semaphore(SEM_LEAD_CREDIT);
            for (uint32_t m = 0; m < geo.size; ++m) {
                const uint32_t px = get_arg_val<uint32_t>(RT_MEMBERS + 2 * m);
                const uint32_t py = get_arg_val<uint32_t>(RT_MEMBERS + 2 * m + 1);
                noc_semaphore_inc(get_noc_addr(px, py, sem), 1);
            }
        }
        if (geo.roots(r)) {
            cb_reserve_back(cb_root_gather, geo.groups * (root_granted - root_pushed + 1));
            ++root_granted;
            const uint32_t sem = get_semaphore(SEM_ROOT_CREDIT + geo.group);
            for (uint32_t g = 0; g < geo.groups; ++g) {
                const uint32_t px = get_arg_val<uint32_t>(rt_leaders + 2 * g);
                const uint32_t py = get_arg_val<uint32_t>(rt_leaders + 2 * g + 1);
                noc_semaphore_inc(get_noc_addr(px, py, sem), 1);
            }
        }
    };
    // COLLECT: wait for every sender of the row's slot set, then hand the set to compute.
    auto collect_post_row = [&](uint32_t r) {
        if (geo.leader && r >= COMBINE_LAG && r - COMBINE_LAG < num_rows) {
            const uint32_t rl = r - COMBINE_LAG;
            auto* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_GATHER + rl % COMBINE_DEPTH));
            noc_semaphore_wait_min(sem, (rl / COMBINE_DEPTH + 1) * geo.size);
            cb_push_back(cb_gather, geo.slots);
            ++lead_pushed;
        }
        if (r >= 2 * COMBINE_LAG && r - 2 * COMBINE_LAG < num_rows && geo.roots(r - 2 * COMBINE_LAG)) {
            const uint32_t k = geo.root_ordinal(r - 2 * COMBINE_LAG);
            auto* sem =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_ROOT_GATHER + k % COMBINE_DEPTH));
            noc_semaphore_wait_min(sem, (k / COMBINE_DEPTH + 1) * geo.groups);
            cb_push_back(cb_root_gather, geo.groups);
            ++root_pushed;
        }
    };

    // Issue the reads of one F/X block (nominal reserve; only valid tiles read). No barrier.
    auto issue_block = [&](uint32_t row, uint32_t b) {
        const uint32_t c0 = c_start + b * block_c;
        const uint32_t extent = (num_c - b * block_c) < block_c ? (num_c - b * block_c) : block_c;
        cb_reserve_back(cb_f_tiles, block_c);
        cb_reserve_back(cb_x_tiles, n * block_c);
        uint32_t f_l1 = get_write_ptr(cb_f_tiles);
        const uint32_t x_l1 = get_write_ptr(cb_x_tiles);
        const uint32_t f_page0 = row * tensor_c_tiles + c0;
        for (uint32_t c = 0; c < extent; ++c) {
            noc_async_read_tile(f_page0 + c, f_acc, f_l1);
            f_l1 += f_tile_bytes;
        }
        for (uint32_t i = 0; i < n; ++i) {
            const uint32_t x_page0 = (row * n + i) * tensor_c_tiles + c0;
            uint32_t x_dst = x_l1 + i * block_c * x_tile_bytes;
            for (uint32_t c = 0; c < extent; ++c) {
                noc_async_read_tile(x_page0 + c, x_acc, x_dst);
                x_dst += x_tile_bytes;
            }
        }
    };
    auto push_block = [&]() {
        cb_push_back(cb_f_tiles, block_c);
        cb_push_back(cb_x_tiles, n * block_c);
    };

    const uint32_t total = fused ? num_rows + 2 * COMBINE_LAG : num_rows;
    for (uint32_t r = 0; r < total; ++r) {
        if (r >= num_rows) {
            if constexpr (fused) {
                collect_post_row(r);  // drain the trailing combines
            }
            continue;
        }
        const uint32_t row = row_start + r;

        // load_coefficients: source reads + the row's first block, one barrier.
        noc_async_read_tile(row * post_width_tiles + post_col / 32, post_acc, src_post_l1);
        if constexpr (n_src > 1) {
            noc_async_read_tile(row * comb_width_tiles + comb_col / 32, comb_acc, src_comb_l1);
        }
        issue_block(row, 0);
        if constexpr (fused) {
            grant(r);
        }
        noc_async_read_barrier();

        cb_reserve_back(cb_coef_cols, n_coef_cols);
        const uint32_t coef_l1 = get_write_ptr(cb_coef_cols);
        expand_coefficients(src_post_l1, post_col % 32, coef_l1, n);
        expand_coefficients(src_comb_l1, comb_col % 32, coef_l1 + n * coef_tile_bytes, n * n);
        cb_push_back(cb_coef_cols, n_coef_cols);
        push_block();

        // load_block for the rest of the row.
        for (uint32_t b = 1; b < num_blocks; ++b) {
            issue_block(row, b);
            noc_async_read_barrier();
            push_block();
        }
        if constexpr (fused) {
            collect_post_row(r);
        }
    }
    if constexpr (fused) {
        noc_async_atomic_barrier();
    }
}
