// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// mhc_post reader (RISCV_1, NoC0).
//
// Per token-tile row t of this core:
//   load_coefficients_row: read the post / comb tile of row t (once when both are the same buffer) into the
//     private landing cb_coef_raw, then copy coefficient column k into column 0 of its own page of
//     cb_coef_cols (unpack-to-dest COL broadcast reads column 0 of faces 0 / 2). Page order:
//     [post_0..post_{n-1} | comb_00..comb_{n-1,n-1}] (comb row-major, comb[i][j] at page n + i*n + j).
//   load_block: per pass of block_cols units (column tiles c), one F tile and the n X tiles of stream i = 0..n-1
//     per unit, unit-major, one barrier per block; the ring holds IN_BLOCKS blocks.
// R2 (stats / collapse):
//   collapse: P (collapse_pre) columns 0..n-1 expand into pages n + n^2 .. of cb_coef_cols (one read when P is the
//     packed mixes tensor).
//   W slice: the rows of W for my columns, laid out per column pass in the matmul's K order (= the order compute
//     packs cb_mixed_bf16): k = j*block_cols + u. Resident (read once, pushed once) or, W-streamed, re-read per
//     row pass by pass.
//   A ragged pass runs through the full-block compute (one instantiation of every chain: kernel-config budget):
//     its filler F / X slots and W slots are ZEROED, so the filler units contribute exact zeros to X'@W, to the
//     sum of squares and to y (the writer drops their X' / y).
// norm: the norm weight tiles of my columns (filler zeroed) are resident in cb_norm_w (issued up front, pushed
//   behind the first row's barrier). After my rows, the roots unicast each row's rstd (32 fp32 tokens) into slot
//   (t - t_start) of my private landing cb_rstd_land and inc SEM_RSTD_COUNT; once all my rows' rstd landed, each is
//   expanded into column 0 of a cb_rstd_col page (the compute's COL broadcast), row by row. (Lagged norm knob,
//   Refinement 3: the writer does this per row instead.)

// -Os: the worker program's binaries + 2x its args / CB config must fit the 70 656 B kernel-config ring for trace
// replays to overlap (else every launch waits for the previous one: DISPATCH-BOUND); the dataflow kernels' code speed
// is not on the critical path (they issue NoC transfers and wait on the compute).
#pragma GCC optimize("Os")
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

namespace {

constexpr uint32_t FACE_WORDS = 256;  // 16 x 16 fp32
constexpr uint32_t FACE_DIM = 16;

// Copy column `col` of a 32x32 fp32 tile into column 0 of `dst` (32 words: 16 in face 0, 16 in face 2).
// Plain (non-volatile) pointers: both are this RISC's private L1 pages, landed behind a read barrier.
// noinline: three call sites (post / comb / P), 24 calls a row off the critical path (kernel-config buffer budget)
__attribute__((noinline)) void copy_column_to_col0(const uint32_t* src, uint32_t col, uint32_t* dst) {
    const uint32_t src_face_col = (col / FACE_DIM);
    const uint32_t c = col % FACE_DIM;
    for (uint32_t half = 0; half < 2; ++half) {
        const uint32_t* s = src + (half * 2 + src_face_col) * FACE_WORDS + c;
        uint32_t* d = dst + (half * 2) * FACE_WORDS;
#pragma GCC unroll 16
        for (uint32_t r = 0; r < FACE_DIM; ++r) {
            d[r * FACE_DIM] = s[r * FACE_DIM];
        }
    }
}

// Zero `bytes` of local L1 from the hardware zero region (async; covered by the next read barrier).
inline void zero_l1(uint32_t l1, uint32_t bytes) {
    const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
    for (uint32_t off = 0; off < bytes; off += MEM_ZEROS_SIZE) {
        noc_async_read(zeros, l1 + off, MEM_ZEROS_SIZE);
    }
}

}  // namespace

void kernel_main() {
    constexpr uint32_t cb_sublayer_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t cb_stream_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t cb_coef_raw = get_compile_time_arg_val(2);
    constexpr uint32_t cb_coef_cols = get_compile_time_arg_val(3);
    constexpr uint32_t n = get_compile_time_arg_val(4);
    constexpr uint32_t block_rows = get_compile_time_arg_val(5);
    constexpr uint32_t block_cols = get_compile_time_arg_val(6);
    constexpr uint32_t c_tiles = get_compile_time_arg_val(7);  // tensor_c_tiles
    constexpr uint32_t post_wt = get_compile_time_arg_val(8);
    constexpr uint32_t comb_wt = get_compile_time_arg_val(9);
    constexpr bool has_w = get_compile_time_arg_val(10) != 0;
    constexpr bool collapse = get_compile_time_arg_val(11) != 0;
    constexpr bool w_resident = get_compile_time_arg_val(12) != 0;
    constexpr uint32_t cb_w_slice = get_compile_time_arg_val(13);
    constexpr uint32_t pre_wt = get_compile_time_arg_val(14);
    constexpr bool norm = get_compile_time_arg_val(15) != 0;
    constexpr uint32_t cb_norm_w = get_compile_time_arg_val(16);
    constexpr uint32_t cb_rstd_col = get_compile_time_arg_val(17);
    constexpr uint32_t cb_rstd_land = get_compile_time_arg_val(18);
    constexpr uint32_t sem_rstd_count_id = get_compile_time_arg_val(19);
    // norm schedule knob (Refinement 3): 0 = deferred roots, rstd expanded here after my rows; > 0 = lagged, the
    // writer expands each row's rstd norm_lag rows after it (this tail is compiled out)
    constexpr uint32_t root_lag = get_compile_time_arg_val(20);
    constexpr bool deferred_norm = norm && root_lag == 0;
    // slice sharing across the R row groups (see issue_slices); 1 = every core reads its whole slices
    constexpr uint32_t slice_share = get_compile_time_arg_val(21);
    constexpr uint32_t sem_w_share_id = get_compile_time_arg_val(22);
    constexpr auto f_args = TensorAccessorArgs<23>();
    constexpr auto x_args = TensorAccessorArgs<f_args.next_compile_time_args_offset()>();
    constexpr auto post_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto comb_args = TensorAccessorArgs<post_args.next_compile_time_args_offset()>();
    [[maybe_unused]] constexpr auto pre_args = TensorAccessorArgs<comb_args.next_compile_time_args_offset()>();
    [[maybe_unused]] constexpr auto w_args = TensorAccessorArgs<pre_args.next_compile_time_args_offset()>();
    [[maybe_unused]] constexpr auto nw_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
    constexpr uint32_t n_mix_coef = n + n * n;
    constexpr uint32_t n_coef = n_mix_coef + (collapse ? n : 0);
    constexpr uint32_t w_pass_tiles = n * block_cols;  // W tiles of one column pass (ragged passes pad)

    const uint32_t t_start = get_arg_val<uint32_t>(0);
    const uint32_t t_count = get_arg_val<uint32_t>(1);
    const uint32_t c_start = get_arg_val<uint32_t>(2);
    const uint32_t c_count = get_arg_val<uint32_t>(3);
    [[maybe_unused]] const uint32_t group = get_arg_val<uint32_t>(4);
    [[maybe_unused]] const uint32_t member_q = get_arg_val<uint32_t>(5);

    const uint32_t f_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t x_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t post_addr = get_common_arg_val<uint32_t>(2);
    const uint32_t comb_addr = get_common_arg_val<uint32_t>(3);
    const uint32_t post_col = get_common_arg_val<uint32_t>(4);
    const uint32_t comb_col = get_common_arg_val<uint32_t>(5);
    [[maybe_unused]] const uint32_t pre_addr = get_common_arg_val<uint32_t>(6);
    [[maybe_unused]] const uint32_t w_addr = get_common_arg_val<uint32_t>(7);
    [[maybe_unused]] const uint32_t nw_addr = get_common_arg_val<uint32_t>(8);

    const uint32_t f_page = get_tile_size(cb_sublayer_tiles);
    const uint32_t x_page = get_tile_size(cb_stream_tiles);
    const uint32_t coef_page = get_tile_size(cb_coef_raw);
    const auto f_acc = TensorAccessor(f_args, f_addr, f_page);
    const auto x_acc = TensorAccessor(x_args, x_addr, x_page);
    const auto post_acc = TensorAccessor(post_args, post_addr, coef_page);
    const auto comb_acc = TensorAccessor(comb_args, comb_addr, coef_page);
    const bool coef_shared = (post_addr == comb_addr);  // packed mixes: one tile serves post and comb

    // Reader-private landing of the raw coefficient tiles (never pushed), double-buffered by row parity so the
    // next row's tiles are in flight while this row expands. Per buffer: post, comb, P (shared when aliased).
    constexpr uint32_t n_raw = collapse ? 3 : 2;
    const uint32_t raw_base = get_write_ptr(cb_coef_raw);
    const uint32_t coef_words = coef_page / sizeof(uint32_t);  // fp32 words per coefficient page
    // collapse: P shares the landing of post / comb when it is the same buffer, else its own page
    const bool pre_is_post = collapse && pre_addr == post_addr;
    const bool pre_is_comb = collapse && pre_addr == comb_addr;
    auto raw_post = [&](uint32_t b) { return raw_base + b * n_raw * coef_page; };
    auto raw_comb = [&](uint32_t b) { return coef_shared ? raw_post(b) : raw_post(b) + coef_page; };
    auto raw_pre = [&](uint32_t b) {
        return pre_is_post ? raw_post(b) : (pre_is_comb ? raw_comb(b) : raw_post(b) + 2 * coef_page);
    };
    // load_coefficients_row, issue half: the raw tiles of row t into landing buffer b (no barrier)
    auto issue_coef = [&](uint32_t t, uint32_t b) {
        noc_async_read(post_acc.get_noc_addr(t * post_wt), raw_post(b), coef_page);
        if (!coef_shared) {
            noc_async_read(comb_acc.get_noc_addr(t * comb_wt), raw_comb(b), coef_page);
        }
        if constexpr (collapse) {
            if (!pre_is_post && !pre_is_comb) {
                const auto pre_acc = TensorAccessor(pre_args, pre_addr, coef_page);
                noc_async_read(pre_acc.get_noc_addr(t * pre_wt), raw_pre(b), coef_page);
            }
        }
    };
    // load_coefficients_row, expand half: columns -> column 0 of the cb_coef_cols pages (landed buffer b)
    auto expand_coef = [&](uint32_t b) {
        cb_reserve_back(cb_coef_cols, n_coef);
        uint32_t* cols = reinterpret_cast<uint32_t*>(get_write_ptr(cb_coef_cols));
        const uint32_t* post_src = reinterpret_cast<const uint32_t*>(raw_post(b));
        const uint32_t* comb_src = reinterpret_cast<const uint32_t*>(raw_comb(b));
        for (uint32_t k = 0; k < n; ++k) {
            copy_column_to_col0(post_src, post_col + k, cols + k * coef_words);
        }
        for (uint32_t k = 0; k < n * n; ++k) {
            copy_column_to_col0(comb_src, comb_col + k, cols + (n + k) * coef_words);
        }
        if constexpr (collapse) {
            const uint32_t* pre_src = reinterpret_cast<const uint32_t*>(raw_pre(b));
            for (uint32_t k = 0; k < n; ++k) {  // P_i = collapse_pre column i
                copy_column_to_col0(pre_src, k, cols + (n_mix_coef + k) * coef_words);
            }
        }
        cb_push_back(cb_coef_cols, n_coef);
    };
    // W slice (R2): the W tile rows of my columns for column pass `p`, in the matmul's K order.
    [[maybe_unused]] const uint32_t w_page = get_tile_size(cb_w_slice);
    // Only the pass's slots k in [k_lo, k_hi) (W sharing: my share of the slice; default the whole pass).
    auto read_w_pass = [&](uint32_t p, uint32_t l1, uint32_t k_lo, uint32_t k_hi) {
        if constexpr (has_w) {
            const auto w_acc = TensorAccessor(w_args, w_addr, w_page);
            const uint32_t cb0 = p * block_cols;
            const uint32_t cols = (c_count - cb0) < block_cols ? (c_count - cb0) : block_cols;
            for (uint32_t j = 0; j < n; ++j) {
                for (uint32_t u = 0; u < block_cols; ++u) {
                    const uint32_t k = j * block_cols + u;
                    if (k < k_lo || k >= k_hi) {
                        continue;
                    }
                    if (u < cols) {
                        noc_async_read(w_acc.get_noc_addr(j * c_tiles + c_start + cb0 + u), l1 + k * w_page, w_page);
                    } else {
                        zero_l1(l1 + k * w_page, w_page);  // filler unit of a ragged pass
                    }
                }
            }
        }
    };
    // Resident slices — the W slice (R2, W-resident) and the norm weight tiles (norm), both one per column of mine
    // (+ zeroed filler of a ragged pass), identical on the R cores of my column member. Issued behind the first
    // block's barrier (the first block and the coefficients land first, so the compute mixes pass 0 while they
    // arrive; it waits for W only at its first matmul K-block, for the norm weight only in norm_block), pushed
    // behind the next barrier. Slice sharing (slice_share = R > 1): each of the R cores reads 1/R of every slice
    // from DRAM and NoC-writes it into the same L1 of the other R - 1 (a slice crosses DRAM once, not R times), then
    // incs their SEM_W_SHARE.
    struct Slice {
        uint32_t l1, page, used, lo, hi;  // used: slots of my columns (+ filler); [lo, hi): my share
    };
    [[maybe_unused]] Slice w_sl{0, 0, 0, 0, 0}, nw_sl{0, 0, 0, 0, 0};
    [[maybe_unused]] uint32_t w_total = 0, nw_total = 0;
    auto my_share = [&](Slice& sl) {
        const uint32_t share = (sl.used + slice_share - 1) / slice_share;
        const uint32_t idx = slice_share > 1 ? group : 0;
        sl.lo = idx * share < sl.used ? idx * share : sl.used;
        sl.hi = sl.lo + share < sl.used ? sl.lo + share : sl.used;
    };
    auto issue_slices = [&]() {
        if constexpr (has_w && w_resident) {
            const uint32_t passes = (c_count + block_cols - 1) / block_cols;
            w_total = get_local_cb_interface(cb_w_slice).fifo_num_pages;
            cb_reserve_back(cb_w_slice, w_total);
            w_sl = Slice{get_write_ptr(cb_w_slice), w_page, passes * w_pass_tiles, 0, 0};
            my_share(w_sl);
            for (uint32_t p = 0; p < passes; ++p) {
                const uint32_t s0 = p * w_pass_tiles;
                const uint32_t lo = w_sl.lo > s0 ? w_sl.lo - s0 : 0;
                const uint32_t hi = w_sl.hi > s0 ? (w_sl.hi - s0 < w_pass_tiles ? w_sl.hi - s0 : w_pass_tiles) : 0;
                if (lo < hi) {
                    read_w_pass(p, w_sl.l1 + s0 * w_page, lo, hi);
                }
            }
        }
        if constexpr (norm) {
            nw_total = get_local_cb_interface(cb_norm_w).fifo_num_pages;
            const uint32_t nw_page = get_tile_size(cb_norm_w);
            const auto nw_acc = TensorAccessor(nw_args, nw_addr, nw_page);
            cb_reserve_back(cb_norm_w, nw_total);
            nw_sl = Slice{get_write_ptr(cb_norm_w), nw_page, nw_total, 0, 0};
            my_share(nw_sl);
            for (uint32_t k = nw_sl.lo; k < nw_sl.hi; ++k) {
                if (k < c_count) {
                    noc_async_read(nw_acc.get_noc_addr(c_start + k), nw_sl.l1 + k * nw_page, nw_page);
                } else {
                    zero_l1(nw_sl.l1 + k * nw_page, nw_page);
                }
            }
        }
    };
    // Slice sharing: forward my shares (landed behind a read barrier) to the R - 1 peers, then wait for theirs.
    auto share_slices = [&]() {
        if constexpr (slice_share > 1) {
            const uint32_t sem = get_semaphore(sem_w_share_id);
            auto peer = [&](uint32_t g, uint32_t addr) {  // RT args 6..: the R cores of my column member, by group
                const uint32_t xy = get_arg_val<uint32_t>(6 + g);
                return get_noc_addr(xy & 0xFFFF, xy >> 16, addr);
            };
            auto forward = [&](const Slice& sl) {
                if (sl.lo < sl.hi) {
                    const uint32_t a = sl.l1 + sl.lo * sl.page;
                    for (uint32_t g = 0; g < slice_share; ++g) {
                        if (g != group) {
                            noc_async_write(a, peer(g, a), (sl.hi - sl.lo) * sl.page);
                        }
                    }
                }
            };
            forward(w_sl);
            forward(nw_sl);
            noc_async_write_barrier();
            for (uint32_t g = 0; g < slice_share; ++g) {  // every peer counts every core (empty shares too)
                if (g != group) {
                    noc_semaphore_inc(peer(g, sem), 1);
                }
            }
            volatile tt_l1_ptr uint32_t* cnt = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem);
            noc_semaphore_wait_min(cnt, slice_share - 1);
            noc_semaphore_set(cnt, 0);
            noc_async_atomic_barrier();
        }
    };
    // behind a read barrier that covered issue_slices: share them, push W and the norm weight
    [[maybe_unused]] bool slices_pending = false;
    auto push_slices = [&]() {
        share_slices();
        if constexpr (has_w && w_resident) {
            cb_push_back(cb_w_slice, w_total);
        }
        if constexpr (norm) {
            cb_push_back(cb_norm_w, nw_total);
        }
        slices_pending = false;
    };
    if (t_count > 0) {
        issue_coef(t_start, 0);
    }

    uint32_t row_idx = 0;
    for (uint32_t rb = 0; rb < t_count; rb += block_rows) {
        const uint32_t rows_in_block = (t_count - rb) < block_rows ? (t_count - rb) : block_rows;
        for (uint32_t r = 0; r < rows_in_block; ++r, ++row_idx) {
            const uint32_t t = t_start + rb + r;
            const uint32_t buf = row_idx & 1;

            // ---- load_block, one pass = one block of block_cols ring slots, one barrier ----
            // A ragged pass reads only its cols_in_pass units but still occupies block_cols slots (filler,
            // never read): every consumer window then starts on a block boundary and never wraps the ring.
            // The first pass's reads are in flight with row t's coefficient tiles (prefetched a row earlier);
            // behind its barrier the reader prefetches row t+1's tiles, then expands row t's.
            const uint32_t f_row = t * c_tiles;
            const uint32_t x_row = t * n * c_tiles;
            for (uint32_t cb0 = 0; cb0 < c_count; cb0 += block_cols) {
                const uint32_t cols_in_pass = (c_count - cb0) < block_cols ? (c_count - cb0) : block_cols;
                if constexpr (has_w && !w_resident) {  // R2-W-streamed: this pass's W rows again
                    cb_reserve_back(cb_w_slice, w_pass_tiles);
                    read_w_pass(cb0 / block_cols, get_write_ptr(cb_w_slice), 0, w_pass_tiles);
                }
                cb_reserve_back(cb_sublayer_tiles, block_cols);
                cb_reserve_back(cb_stream_tiles, block_cols * n);
                uint32_t f_l1 = get_write_ptr(cb_sublayer_tiles);
                uint32_t x_l1 = get_write_ptr(cb_stream_tiles);
                for (uint32_t u = 0; u < cols_in_pass; ++u) {
                    const uint32_t c = c_start + cb0 + u;
#ifndef ABL_NOREAD
                    noc_async_read(f_acc.get_noc_addr(f_row + c), f_l1, f_page);
#endif
                    f_l1 += f_page;
                    for (uint32_t i = 0; i < n; ++i) {
#ifndef ABL_NOREAD
                        noc_async_read(x_acc.get_noc_addr(x_row + i * c_tiles + c), x_l1, x_page);
#endif
                        x_l1 += x_page;
                    }
                }
                if constexpr (has_w) {  // filler units of a ragged pass: zeros (computed, contribute exactly 0)
                    const uint32_t filler = block_cols - cols_in_pass;
                    if (filler > 0) {
                        zero_l1(f_l1, filler * f_page);
                        zero_l1(x_l1, filler * n * x_page);
                    }
                }
                noc_async_read_barrier();
                if (slices_pending) {
                    push_slices();
                }
                if (row_idx == 0 && cb0 == 0) {
                    issue_slices();
                    slices_pending = (has_w && w_resident) || norm || slice_share > 1;
                }
                if (cb0 == 0) {
                    if (row_idx + 1 < t_count) {
                        issue_coef(t + 1, buf ^ 1);
                    }
                    expand_coef(buf);
                }
                if constexpr (has_w && !w_resident) {
                    cb_push_back(cb_w_slice, w_pass_tiles);
                }
                cb_push_back(cb_sublayer_tiles, block_cols);
                cb_push_back(cb_stream_tiles, block_cols * n);
            }
        }
    }

    if (slices_pending) {  // my only block was the first: W / the norm weight still in flight
        noc_async_read_barrier();
        push_slices();
    }

    // ---- norm (deferred): every row's rstd from its root -> column 0 of a cb_rstd_col page, in row order ----
    if constexpr (deferred_norm) {
        volatile tt_l1_ptr uint32_t* landed =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_rstd_count_id));
        noc_semaphore_wait_min(landed, t_count);  // the roots finish their rows in any order: wait for all
        noc_semaphore_set(landed, 0);
        invalidate_l1_cache();
        const uint32_t land = get_write_ptr(cb_rstd_land);  // never pushed: private landing, same address everywhere
        for (uint32_t r = 0; r < t_count; ++r) {
            const volatile tt_l1_ptr uint32_t* src =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(land + r * 32 * sizeof(uint32_t));
            cb_reserve_back(cb_rstd_col, 1);
            uint32_t* dst = reinterpret_cast<uint32_t*>(get_write_ptr(cb_rstd_col));
            for (uint32_t l = 0; l < 32; ++l) {  // token l -> (row l, column 0): face 0 rows 0..15, face 2 16..31
                dst[(l / FACE_DIM) * 2 * FACE_WORDS + (l % FACE_DIM) * FACE_DIM] = src[l];
            }
            cb_push_back(cb_rstd_col, 1);
        }
    }
}
