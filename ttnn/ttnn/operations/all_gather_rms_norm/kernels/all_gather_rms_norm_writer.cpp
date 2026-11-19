// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// all_gather_rms_norm member writer (BRISC / NoC1): load_affine_rows (once) + exchange_partials_block (k > 1) +
// store_block.
//
// load_affine_rows: this member's gamma (and beta) columns into cb_gamma / cb_beta. Compute consumes them ROW-
//   broadcast, so a page only needs row 0 of faces 0/1 = the 32 datums of the member's slice-local tile j, i.e.
//   dense columns [q * W_local + 32 * (c0 + j), + 32) of the full-W weight (zero past the slice's valid width).
//   The dense "sticks" covering them (TILE: row 0 of gamma tile s = two 64 B row-0 reads of faces 0/1; ROW_MAJOR
//   sticks: page s) are staged in L1, then scattered element-wise into the face-0 / face-1 row 0 of each page
//   (the shift s_q = (q * W_local) mod 32 of a non-aligned slice is just an element offset here). BH DRAM reads
//   keep their source offset mod 64 B: every staged read lands on a 64 B-aligned slot.
// exchange_partials_block (base op, symmetric all-gather, k <= 8): member m writes its n_stat * rows column-0
//   partial tiles (faces 0/2 only) into slot ((r * k + m) * n_stat + s) of EVERY member's cb_gathered (lockstep slot
//   address = this core's own write pointer), flushes, bumps each member's SEM_PARTIALS once, waits for k arrivals.
// store_block: aligned output — the member's tile-row r goes to output tiles t * out_w_tiles + out_col0 + j,
//   in units of <= out_unit_tiles per wait / flush / pop (clamped at the cb_out ring end).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "all_gather_rms_norm_common.hpp"

using namespace agrms_cb;

constexpr uint32_t out_tile_bytes = get_compile_time_arg_val(0);
constexpr uint32_t pitch = get_compile_time_arg_val(1);
constexpr uint32_t out_w_tiles = get_compile_time_arg_val(2);
constexpr uint32_t k = get_compile_time_arg_val(3);
constexpr uint32_t out_unit = get_compile_time_arg_val(4);
constexpr uint32_t out_cb_pages = get_compile_time_arg_val(5);
constexpr uint32_t n_stat = get_compile_time_arg_val(6);
constexpr uint32_t sem_partials_id = get_compile_time_arg_val(7);
constexpr uint32_t acc_tile_bytes = get_compile_time_arg_val(8);
constexpr uint32_t g_elem = get_compile_time_arg_val(9);
constexpr bool g_is_tile = get_compile_time_arg_val(10) != 0;
constexpr uint32_t g_page_bytes = get_compile_time_arg_val(11);  // gamma tensor page (TILE: tile, RM: stick)
constexpr uint32_t g_l1_page = get_compile_time_arg_val(12);     // cb_gamma page (a gamma-dtype tile)
constexpr uint32_t slot_bytes = get_compile_time_arg_val(13);
constexpr uint32_t sem_foreign_id = get_compile_time_arg_val(14);
constexpr uint32_t max_fin = get_compile_time_arg_val(15);   // dense: foreign slots per tile-row (home side)
constexpr uint32_t max_fout = get_compile_time_arg_val(16);  // dense: contribution tiles per tile-row (sender side)
// slices tile-aligned (W_local % 32 == 0): every local gamma tile IS a dense gamma tile -> direct row-0 reads
constexpr bool affine_aligned = get_compile_time_arg_val(17) != 0;
constexpr auto out_args = TensorAccessorArgs<18>();
constexpr auto gamma_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();
constexpr auto beta_args = TensorAccessorArgs<gamma_args.next_compile_time_args_offset()>();

constexpr uint32_t TILE_DIM = 32;
constexpr uint32_t FACE_DIM = 16;

FORCE_INLINE uint32_t arg(uint32_t i) { return get_arg_val<uint32_t>(i); }

#ifdef AGRMS_HAS_GAMMA
constexpr uint32_t g_face_bytes = FACE_DIM * FACE_DIM * g_elem;
constexpr uint32_t READ_ALIGN = 64;
// staged stick: datum e at slot + (e < 16 ? e : half + e - 16) * ... (TILE: faces 0/1 row 0 at +0 / +64)
constexpr uint32_t half_off = g_is_tile ? READ_ALIGN : FACE_DIM * g_elem;

FORCE_INLINE uint32_t staged_elem_addr(uint32_t slot, uint32_t e) {
    return e < FACE_DIM ? slot + e * g_elem : slot + half_off + (e - FACE_DIM) * g_elem;
}

FORCE_INLINE void put_elem(uint32_t dst, uint32_t src) {
    if constexpr (g_elem == 4) {
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst) =
            src ? *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src) : 0u;
    } else {
        *reinterpret_cast<volatile tt_l1_ptr uint16_t*>(dst) =
            src ? *reinterpret_cast<volatile tt_l1_ptr uint16_t*>(src) : 0u;
    }
}

// load_affine_rows for one weight tensor (gamma or beta) into `cb`, split in two so the DRAM reads are in flight
// while the writer serves the partial exchange (base op gamma_after_send): issue() reserves and issues the reads,
// finish() waits for them, completes the rows and pushes.
//   aligned slices: page j = dense tile s = q * sw + c0 + j. TILE: faces 0/1 row 0 straight from DRAM (2 x 64 B);
//     ROW_MAJOR bf16: the 64 B stick into face-0 rows 0..1, then one local copy of its upper half to face-1 row 0;
//     ROW_MAJOR fp32: the two 64 B halves straight to face 0 / face 1 row 0 (a DRAM read keeps its offset mod 64 B).
//   non-aligned slices (element shift s_q): stage the dense sticks, RISC-V scatter into the pages.
template <typename Accessor>
struct AffineLoad {
    const Accessor& acc;
    uint32_t cb;
    uint32_t stage;
    uint32_t parts_idx;
    uint32_t n_parts;
    uint32_t page_base = 0;

    FORCE_INLINE void issue() {
        cb_reserve_back(cb, pitch);
        page_base = get_write_ptr(cb);
        uint32_t slot = 0;
        for (uint32_t p = 0; p < n_parts; ++p) {
            const uint32_t dc0 = arg(parts_idx + 4 * p);
            const uint32_t rem = arg(parts_idx + 4 * p + 1);
            const uint32_t n = arg(parts_idx + 4 * p + 2);
            const uint32_t off = arg(parts_idx + 4 * p + 3);
            if constexpr (affine_aligned) {
                const uint32_t s0 = dc0 / TILE_DIM;
                for (uint32_t j = 0; j < n; ++j) {
                    const uint32_t page = page_base + (off + j) * g_l1_page;
                    if constexpr (g_is_tile) {
                        noc_async_read(acc.get_noc_addr(s0 + j, 0), page, READ_ALIGN);
                        noc_async_read(acc.get_noc_addr(s0 + j, g_face_bytes), page + g_face_bytes, READ_ALIGN);
                    } else if constexpr (FACE_DIM * g_elem % READ_ALIGN == 0) {
                        noc_async_read(acc.get_noc_addr(s0 + j, 0), page, FACE_DIM * g_elem);
                        noc_async_read(
                            acc.get_noc_addr(s0 + j, FACE_DIM * g_elem), page + g_face_bytes, FACE_DIM * g_elem);
                    } else {
                        noc_async_read(acc.get_noc_addr(s0 + j, 0), page, TILE_DIM * g_elem);
                    }
                }
            } else {
                const uint32_t cols = rem < n * TILE_DIM ? rem : n * TILE_DIM;
                const uint32_t s0 = dc0 / TILE_DIM, s1 = (dc0 + cols - 1) / TILE_DIM;
                for (uint32_t s = s0; s <= s1; ++s, ++slot) {
                    const uint32_t dst = stage + slot * slot_bytes;
                    if constexpr (g_is_tile) {
                        noc_async_read(acc.get_noc_addr(s, 0), dst, READ_ALIGN);
                        noc_async_read(acc.get_noc_addr(s, g_face_bytes), dst + READ_ALIGN, READ_ALIGN);
                    } else {
                        noc_async_read(acc.get_noc_addr(s, 0), dst, TILE_DIM * g_elem);
                    }
                }
            }
        }
    }

    FORCE_INLINE void finish() {
        noc_async_read_barrier();
        if constexpr (affine_aligned) {
            if constexpr (!g_is_tile && FACE_DIM * g_elem % READ_ALIGN != 0) {
                // bf16 sticks: face-0 row 1 (the upper 16 datums) -> face-1 row 0, local NoC copies
                for (uint32_t p = 0; p < n_parts; ++p) {
                    const uint32_t n = arg(parts_idx + 4 * p + 2);
                    const uint32_t off = arg(parts_idx + 4 * p + 3);
                    for (uint32_t j = 0; j < n; ++j) {
                        const uint32_t page = page_base + (off + j) * g_l1_page;
                        noc_async_read(get_noc_addr(page + FACE_DIM * g_elem), page + g_face_bytes, FACE_DIM * g_elem);
                    }
                }
                noc_async_read_barrier();
            }
        } else {
            uint32_t slot = 0;
            for (uint32_t p = 0; p < n_parts; ++p) {
                const uint32_t dc0 = arg(parts_idx + 4 * p);
                const uint32_t rem = arg(parts_idx + 4 * p + 1);
                const uint32_t n = arg(parts_idx + 4 * p + 2);
                const uint32_t off = arg(parts_idx + 4 * p + 3);
                const uint32_t cols = rem < n * TILE_DIM ? rem : n * TILE_DIM;
                const uint32_t s0 = dc0 / TILE_DIM, s1 = (dc0 + cols - 1) / TILE_DIM;
                for (uint32_t j = 0; j < n; ++j) {
                    const uint32_t page = page_base + (off + j) * g_l1_page;
                    for (uint32_t e = 0; e < TILE_DIM; ++e) {
                        const uint32_t c = j * TILE_DIM + e;  // column relative to the part start
                        const uint32_t dst =
                            page + (e < FACE_DIM ? e * g_elem : g_face_bytes + (e - FACE_DIM) * g_elem);
                        if (c < cols) {
                            const uint32_t dense = dc0 + c;
                            const uint32_t sl = slot + dense / TILE_DIM - s0;
                            put_elem(dst, staged_elem_addr(stage + sl * slot_bytes, dense % TILE_DIM));
                        } else {
                            put_elem(dst, 0);
                        }
                    }
                }
                slot += s1 - s0 + 1;
            }
        }
        cb_push_back(cb, pitch);
    }
};
#endif

void kernel_main() {
    const uint32_t out_addr = arg(0);
    const uint32_t row_start = arg(1);
    const uint32_t rows = arg(2);
    const uint32_t out_col0 = arg(3);
    const uint32_t core_w = arg(4);
    const uint32_t member_idx = arg(5);
    [[maybe_unused]] const uint32_t gamma_addr = arg(6);
    [[maybe_unused]] const uint32_t beta_addr = arg(7);
    [[maybe_unused]] const uint32_t n_parts = arg(8);
    const uint32_t parts_idx = 9;
    const uint32_t peer_idx = arg(parts_idx + 4 * n_parts);

#ifdef AGRMS_HAS_GAMMA
    // non-aligned: gamma and beta share the staging area -> beta is staged only after gamma's scatter
    const uint32_t stage = (get_write_ptr(cb_affine_stage) + READ_ALIGN - 1) & ~(READ_ALIGN - 1);
    const auto g_acc = TensorAccessor(gamma_args, gamma_addr, g_page_bytes);
    AffineLoad<decltype(g_acc)> gamma_load{g_acc, cb_gamma, stage, parts_idx, n_parts};
#ifdef AGRMS_HAS_BETA
    const auto b_acc = TensorAccessor(beta_args, beta_addr, g_page_bytes);
    AffineLoad<decltype(b_acc)> beta_load{b_acc, cb_beta, stage, parts_idx, n_parts};
#endif
    auto finish_affine = [&]() {
        AGZ("wr_gamma");
        if constexpr (affine_aligned) {
            gamma_load.finish();
#ifdef AGRMS_HAS_BETA
            beta_load.finish();
#endif
        } else {
            gamma_load.issue();
            gamma_load.finish();
#ifdef AGRMS_HAS_BETA
            beta_load.issue();
            beta_load.finish();
#endif
        }
    };
    if constexpr (affine_aligned) {  // DRAM reads in flight while the partials are exchanged
        gamma_load.issue();
#ifdef AGRMS_HAS_BETA
        beta_load.issue();
#endif
    }
#endif

    // ---- exchange_partials_block (k > 1) ----
    if constexpr (k > 1) {
        AGZ("wr_xchg");
        const uint32_t sem_addr = get_semaphore(sem_partials_id);
        cb_wait_front(cb_partial, n_stat * rows);
        cb_reserve_back(cb_gathered, n_stat * k * rows);
        const uint32_t slot_base = get_write_ptr(cb_gathered);
        const uint32_t src = get_read_ptr(cb_partial);
        constexpr uint32_t face_bytes = acc_tile_bytes / 4;
        for (uint32_t m = 0; m < k; ++m) {
            const uint32_t px = arg(peer_idx + 2 * m), py = arg(peer_idx + 2 * m + 1);
            for (uint32_t r = 0; r < rows; ++r) {
                for (uint32_t s = 0; s < n_stat; ++s) {
                    const uint32_t from = src + (r * n_stat + s) * acc_tile_bytes;
                    const uint64_t to =
                        get_noc_addr(px, py, slot_base + ((r * k + member_idx) * n_stat + s) * acc_tile_bytes);
                    noc_async_write(from, to, face_bytes);                                    // face 0
                    noc_async_write(from + 2 * face_bytes, to + 2 * face_bytes, face_bytes);  // face 2
                }
            }
        }
        noc_async_writes_flushed();
        for (uint32_t m = 0; m < k; ++m) {
            noc_semaphore_inc(get_noc_addr(arg(peer_idx + 2 * m), arg(peer_idx + 2 * m + 1), sem_addr), 1);
        }
        cb_pop_front(cb_partial, n_stat * rows);
        noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_addr), k);
        cb_push_back(cb_gathered, n_stat * k * rows);
    }
#ifdef AGRMS_HAS_GAMMA
    finish_affine();
#endif

#ifdef AGRMS_DENSE
    // ---- assemble_dense_block, foreign half: my contributions to peers' home output tiles go into their cb_foreign
    // slot (r * max_fin + slot) (lockstep address = my own cb_foreign write pointer), one SEM_FOREIGN increment per
    // home; then wait for every contributor to my home tiles.
    {
        const uint32_t d_idx = peer_idx + 2 * k;
        const uint32_t n_fout = arg(d_idx);
        const uint32_t n_contrib = arg(d_idx + 1 + 2 * n_fout);
        const uint32_t n_homes = arg(d_idx + 2 + 2 * n_fout);
        const uint32_t homes_idx = d_idx + 3 + 2 * n_fout;
        const uint32_t sem_addr = get_semaphore(sem_foreign_id);
        if (n_fout > 0) {
            cb_wait_front(cb_contrib, rows * n_fout);
            const uint32_t src = get_read_ptr(cb_contrib);
            const uint32_t fbase = get_write_ptr(cb_foreign);
            for (uint32_t r = 0; r < rows; ++r) {
                for (uint32_t f = 0; f < n_fout; ++f) {
                    const uint32_t h = arg(d_idx + 1 + 2 * f), slot = arg(d_idx + 2 + 2 * f);
                    noc_async_write(
                        src + (r * n_fout + f) * out_tile_bytes,
                        get_noc_addr(
                            arg(peer_idx + 2 * h),
                            arg(peer_idx + 2 * h + 1),
                            fbase + (r * max_fin + slot) * out_tile_bytes),
                        out_tile_bytes);
                }
            }
            noc_async_writes_flushed();
            for (uint32_t i = 0; i < n_homes; ++i) {
                const uint32_t h = arg(homes_idx + i);
                noc_semaphore_inc(get_noc_addr(arg(peer_idx + 2 * h), arg(peer_idx + 2 * h + 1), sem_addr), 1);
            }
            cb_pop_front(cb_contrib, rows * n_fout);
        }
        if (n_contrib > 0) {
            cb_reserve_back(cb_foreign, rows * max_fin);
            noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_addr), n_contrib);
            cb_push_back(cb_foreign, rows * max_fin);
        }
    }
#endif

    // ---- store_block ----
    AGZ("wr_store");
    const auto out_acc = TensorAccessor(out_args, out_addr, out_tile_bytes);
    uint32_t ring_pos = 0;
    for (uint32_t r = 0; r < rows; ++r) {
        const uint32_t first = (row_start + r) * out_w_tiles + out_col0;
        for (uint32_t j0 = 0; j0 < core_w;) {
            uint32_t n = (core_w - j0) < out_unit ? (core_w - j0) : out_unit;
            n = (out_cb_pages - ring_pos) < n ? (out_cb_pages - ring_pos) : n;
            cb_wait_front(cb_out, n);
            const uint32_t l1 = get_read_ptr(cb_out);
            for (uint32_t i = 0; i < n; ++i) {
                noc_async_write(l1 + i * out_tile_bytes, out_acc.get_noc_addr(first + j0 + i), out_tile_bytes);
            }
            noc_async_writes_flushed();
            cb_pop_front(cb_out, n);
            j0 += n;
            ring_pos = (ring_pos + n == out_cb_pages) ? 0 : ring_pos + n;
        }
    }
    noc_async_write_barrier();
    noc_async_atomic_barrier();
}
