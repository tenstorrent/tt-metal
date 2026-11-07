// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// all_gather_rms_norm member writer (BRISC / NoC1): load_gamma_slice (once) + exchange_partials_block (k > 1) +
// store_block. Ported from the base op's writer (rms_norm_writer.cpp: GammaSliceLoad, exchange_send /
// exchange_recv_partials, the TILE store loop).
//
// load_gamma_slice: gamma pages [g0, g0 + n) (regime A: the member's landing = output tiles; regime N: every dense
//   output tile) into the compact cb_gamma pages compute multiplies ROW-broadcast (row 0 of faces 0/1 only):
//   TILE gamma (1,1,1,W): per tile, face 0 row 0 and face 1 row 0 straight from DRAM (64 B reads).
//   ROW_MAJOR sticks (1,1,ceil(W/32),32): page t IS gamma[32t .. 32t+31]: one 64 B read of the stick into face 0
//     (rows 0..1 for bf16) + a local copy of its upper half into face 1 row 0 (bf16), or two 64 B reads straight into
//     face 0 / face 1 row 0 (fp32).
// exchange_partials_block (symmetric, k <= K_SYM_MAX): every member writes its rows' column-0 partials (faces 0/2)
//   into slot (r*k + m) of EVERY member's cb_gathered_sumsq (lockstep slot base = this core's own write pointer),
//   flushes, bumps each member's sem_partials by 1; then waits for k*(blk+1) arrivals and publishes the slots.
// store_block: per tile-row, in units of <= out_unit tiles (clamped to the cb_out ring end): wait, write, flush, pop.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "all_gather_rms_norm_common.hpp"

using namespace agrms_cb;

constexpr uint32_t out_page_bytes = get_compile_time_arg_val(0);
constexpr uint32_t out_w_tiles = get_compile_time_arg_val(1);
constexpr uint32_t w_split_cores = get_compile_time_arg_val(2);  // k
constexpr uint32_t out_unit = get_compile_time_arg_val(3);
constexpr uint32_t out_cb_pages = get_compile_time_arg_val(4);
constexpr uint32_t sem_partials_id = get_compile_time_arg_val(5);
constexpr bool partial_faces_only = get_compile_time_arg_val(6) != 0;
constexpr bool gamma_is_tile = get_compile_time_arg_val(7) != 0;
constexpr uint32_t gamma_elem_bytes = get_compile_time_arg_val(8);
constexpr uint32_t gamma_l1_page_bytes = get_compile_time_arg_val(9);  // compact cb_gamma page stride
constexpr uint32_t gamma_tail_pages = get_compile_time_arg_val(10);
constexpr uint32_t gamma_dram_page_bytes = get_compile_time_arg_val(11);
constexpr bool regime_n = get_compile_time_arg_val(12) != 0;
constexpr uint32_t block_row_tiles = get_compile_time_arg_val(13);
constexpr uint32_t n_groups = get_compile_time_arg_val(14);
constexpr auto out_args = TensorAccessorArgs<15>();
[[maybe_unused]] constexpr auto gamma_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();

constexpr uint32_t FACE_DIM = 16;
constexpr uint32_t GAMMA_READ_ALIGN = 64;  // BH DRAM read alignment (>= one 16-datum face row)

#ifdef AGRMS_HAS_GAMMA
template <typename Accessor>
FORCE_INLINE void load_gamma_slice(const Accessor& g_acc, uint32_t g0, uint32_t n) {
    constexpr uint32_t face_bytes = FACE_DIM * FACE_DIM * gamma_elem_bytes;
    constexpr uint32_t half_row_bytes = FACE_DIM * gamma_elem_bytes;  // one 16-datum face row
    cb_reserve_back(cb_gamma, n + gamma_tail_pages);
    const uint32_t base = get_write_ptr(cb_gamma);
    for (uint32_t j = 0; j < n; ++j) {
        const uint32_t page = base + j * gamma_l1_page_bytes;
        if constexpr (gamma_is_tile) {
            noc_async_read(g_acc.get_noc_addr(g0 + j, 0), page, GAMMA_READ_ALIGN);
            noc_async_read(g_acc.get_noc_addr(g0 + j, face_bytes), page + face_bytes, GAMMA_READ_ALIGN);
        } else if constexpr (half_row_bytes % GAMMA_READ_ALIGN == 0) {
            // fp32: each half of the stick is itself 64 B-aligned in DRAM: straight into face 0 / face 1 row 0
            noc_async_read(g_acc.get_noc_addr(g0 + j, 0), page, half_row_bytes);
            noc_async_read(g_acc.get_noc_addr(g0 + j, half_row_bytes), page + face_bytes, half_row_bytes);
        } else {
            // bf16: the 64 B stick lands in face 0 rows 0..1; row 1 (the upper half) is moved to face 1 below
            noc_async_read(g_acc.get_noc_addr(g0 + j, 0), page, 2 * half_row_bytes);
        }
    }
    noc_async_read_barrier();
    if constexpr (!gamma_is_tile && half_row_bytes % GAMMA_READ_ALIGN != 0) {
        for (uint32_t j = 0; j < n; ++j) {
            const uint32_t page = base + j * gamma_l1_page_bytes;
            noc_async_read(get_noc_addr(page + half_row_bytes), page + face_bytes, half_row_bytes);
        }
        noc_async_read_barrier();
    }
    cb_push_back(cb_gamma, n + gamma_tail_pages);
}
#endif

FORCE_INLINE uint64_t peer_addr(uint32_t peers_idx, uint32_t m, uint32_t l1_addr) {
    const uint32_t xy = get_arg_val<uint32_t>(peers_idx + m);
    return get_noc_addr(xy & 0xFFFF, xy >> 16, l1_addr);
}

// One partial tile to a peer: faces 0 and 2 (column 0) or the whole tile.
FORCE_INLINE void send_acc_tile(uint32_t src, uint64_t dst, uint32_t tile_bytes) {
    if constexpr (partial_faces_only) {
        const uint32_t face_bytes = tile_bytes / 4;
        noc_async_write(src, dst, face_bytes);
        noc_async_write(src + 2 * face_bytes, dst + 2 * face_bytes, face_bytes);
    } else {
        noc_async_write(src, dst, tile_bytes);
    }
}

void kernel_main() {
    uint32_t a = 0;
    const uint32_t out_addr = get_arg_val<uint32_t>(a++);
    const uint32_t grp = get_arg_val<uint32_t>(a++);
    const uint32_t n_rows = get_arg_val<uint32_t>(a++);
    const uint32_t w_tile_start = get_arg_val<uint32_t>(a++);
    const uint32_t core_w = get_arg_val<uint32_t>(a++);
    const uint32_t member_idx = get_arg_val<uint32_t>(a++);
    [[maybe_unused]] const uint32_t gamma_addr = get_arg_val<uint32_t>(a++);
    const uint32_t peers_idx = a;  // k packed (x | y << 16) NoC coords of the group's members

    // output tiles this member writes per row: regime A its landing (= output) tiles, regime N the dense row
    const uint32_t t_first = regime_n ? 0 : w_tile_start;
    const uint32_t t_count = regime_n ? out_w_tiles : core_w;

#ifdef AGRMS_HAS_GAMMA
    {
        const auto g_acc = TensorAccessor(gamma_args, gamma_addr, gamma_dram_page_bytes);
        load_gamma_slice(g_acc, t_first, t_count);
    }
#endif

    const auto out_acc = TensorAccessor(out_args, out_addr, out_page_bytes);
    [[maybe_unused]] const uint32_t sem_addr = get_semaphore(sem_partials_id);
    [[maybe_unused]] uint32_t partials_expected = 0;
    [[maybe_unused]] const uint32_t acc_tile_bytes = get_tile_size(cb_partial_sumsq);

    uint32_t out_ring_pos = 0;
    for (uint32_t i0 = 0; i0 < n_rows; i0 += block_row_tiles) {
        const uint32_t rows = (n_rows - i0) < block_row_tiles ? (n_rows - i0) : block_row_tiles;

        // ---- exchange_partials_block (symmetric all-gather of the k members' partials) ----
        if constexpr (w_split_cores > 1) {
            constexpr uint32_t k = w_split_cores;
            cb_wait_front(cb_partial_sumsq, rows);
            const uint32_t src = get_read_ptr(cb_partial_sumsq);
            cb_reserve_back(cb_gathered_sumsq, k * rows);
            const uint32_t slot_base = get_write_ptr(cb_gathered_sumsq);
            for (uint32_t m = 0; m < k; ++m) {
                for (uint32_t r = 0; r < rows; ++r) {
                    send_acc_tile(
                        src + r * acc_tile_bytes,
                        peer_addr(peers_idx, m, slot_base + (r * k + member_idx) * acc_tile_bytes),
                        acc_tile_bytes);
                }
            }
            // same NoC, VC and destination: the increments arrive after the data (flush, not barrier)
            noc_async_writes_flushed();
            for (uint32_t m = 0; m < k; ++m) {
                noc_semaphore_inc(peer_addr(peers_idx, m, sem_addr), 1);
            }
            cb_pop_front(cb_partial_sumsq, rows);
            partials_expected += k;
            noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_addr), partials_expected);
            cb_push_back(cb_gathered_sumsq, k * rows);
        }

        // ---- store_block ----
        for (uint32_t i = i0; i < i0 + rows; ++i) {
            const uint32_t r = grp + i * n_groups;
            const uint32_t first_page = r * out_w_tiles + t_first;
            for (uint32_t j0 = 0; j0 < t_count;) {
                uint32_t n = (t_count - j0) < out_unit ? (t_count - j0) : out_unit;
                n = (out_cb_pages - out_ring_pos) < n ? (out_cb_pages - out_ring_pos) : n;
                cb_wait_front(cb_out_tiles, n);
                const uint32_t l1 = get_read_ptr(cb_out_tiles);
                for (uint32_t j = 0; j < n; ++j) {
                    noc_async_write(l1 + j * out_page_bytes, out_acc.get_noc_addr(first_page + j0 + j), out_page_bytes);
                }
                noc_async_writes_flushed();
                cb_pop_front(cb_out_tiles, n);
                j0 += n;
                out_ring_pos = (out_ring_pos + n == out_cb_pages) ? 0 : out_ring_pos + n;
            }
        }
    }
    noc_async_write_barrier();
    noc_async_atomic_barrier();
}
