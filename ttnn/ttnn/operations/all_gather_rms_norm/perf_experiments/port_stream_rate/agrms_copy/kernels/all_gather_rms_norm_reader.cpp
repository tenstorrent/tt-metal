// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// all_gather_rms_norm member reader (NCRISC / NoC0). Block operations (op_design.md -> Block schedule):
//   load_local_blocks    once: this member's columns of the chip's OWN slice, for every local row, DRAM ->
//                        cb_x_landing slots (one barrier). The ports re-send them from here (no second DRAM read).
//   signal_local_loaded  once: +1 on sem_local_loaded of every port of this chip.
//   per block:
//   await_landing_block  per incoming stream s: sem_landed[s] >= (last local row + 1) * n_s (n_s = 1 if stream s
//                        touches this member at all, else 0); regime N: zero the W padding of every slice's tail tile;
//                        then one cb_push_back of the block's landing pages (the credit compute waits on).
//   compose_aligned_block (regime N) dense output tile t of a row <- datums of the landing tiles (slice x / W_local,
//                        column x % W_local), zeros for x >= W; streamed in out_unit units into cb_x_aligned.
// cb_x_landing is never reserved for writing: its bytes arrive by DMA (this kernel's DRAM reads and the peers'
// fabric writes); the CB credit only publishes them to compute. The ring holds every block and never wraps.
// Native HEIGHT shards (own_in_shard, Refinement 1): the own slice is NOT loaded — it is the resident input shard,
// published once on cb_x_in (zero-copy CB backed on the shard; compute and the ports read it in place). Regime N
// zeroes the W padding of the own tail tile there (the shard's tile padding, never logical data) and composes the
// own slice's columns straight from it.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "all_gather_rms_norm_common.hpp"

using namespace agrms_cb;

constexpr uint32_t page_bytes = get_compile_time_arg_val(0);
constexpr uint32_t Wt_lp = get_compile_time_arg_val(1);
constexpr uint32_t w_pitch = get_compile_time_arg_val(2);
constexpr uint32_t n_groups = get_compile_time_arg_val(3);
constexpr uint32_t block_row_tiles = get_compile_time_arg_val(4);
constexpr bool regime_n = get_compile_time_arg_val(5) != 0;
constexpr uint32_t W_local = get_compile_time_arg_val(6);
constexpr uint32_t G = get_compile_time_arg_val(7);
constexpr uint32_t out_w_tiles = get_compile_time_arg_val(8);
constexpr uint32_t out_unit = get_compile_time_arg_val(9);
constexpr uint32_t elem_bytes = get_compile_time_arg_val(10);
constexpr uint32_t num_streams = get_compile_time_arg_val(11);
constexpr uint32_t sem_local_loaded_id = get_compile_time_arg_val(12);
constexpr uint32_t aligned_cb_pages = get_compile_time_arg_val(13);  // regime N: cb_x_aligned capacity
constexpr bool own_in_shard = get_compile_time_arg_val(14) != 0;     // native HEIGHT: own slice = input shard
constexpr bool stage = get_compile_time_arg_val(15) != 0;            // regime N bfp8: compose from the bf16 staging row
constexpr auto x_args = TensorAccessorArgs<16>();

constexpr uint32_t TILE_DIM = 32;
constexpr uint32_t FACE_DIM = 16;
constexpr uint32_t FACE_DATUMS = FACE_DIM * FACE_DIM;
constexpr uint32_t W = G * W_local;
constexpr uint32_t w_valid_tail = W_local - TILE_DIM * (Wt_lp - 1);
// bfloat8_b tiles: 64 shared-exponent bytes (one per face row), then one mantissa byte per datum in face order.
// Zeroing mantissa bytes yields exact zeros whatever the shared exponent, so the tail zero-fill is in place.
constexpr uint32_t land_hdr = elem_bytes == 1 ? 64 : 0;
// The compose works on 2/4-byte datums: the input-dtype landing tiles, or (stage) the bf16 staging row.
constexpr uint32_t cmp_elem = stage ? 2 : elem_bytes;
constexpr uint32_t cmp_page = stage ? TILE_DIM * TILE_DIM * 2 : page_bytes;
static_assert(!regime_n || cmp_elem != 1, "regime N on bfloat8_b composes from the bf16 staging row");

// datum offset (in datums) of (row, col) inside a tile: faces 0/1 (rows < 16), 2/3, each 16 x 16 row-major
FORCE_INLINE uint32_t datum_ofs(uint32_t row, uint32_t col) {
    return ((row >= FACE_DIM ? 2u : 0u) + (col >= FACE_DIM ? 1u : 0u)) * FACE_DATUMS + (row % FACE_DIM) * FACE_DIM +
           (col % FACE_DIM);
}

// zero columns [c0, 32) of every row of the tile at `tile` (datums of `eb` bytes after a `hdr`-byte header)
template <uint32_t eb = elem_bytes, uint32_t hdr = land_hdr>
FORCE_INLINE void zero_cols(uint32_t tile, uint32_t c0) {
    for (uint32_t row = 0; row < TILE_DIM; ++row) {
        for (uint32_t c = c0; c < TILE_DIM;) {
            const uint32_t n = FACE_DIM - (c % FACE_DIM);
            dataflow_kernel_lib::fill_l1_range<eb>(tile + hdr + datum_ofs(row, c) * eb, n * eb, 0);
            c += n;
        }
    }
}

template <typename T>
FORCE_INLINE void copy_run(uint32_t dst, uint32_t src, uint32_t n) {
    volatile tt_l1_ptr T* d = reinterpret_cast<volatile tt_l1_ptr T*>(dst);
    volatile tt_l1_ptr T* s = reinterpret_cast<volatile tt_l1_ptr T*>(src);
    for (uint32_t i = 0; i < n; ++i) {
        d[i] = s[i];
    }
}

// Regime N: dense output tile t of local row li, composed from the landing tiles of that row (own slice p_self
// from the input shard row xin_row when own_in_shard). stage: landing_row is the bf16 staging row instead (it holds
// every slice, own included), xin_row unused.
FORCE_INLINE void compose_tile(uint32_t dst_tile, uint32_t landing_row, uint32_t xin_row, uint32_t p_self, uint32_t t) {
    uint32_t x = t * TILE_DIM;
    const uint32_t x_end = x + TILE_DIM;
    while (x < x_end) {
        const uint32_t xo = x - t * TILE_DIM;
        if (x >= W) {
            zero_cols<cmp_elem, 0>(dst_tile, xo);
            return;
        }
        const uint32_t q = x / W_local;
        const uint32_t c = x - q * W_local;
        const uint32_t cc = c % TILE_DIM;
        const uint32_t src_tile = (own_in_shard && !stage && q == p_self)
                                      ? xin_row + (c / TILE_DIM) * page_bytes
                                      : landing_row + (q * Wt_lp + c / TILE_DIM) * cmp_page;
        uint32_t len = TILE_DIM - xo;
        len = (FACE_DIM - xo % FACE_DIM) < len ? (FACE_DIM - xo % FACE_DIM) : len;
        len = (W_local - c) < len ? (W_local - c) : len;
        len = (FACE_DIM - cc % FACE_DIM) < len ? (FACE_DIM - cc % FACE_DIM) : len;
        for (uint32_t row = 0; row < TILE_DIM; ++row) {
            const uint32_t d = dst_tile + datum_ofs(row, xo) * cmp_elem;
            const uint32_t s = src_tile + datum_ofs(row, cc) * cmp_elem;
            if constexpr (cmp_elem == 4) {
                copy_run<uint32_t>(d, s, len);
            } else {
                copy_run<uint16_t>(d, s, len);
            }
        }
        x += len;
    }
}

void kernel_main() {
#ifdef AGRMS_ABLATE_EMPTY
    return;
#endif
    uint32_t a = 0;
    const uint32_t x_addr = get_arg_val<uint32_t>(a++);
    const uint32_t grp = get_arg_val<uint32_t>(a++);
    const uint32_t n_rows = get_arg_val<uint32_t>(a++);
    const uint32_t w_tile_start = get_arg_val<uint32_t>(a++);
    const uint32_t core_w = get_arg_val<uint32_t>(a++);
    const uint32_t p_self = get_arg_val<uint32_t>(a++);
    const uint32_t n_ports = get_arg_val<uint32_t>(a++);
    const uint32_t ports_idx = a;
    a += n_ports;
    const uint32_t streams_idx = a;  // n_s for s in [0, num_streams)

    const uint32_t landing = get_write_ptr(cb_x_landing);
    const auto x_acc = TensorAccessor(x_args, x_addr, page_bytes);

    // ---- load_local_blocks + signal_local_loaded ----
    const uint32_t s_lo = p_self * Wt_lp, s_hi = s_lo + Wt_lp;
    const uint32_t own_lo = w_tile_start > s_lo ? w_tile_start : s_lo;
    const uint32_t own_hi = (w_tile_start + core_w) < s_hi ? (w_tile_start + core_w) : s_hi;
    [[maybe_unused]] uint32_t xin = 0;
    if constexpr (own_in_shard) {
        cb_reserve_back(cb_x_in, n_rows * Wt_lp);
        xin = get_write_ptr(cb_x_in);
        if constexpr (regime_n && w_valid_tail < TILE_DIM) {
            for (uint32_t i = 0; i < n_rows; ++i) {
                zero_cols(xin + (i * Wt_lp + Wt_lp - 1) * page_bytes, w_valid_tail);
            }
        }
        cb_push_back(cb_x_in, n_rows * Wt_lp);
    }
    if (own_lo < own_hi) {
        AGRMS_ZONE("AG_OWN_LOAD");
        for (uint32_t i = 0; i < n_rows && !own_in_shard; ++i) {
            const uint32_t r = grp + i * n_groups;
            uint32_t dst = landing + (i * w_pitch + (own_lo - w_tile_start)) * page_bytes;
            for (uint32_t t = own_lo; t < own_hi; ++t, dst += page_bytes) {
                noc_async_read(x_acc.get_noc_addr(r * Wt_lp + (t - s_lo)), dst, page_bytes);
            }
        }
        if constexpr (!own_in_shard) {
            noc_async_read_barrier();
        }
        const uint32_t loaded_addr = get_semaphore(sem_local_loaded_id);
        for (uint32_t pi = 0; pi < n_ports; ++pi) {
            const uint32_t xy = get_arg_val<uint32_t>(ports_idx + pi);
            noc_semaphore_inc(get_noc_addr(xy & 0xFFFF, xy >> 16, loaded_addr), 1);
        }
    }

    // 1.0 scaler of compute's FPU row collapse
#ifndef AGRMS_ABLATE_TRANSPORT
    dataflow_kernel_lib::
        calculate_and_prepare_reduce_scaler<cb_ones, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>();
#endif

    [[maybe_unused]] uint32_t aligned_pos = 0;  // cb_x_aligned page position of the write pointer
    for (uint32_t i0 = 0; i0 < n_rows; i0 += block_row_tiles) {
        const uint32_t rows = (n_rows - i0) < block_row_tiles ? (n_rows - i0) : block_row_tiles;
        // ---- await_landing_block ----
        {
            AGRMS_ZONE("AG_LAND_WAIT");  // stall on the transport (occupancy, not cost)
            for (uint32_t s = 0; s < num_streams; ++s) {
                const uint32_t n_s = get_arg_val<uint32_t>(streams_idx + s);
#ifdef AGRMS_ABLATE_LOCAL
                continue;  // perf probe: no transport
#endif
                if (n_s != 0) {
                    // stream s notifies on semaphore id s
                    noc_semaphore_wait_min(
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(s)), (i0 + rows) * n_s);
                }
            }
        }
        if constexpr (regime_n) {
            // W padding of every slice's tail tile must not enter sum(x^2) nor the compose
            if constexpr (w_valid_tail < TILE_DIM) {
                for (uint32_t i = i0; i < i0 + rows; ++i) {
                    for (uint32_t q = 0; q < G; ++q) {
                        if (own_in_shard && q == p_self) {
                            continue;  // zeroed in the input shard above
                        }
                        zero_cols(landing + (i * w_pitch + q * Wt_lp + Wt_lp - 1) * page_bytes, w_valid_tail);
                    }
                }
            }
        }
#ifdef AGRMS_ABLATE_TRANSPORT
        continue;  // perf probe: nothing downstream of the landing
#endif
        cb_reserve_back(cb_x_landing, rows * w_pitch);
        cb_push_back(cb_x_landing, rows * w_pitch);

        // ---- compose_aligned_block (regime N) ----
        if constexpr (regime_n) {
            for (uint32_t i = i0; i < i0 + rows; ++i) {
                uint32_t landing_row = landing + i * w_pitch * page_bytes;
                const uint32_t xin_row = xin + i * Wt_lp * page_bytes;
                if constexpr (stage) {
                    // compute unpacked this row's landing tiles (own slice included) into one bf16 staging row
                    cb_wait_front(cb_x_stage, w_pitch);
                    landing_row = get_read_ptr(cb_x_stage);
                }
                for (uint32_t t0 = 0; t0 < out_w_tiles;) {
                    // a unit is written linearly from the write pointer: clamp it to the ring end
                    uint32_t n = (out_w_tiles - t0) < out_unit ? (out_w_tiles - t0) : out_unit;
                    n = (aligned_cb_pages - aligned_pos) < n ? (aligned_cb_pages - aligned_pos) : n;
                    cb_reserve_back(cb_x_aligned, n);
                    const uint32_t wptr = get_write_ptr(cb_x_aligned);
                    for (uint32_t j = 0; j < n; ++j) {
                        compose_tile(wptr + j * cmp_page, landing_row, xin_row, p_self, t0 + j);
                    }
                    cb_push_back(cb_x_aligned, n);
                    aligned_pos = (aligned_pos + n == aligned_cb_pages) ? 0 : aligned_pos + n;
                    t0 += n;
                }
                if constexpr (stage) {
                    cb_pop_front(cb_x_stage, w_pitch);
                }
            }
        }
    }
    noc_async_atomic_barrier();
}
