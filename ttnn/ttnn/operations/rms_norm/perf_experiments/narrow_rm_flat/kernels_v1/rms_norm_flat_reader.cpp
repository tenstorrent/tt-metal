// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// rms_norm flat reader (NCRISC / NoC0) — ROW_MAJOR WIDTH / BLOCK_SHARDED shards of width w_s | 16 datums.
//
// The shard (N sticks x w_s datums, sticks back to back) is read by compute as F_full "flat tiles" of 1024
// consecutive datums straight from the input shard (cb_x is backed on it, tile-sized pages): the reader only
// publishes those pages. Then, in the order compute needs them:
//   1. cb_mm 0/1 constants (rd_const): FLAT_CMP — S_j (j < R): S_j[i][c] = 1 iff c == j * G + i / w_s, i % w_s < v
//      (G = 32 / w_s stick groups per logical row, v = this member's valid columns) and E_j = S_j^T without the
//      mask; broadcast form — B[i][c] = 1 iff i / w_s == c / w_s, i % w_s < v. Zeroed by the NoC zero engine,
//      ones by RISC-V stores (<= 32 * R or 32 * w_s datums).
//   2. the last partial flat tile (rd_tail): the shard bytes past F_full tiles copied into cb_xtail (one local
//      NoC read; the caller's shard is never written) and zero-filled to the tile end (reading the shard-backed
//      CB past the shard end could leave L1).
//   3. gamma_flat (rd_gamma): ONE 64 B-aligned DRAM read of the window holding this member's w_s gamma datums
//      (BH DRAM reads are 64 B aligned; the slice is w_s * g_elem <= 64 B and aligned to its own size, so it never
//      straddles a window), pad columns zeroed, then repeated 1024 / w_s times by RISC-V word stores.
//
// Raw dataflow API (documented helper bypass): no kernel_lib helper builds a 0/1 selection matrix or repeats a
// sub-row pattern over a tile; the per-datum stores are the whole job.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
#include "rms_norm_flat_common.hpp"

using namespace rms_norm_flat_cb;

constexpr uint32_t F_full = get_compile_time_arg_val(0);
constexpr uint32_t tail_bytes = get_compile_time_arg_val(1);
constexpr uint32_t x_tile_bytes = get_compile_time_arg_val(2);
constexpr uint32_t w_s = get_compile_time_arg_val(3);
constexpr uint32_t elem_bytes = get_compile_time_arg_val(4);
constexpr uint32_t R = get_compile_time_arg_val(5);
constexpr bool acc_fp32 = get_compile_time_arg_val(6) != 0;
constexpr uint32_t g_elem_bytes = get_compile_time_arg_val(7);
constexpr uint32_t g_page_bytes = get_compile_time_arg_val(8);
constexpr uint32_t n_mm = get_compile_time_arg_val(9);
[[maybe_unused]] constexpr auto g_args = TensorAccessorArgs<10>();

constexpr uint32_t G = 32 / w_s;  // stick groups per logical tile row
constexpr uint32_t acc_bytes = acc_fp32 ? 4 : 2;
constexpr uint32_t acc_tile_bytes = 1024 * acc_bytes;
static_assert(16 % w_s == 0, "flat regime: w_s must divide a 16-datum face row");

// Datum index of logical (r, c) inside a 32x32 tile (faces row-major, 16x16 each).
FORCE_INLINE uint32_t tile_pos(uint32_t r, uint32_t c) {
    return ((r >> 4) * 2 + (c >> 4)) * 256 + (r & 15) * 16 + (c & 15);
}

FORCE_INLINE void set_one(uint32_t tile_base, uint32_t r, uint32_t c) {
    const uint32_t p = tile_pos(r, c);
    if constexpr (acc_fp32) {
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tile_base)[p] = 0x3F800000u;
    } else {
        reinterpret_cast<volatile tt_l1_ptr uint16_t*>(tile_base)[p] = 0x3F80u;
    }
}

void kernel_main() {
    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t valid_cols = get_arg_val<uint32_t>(1);  // valid columns of this member's slice (<= w_s)
    [[maybe_unused]] const uint32_t col0 = get_arg_val<uint32_t>(2);
    [[maybe_unused]] const uint32_t g_addr = get_arg_val<uint32_t>(3);

    // ---- full flat tiles: cb_x IS the shard ----
    if constexpr (F_full > 0) {
        cb_reserve_back(cb_x, F_full);
        cb_push_back(cb_x, F_full);
    }

    Noc noc;
    // ---- 0/1 constants ----
    {
        MaybeDeviceZoneScope("rd_const");
        cb_reserve_back(cb_mm, n_mm);
        CircularBuffer mm(cb_mm);
        noc.async_write_zeros(mm, n_mm * acc_tile_bytes);
        noc.write_zeros_l1_barrier();
        const uint32_t base = get_write_ptr(cb_mm);
#ifdef FLAT_CMP
        for (uint32_t j = 0; j < R; ++j) {
            const uint32_t s_tile = base + j * acc_tile_bytes;
            const uint32_t e_tile = base + (R + j) * acc_tile_bytes;
            for (uint32_t i = 0; i < 32; ++i) {
                const uint32_t c = j * G + i / w_s;
                if (i % w_s < valid_cols) {
                    set_one(s_tile, i, c);
                }
                set_one(e_tile, c, i);
            }
        }
#else
        for (uint32_t i = 0; i < 32; ++i) {
            if (i % w_s < valid_cols) {
                const uint32_t c0 = (i / w_s) * w_s;
                for (uint32_t c = c0; c < c0 + w_s; ++c) {
                    set_one(base, i, c);
                }
            }
        }
#endif
        cb_push_back(cb_mm, n_mm);
    }

    // ---- last partial flat tile: local copy + zero fill ----
    if constexpr (tail_bytes > 0) {
        MaybeDeviceZoneScope("rd_tail");
        cb_reserve_back(cb_xtail, 1);
        const uint32_t l1 = get_write_ptr(cb_xtail);
        noc_async_read(get_noc_addr(x_addr + F_full * x_tile_bytes), l1, tail_bytes);
        CircularBuffer xt(cb_xtail);
        noc.async_write_zeros(xt, x_tile_bytes - tail_bytes, {.offset_bytes = tail_bytes});
        noc.write_zeros_l1_barrier();
        noc_async_read_barrier();
        cb_push_back(cb_xtail, 1);
    }

#ifdef RMS_HAS_GAMMA
    // ---- gamma_flat: this member's w_s gamma datums repeated over one tile ----
    {
        MaybeDeviceZoneScope("rd_gamma");
        constexpr uint32_t slice_bytes = w_s * g_elem_bytes;  // 16..64 B
        constexpr uint32_t slice_words = slice_bytes / 4;
        static_assert(slice_bytes % 4 == 0 && slice_bytes <= 64, "gamma slice must be whole words within 64 B");
        const auto g_acc = TensorAccessor(g_args, g_addr, g_page_bytes);
        cb_reserve_back(cb_gamma, 1);
        const uint32_t base = get_write_ptr(cb_gamma);
        const uint32_t byte0 = col0 * g_elem_bytes;
        const uint32_t lead = byte0 & 63;
        noc_async_read(g_acc.get_noc_addr(0, byte0 - lead), base, 64);
        noc_async_read_barrier();
        uint32_t tmp[slice_words];
        volatile tt_l1_ptr uint32_t* src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base + lead);
        for (uint32_t w = 0; w < slice_words; ++w) {
            tmp[w] = src[w];
        }
        if (valid_cols < w_s) {  // W tail: gamma pad columns = 0
            uint8_t* b = reinterpret_cast<uint8_t*>(tmp);
            for (uint32_t e = valid_cols * g_elem_bytes; e < slice_bytes; ++e) {
                b[e] = 0;
            }
        }
        volatile tt_l1_ptr uint32_t* dst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base);
        constexpr uint32_t reps = 1024 / w_s;
        for (uint32_t rep = 0; rep < reps; ++rep) {
#pragma GCC unroll 16
            for (uint32_t w = 0; w < slice_words; ++w) {
                dst[rep * slice_words + w] = tmp[w];
            }
        }
        cb_push_back(cb_gamma, 1);
    }
#endif
}
