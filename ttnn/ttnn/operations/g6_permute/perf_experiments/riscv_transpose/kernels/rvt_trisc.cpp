// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// riscv_transpose bake-off: the three TRISCs as transpose helpers of nhwc_rm_transpose. Each TRISC takes
// the row items [lo, hi) of its share (rvt::transpose_rows) of every unit of every block, with plain
// RISC-V loads and stores: no unpacker, FPU or packer touches the data, so every bit pattern survives.
// Per block b: wait until the reader pushed it (tiles_received of cb_input_slots) and the writer freed
// its rows generation (cb_free), transpose, post done = b + 1 on this TRISC's counter (cb_done0 + t).
// Raw (no helper): kernel_lib has no block helper that transposes halfwords between strided L1 rows;
// the compute API's CB calls run on one thread each (wait/pop on UNPACK, reserve/push on PACK), so the
// three threads sync on the stream counters directly (rvt_sync.hpp).
#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/cb_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "g6_permute_common.hpp"
#include "rvt_transpose.hpp"
#include "rvt_sync.hpp"

constexpr uint32_t cb_input_slots = get_compile_time_arg_val(0);
constexpr uint32_t cb_output_rows = get_compile_time_arg_val(1);
constexpr uint32_t cb_free = get_compile_time_arg_val(2);
constexpr uint32_t cb_done0 = get_compile_time_arg_val(3);
constexpr uint32_t block_units = get_compile_time_arg_val(4);
constexpr uint32_t units_per_row = get_compile_time_arg_val(5);
constexpr uint32_t block_w = get_compile_time_arg_val(6);
constexpr uint32_t tail_w = get_compile_time_arg_val(7);
constexpr uint32_t tensor_c = get_compile_time_arg_val(8);
constexpr uint32_t slot_bytes = get_compile_time_arg_val(9);
constexpr uint32_t row_bytes = get_compile_time_arg_val(10);
constexpr uint32_t input_depth = get_compile_time_arg_val(11);
constexpr uint32_t rows_depth = get_compile_time_arg_val(12);
// Row-item bounds of TRISC 0, 1, 2: [lo_t, hi_t).
constexpr uint32_t lo_0 = get_compile_time_arg_val(13);
constexpr uint32_t hi_0 = get_compile_time_arg_val(14);
constexpr uint32_t lo_1 = get_compile_time_arg_val(15);
constexpr uint32_t hi_1 = get_compile_time_arg_val(16);
constexpr uint32_t lo_2 = get_compile_time_arg_val(17);
constexpr uint32_t hi_2 = get_compile_time_arg_val(18);

#if defined(TRISC_UNPACK)
constexpr uint32_t trisc = 0;
constexpr uint32_t lo = lo_0, hi = hi_0;
#elif defined(TRISC_MATH)
constexpr uint32_t trisc = 1;
constexpr uint32_t lo = lo_1, hi = hi_1;
#else
constexpr uint32_t trisc = 2;
constexpr uint32_t lo = lo_2, hi = hi_2;
#endif

constexpr uint32_t unit_row_bytes = tensor_c * row_bytes;
constexpr uint32_t block_slot_bytes = block_units * block_w * slot_bytes;
constexpr uint32_t block_row_bytes = block_units * unit_row_bytes;
constexpr uint32_t PB = 16;

#ifdef RVT_STREAM
// Streamed: one post per row item (a running count), so the writer sends rows as they complete.
uint32_t items_posted = 0;
template <uint32_t pages>
FORCE_INLINE void transpose_unit(uint32_t src, uint32_t dst) {
    for (uint32_t m = lo; m < hi; ++m) {
        rvt::transpose_rows<pages, tensor_c, slot_bytes, row_bytes, PB>(src, dst, m, m + 1);
        rvt::ctr_post(cb_done0 + trisc, ++items_posted);
    }
}
#else
template <uint32_t pages>
FORCE_INLINE void transpose_unit(uint32_t src, uint32_t dst) {
    rvt::transpose_rows<pages, tensor_c, slot_bytes, row_bytes, PB>(src, dst, lo, hi);
}
#endif

FORCE_INLINE void transpose_block(uint32_t src, uint32_t dst, uint32_t& wb, uint32_t extent) {
    asm volatile("" ::: "memory");
    for (uint32_t u = 0; u < extent; ++u) {
        if (tail_w != block_w && wb == units_per_row - 1) {
            transpose_unit<tail_w>(src, dst);
            src += tail_w * slot_bytes;
        } else {
            transpose_unit<block_w>(src, dst);
            src += block_w * slot_bytes;
        }
        dst += unit_row_bytes;
        if (++wb == units_per_row) {
            wb = 0;
        }
    }
    asm volatile("" ::: "memory");
}

template <bool zoned>
FORCE_INLINE void handle_block(uint32_t slots_base, uint32_t rows_base, uint32_t& wb, uint32_t extent, uint32_t b) {
    {
        G6_ZONED_STEP(zoned, "RVT-TR-WAIT", {
            rvt::ctr_wait(cb_input_slots, (b + 1) * block_units);
            if (b >= rows_depth) {
                rvt::ctr_wait(cb_free, b + 1 - rows_depth);
            }
        });
    }
    rvt::l1_invalidate();
    if constexpr (lo < hi) {
        G6_ZONED_STEP(
            zoned,
            "RVT-TR-XPOSE",
            transpose_block(
                slots_base + (b % input_depth) * block_slot_bytes,
                rows_base + (b % rows_depth) * block_row_bytes,
                wb,
                extent));
    }
#ifndef RVT_STREAM
    rvt::ctr_post(cb_done0 + trisc, b + 1);
#endif
}

void kernel_main() {
    uint32_t remaining = get_arg_val<uint32_t>(0);
    uint32_t wb = get_arg_val<uint32_t>(1);
    const uint32_t slots_base = ckernel::get_tile_address(cb_input_slots, 0);
    const uint32_t rows_base = ckernel::get_tile_address(cb_output_rows, 0);

    uint32_t b = 0;
    for (; b < g6_permute::ZONED_BLOCKS && remaining > 0; ++b) {
        const uint32_t extent = remaining < block_units ? remaining : block_units;
        handle_block<true>(slots_base, rows_base, wb, extent, b);
        remaining -= extent;
    }
    for (; remaining > 0; ++b) {
        const uint32_t extent = remaining < block_units ? remaining : block_units;
        handle_block<false>(slots_base, rows_base, wb, extent, b);
        remaining -= extent;
    }
}
