// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// x pre-pass (extract + tilize outside the expert op): writer (NCRISC). The core's tilizer (se11_tz.cpp) hands it
// super-blocks (MT row tiles x 32 K tiles of one virtual expert, row tile major; se11_xrd.cpp read them from the
// row-major dispatch buffer). Each of a super-block's 32 / KBLK x blocks ([MT x KBLK] tiles, row tile major) goes to
// the flat expert's x layout in DRAM: block k (in stream order) in region k % NREG at slot k / NREG, region r in bank
// r % BANKS at byte offset (r / BANKS) * REGION_BYTES.
// CT: 0 SB_CB, 1 MT, 2 TILE_BYTES, 3 KBLK, 4 NREG, 5 BANKS
// RT: 0 x layout bank base, 1 region bytes, 2 super-blocks of this core, 3 STRIDE, 4 OFF (this core's super-blocks
//     are OFF, OFF + STRIDE, ... in stream order)
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t sb_cb = get_compile_time_arg_val(0);
    constexpr uint32_t mt = get_compile_time_arg_val(1);
    constexpr uint32_t tb = get_compile_time_arg_val(2);
    constexpr uint32_t kblk = get_compile_time_arg_val(3);
    constexpr uint32_t nreg = get_compile_time_arg_val(4);
    constexpr uint32_t banks = get_compile_time_arg_val(5);
    constexpr uint32_t per_sb = 32 / kblk;
    constexpr uint32_t blk_bytes = mt * kblk * tb, piece = kblk * tb;
    const uint32_t base = get_arg_val<uint32_t>(0), region_bytes = get_arg_val<uint32_t>(1);
    const uint32_t num_sb = get_arg_val<uint32_t>(2), stride = get_arg_val<uint32_t>(3), off = get_arg_val<uint32_t>(4);
    for (uint32_t b = 0; b < num_sb; ++b) {
        cb_wait_front(sb_cb, mt * 32);
        const uint32_t src = get_read_ptr(sb_cb);
        const uint32_t g = b * stride + off;
        for (uint32_t i = 0; i < per_sb; ++i) {
            const uint32_t k = g * per_sb + i, reg = k % nreg;
            const uint64_t dst = get_noc_addr_from_bank_id<true>(
                reg % banks, base + (reg / banks) * region_bytes + (k / nreg) * blk_bytes);
            for (uint32_t m = 0; m < mt; ++m) {
                noc_async_write(src + (m * 32 + i * kblk) * tb, dst + m * piece, piece);
            }
        }
        noc_async_writes_flushed();
        cb_pop_front(sb_cb, mt * 32);
    }
    noc_async_write_barrier();
}
