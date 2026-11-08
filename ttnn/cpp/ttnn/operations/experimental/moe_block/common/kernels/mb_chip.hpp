// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// One program for the whole mesh (common/chip_program.hpp): a data-movement kernel whose args differ by chip
// (MB_CHIP) reads its core's patch block - this chip's own, at the same DRAM address on every chip - and adds the
// differences to its runtime / common runtime args (the host's values: chip 0's), before anything reads them
// (mb_loop_main.hpp's kernel_main); entries tagged 1 << 30 are the core's compute kernel's, written into each TRISC's
// args, the last one the compute's flag. A compute kernel with values (MB_CHIP_WAIT) waits for that flag (its common
// arg MB_CHIP_FLAG) before anything. MB_CHIP's common args at MB_CHIP_CA: the table's address, a block's bytes, the
// DRAM banks, the kernel's first page, the grid's width; then the landing space. A block: n, then n (index | compute <<
// 30 | common << 31, difference).
#pragma once

#ifdef MB_CHIP
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "internal/tt-1xx/cache.h"

namespace mb_chip {
__attribute__((noinline, optimize("Os"))) inline void apply() {
    constexpr uint32_t a = MB_CHIP_CA;
    const uint32_t addr = get_common_arg_val<uint32_t>(a), block = get_common_arg_val<uint32_t>(a + 1);
    const uint32_t nb = get_common_arg_val<uint32_t>(a + 2);
    const uint32_t page = get_common_arg_val<uint32_t>(a + 3) +
                          get_absolute_logical_y() * get_common_arg_val<uint32_t>(a + 4) + get_absolute_logical_x();
    const uint32_t l1 = (get_common_arg_addr(a + 5) + 63) & ~63u;
    noc_async_read(get_noc_addr_from_bank_id<true>(page % nb, addr + (page / nb) * block), l1, block);
    noc_async_read_barrier();
    invalidate_l1_cache();
    const uint32_t li = *GET_MAILBOX_ADDRESS_DEV(launch_msg_rd_ptr);
    const auto* lm = GET_MAILBOX_ADDRESS_DEV(launch[li]);
    const uint32_t kbase = lm->kernel_config.kernel_config_base[PROGRAMMABLE_CORE_TYPE];
    volatile tt_l1_ptr uint32_t* w = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1);
    for (uint32_t i = 0, n = w[0]; i < n; ++i) {
        const uint32_t ix = w[1 + 2 * i], v = w[2 + 2 * i], idx = ix & 0x3FFFFFFFu;
        const bool common = ix >> 31;
        if ((ix >> 30) & 1) {  // (the compute's: every TRISC's args - once a region, the TRISCs may share one)
            uint32_t done[3];
            for (uint32_t t = 2; t < 5; ++t) {
                const uint32_t off =
                    common ? lm->kernel_config.rta_offset[t].crta_offset : lm->kernel_config.rta_offset[t].rta_offset;
                done[t - 2] = off;
                if ((t > 2 && off == done[0]) || (t > 3 && off == done[1])) {
                    continue;
                }
                *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kbase + off + 4 * idx) += v;
            }
        } else {
            const uint32_t at = common ? get_common_arg_addr(idx) : get_arg_addr(idx);
            *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(at) += v;
        }
    }
}
}  // namespace mb_chip
#endif

#ifdef MB_CHIP_WAIT
#include <cstdint>
#include "internal/tt-1xx/cache.h"

namespace mb_chip {
inline void wait() {  // (every TRISC: the core's data-movement kernel wrote this chip's values into my args)
    volatile tt_l1_ptr uint32_t* f = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(&crta_l1_base[MB_CHIP_FLAG]);
    while (*f == 0) {
        invalidate_l1_cache();
    }
}
}  // namespace mb_chip
#endif
