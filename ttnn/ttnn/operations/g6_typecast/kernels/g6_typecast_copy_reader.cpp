// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// g6_typecast R0 `pure_copy`, page mode (pages >= cores), reader half (NCRISC / NoC0): batches of whole
// pages (g6_typecast_copy_units.hpp) into the slots of the shared 2-slot CB. Writes stay on NoC1 (writer):
// keeping them off the read NoC measured 2814 vs 4452 ns on [2,4,64,256] RM (Perf 1).
// Raw NoC calls bypass no helper (no kernel_lib helper for an un-transformed page copy).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "g6_typecast_copy_units.hpp"
#include "g6_typecast_zones.hpp"

void kernel_main() {
    constexpr uint32_t page_bytes = get_compile_time_arg_val(0);
    constexpr uint32_t cb = get_compile_time_arg_val(1);
    constexpr uint32_t buf_bytes = get_compile_time_arg_val(2);
    constexpr auto src_args = TensorAccessorArgs<3>();

    const uint32_t src_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t page0 = get_arg_val<uint32_t>(0);
    const uint32_t count = get_arg_val<uint32_t>(1);
    const uint32_t off = get_arg_val<uint32_t>(2);
    const uint32_t len = get_arg_val<uint32_t>(3);

    const auto src = TensorAccessor(src_args, src_addr, page_bytes);
    g6_copy::for_each_batch(page0, count, off, len, buf_bytes, [&](uint32_t p, uint32_t n, uint32_t o, uint32_t b) {
        {
            G6_ZONE("copy_wait_space");
            cb_reserve_back(cb, 1);
        }
        G6_ZONE("copy_read");
        const uint32_t l1 = get_write_ptr(cb);
        for (uint32_t i = 0; i < n; ++i) {
            noc_async_read(src.get_noc_addr(p + i, o), l1 + i * b, b);
        }
        noc_async_read_barrier();
        cb_push_back(cb, 1);
    });
}
