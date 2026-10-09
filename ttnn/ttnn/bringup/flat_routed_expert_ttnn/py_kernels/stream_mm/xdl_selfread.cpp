// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// x-download probe, self-reading relay (BRISC, NOC0): reads all of x from DRAM itself, chunk c from region c % REGIONS
// (the x tensor is width-sharded over the banks, REGIONS / banks regions per bank) so consecutive chunks hit different
// banks, BATCH chunks per read barrier, into the local CB that its NCRISC multicasts from (xdl_mcpush.cpp).
// CT: 0 CB, 1 CHUNK_BYTES, 2 TOTAL_CHUNKS, 3 REGIONS, 4 BANKS, 5 BATCH, 6 CHUNK_TILES, 7 STRIDE (this relay takes
//     chunks OFFSET, OFFSET + STRIDE, ...)
// RT: 0 x bank base, 1 region bytes, 2 OFFSET
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb = get_compile_time_arg_val(0);
    constexpr uint32_t bytes = get_compile_time_arg_val(1);
    constexpr uint32_t total = get_compile_time_arg_val(2);
    constexpr uint32_t regions = get_compile_time_arg_val(3);
    constexpr uint32_t banks = get_compile_time_arg_val(4);
    constexpr uint32_t batch = get_compile_time_arg_val(5);
    constexpr uint32_t chunk_tiles = get_compile_time_arg_val(6);
    constexpr uint32_t stride = get_compile_time_arg_val(7);
    constexpr uint32_t mine = total / stride;
    const uint32_t base = get_arg_val<uint32_t>(0), region_bytes = get_arg_val<uint32_t>(1),
                   offset = get_arg_val<uint32_t>(2);
    for (uint32_t c = 0; c < mine; c += batch) {
        cb_reserve_back(cb, batch * chunk_tiles);
        const uint32_t l1 = get_write_ptr(cb);
        for (uint32_t i = 0; i < batch; ++i) {
            const uint32_t k = (c + i) * stride + offset, reg = k % regions;
            noc_async_read(
                get_noc_addr_from_bank_id<true>(
                    reg % banks, base + (reg / banks) * region_bytes + (k / regions) * bytes),
                l1 + i * bytes,
                bytes);
        }
        noc_async_read_barrier();
        cb_push_back(cb, batch * chunk_tiles);
    }
}
