// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/**
 * Writer for Sequential Eltwise Test
 *
 * Writes output data to interleaved DRAM buffer.
 * Uses InterleavedAddrGen to properly handle interleaved DRAM buffers.
 */

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t num_tiles = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_id_out = 16;  // output CB

    uint32_t tile_size_bytes = get_tile_size(cb_id_out);

    // Create interleaved address generator for output buffer
    const InterleavedAddrGen<true> dst_addrgen = {
        .bank_base_address = dst_addr,
        .page_size = tile_size_bytes,
    };

    for (uint32_t i = 0; i < num_tiles; ++i) {
        cb_wait_front(cb_id_out, 1);

        uint32_t l1_read_addr = get_read_ptr(cb_id_out);
        uint64_t dst_noc_addr = dst_addrgen.get_noc_addr(i);

        noc_async_write(l1_read_addr, dst_noc_addr, tile_size_bytes);
        noc_async_write_barrier();

        cb_pop_front(cb_id_out, 1);
    }
}
