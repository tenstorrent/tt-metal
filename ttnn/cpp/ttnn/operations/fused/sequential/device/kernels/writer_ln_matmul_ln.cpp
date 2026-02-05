// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/**
 * Writer for Sequential LN -> Matmul -> LN Test
 *
 * Writes the final output [1, Wt] tiles to DRAM.
 */

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t output_addr = get_arg_val<uint32_t>(0);
    uint32_t Wt = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_out = 16;  // output CB

    uint32_t tile_size_bytes = get_tile_size(cb_out);

    // Create interleaved address generator for output buffer
    const InterleavedAddrGen<true> output_addrgen = {
        .bank_base_address = output_addr,
        .page_size = tile_size_bytes,
    };

    for (uint32_t w = 0; w < Wt; w++) {
        cb_wait_front(cb_out, 1);

        uint32_t l1_read_addr = get_read_ptr(cb_out);
        uint64_t output_noc_addr = output_addrgen.get_noc_addr(w);

        noc_async_write(l1_read_addr, output_noc_addr, tile_size_bytes);
        noc_async_write_barrier();

        cb_pop_front(cb_out, 1);
    }
}
