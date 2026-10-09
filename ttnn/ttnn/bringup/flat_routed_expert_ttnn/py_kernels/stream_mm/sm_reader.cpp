// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Streamed-matmul weight reader (BRISC, NOC0) on a reader core beside its DRAM bank. Streams this reader's
// contiguous region of the bank -- chunks in consumption order, one chunk = one K-block for each of its R
// receivers -- into the local chunk CB, BATCH chunks per reserve / read barrier / push.
//
// CT: 0 CB, 1 CHUNK_TILES, 2 TILE_BYTES, 3 NUM_CHUNKS, 4 BATCH
// RT: 0 bank_base, 1 bank_id, 2 bank_offset
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb = get_compile_time_arg_val(0);
    constexpr uint32_t chunk_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t num_chunks = get_compile_time_arg_val(3);
    constexpr uint32_t batch = get_compile_time_arg_val(4);
    constexpr uint32_t chunk_bytes = chunk_tiles * tile_bytes;
    static_assert(num_chunks % batch == 0, "NUM_CHUNKS must be a multiple of BATCH");

    const uint32_t bank_base = get_arg_val<uint32_t>(0);
    const uint32_t bank_id = get_arg_val<uint32_t>(1);
    const uint32_t bank_offset = get_arg_val<uint32_t>(2);
    const uint64_t src = get_noc_addr_from_bank_id<true>(bank_id, bank_base + bank_offset);

    for (uint32_t c = 0; c < num_chunks; c += batch) {
        cb_reserve_back(cb, chunk_tiles * batch);
        const uint32_t l1 = get_write_ptr(cb);
        noc_async_read(src + static_cast<uint64_t>(c) * chunk_bytes, l1, chunk_bytes * batch);
        noc_async_read_barrier();
        cb_push_back(cb, chunk_tiles * batch);
    }
}
