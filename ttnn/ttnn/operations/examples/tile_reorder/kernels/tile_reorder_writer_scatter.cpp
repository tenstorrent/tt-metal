// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// tile_reorder writer (BRISC / NoC1) — SCATTER method (doing the work).
//
// Writes the SAME tile as its 4 faces (512 B each) instead of one 2 KB page.
// This models the realistic generic, tile-unaware permute: it moves the data in
// sub-tile pieces. The bytes and the destination are identical to the relocate
// writer — but the smaller DRAM write transactions achieve lower bandwidth, so
// the DRAM-bound op runs slower. Same output, more/slower NoC traffic.
//
// (A bf16 32x32 tile is 4 faces of 16x16 = 512 B each, contiguous within the
// page; splitting the write is purely a transaction-size change.)

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_id = 0;
    constexpr uint32_t page_bytes = get_compile_time_arg_val(0);
    constexpr auto out_args = TensorAccessorArgs<1>();

    constexpr uint32_t FACES_PER_TILE = 4;
    constexpr uint32_t face_bytes = page_bytes / FACES_PER_TILE;

    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_page = get_arg_val<uint32_t>(1);
    const uint32_t num_pages = get_arg_val<uint32_t>(2);

    const auto out_acc = TensorAccessor(out_args, dst_addr, page_bytes);

    for (uint32_t p = 0; p < num_pages; ++p) {
        const uint32_t out_idx = start_page + p;

        cb_wait_front(cb_id, 1);
        const uint32_t l1_read_addr = get_read_ptr(cb_id);
        const uint64_t dst_base = out_acc.get_noc_addr(out_idx);

        // Naive: one small write per 512 B face AND a barrier after each, so
        // every tiny transaction pays full NoC latency with no overlap (the
        // generic-permute anti-pattern: sub-tile writes, unbatched barriers).
        for (uint32_t f = 0; f < FACES_PER_TILE; ++f) {
            const uint32_t off = f * face_bytes;
            noc_async_write(l1_read_addr + off, dst_base + off, face_bytes);
            noc_async_write_barrier();
        }
        cb_pop_front(cb_id, 1);
    }
}
