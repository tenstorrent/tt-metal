// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Row core writer. Per tile-row of my block:
//   * my fp32 partial -> slot p of CB_GATHER on every core of my row group (myself included), then one increment of
//     each peer's gather semaphore; wait for the G - 1 peers' partials and publish the G slots to compute.
//   * output tiles -> DRAM.
// At the very end: advance this core's persistent call counter (the call parity every kernel here read at start;
// every consumer of it on this core is done by now: compute produced every output tile).

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "agrn_common.hpp"

using namespace agrn;

void kernel_main() {
    constexpr uint32_t out_tile_bytes = get_compile_time_arg_val(0);
    constexpr uint32_t acc_tile_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t acc_page_bytes = acc_tile_bytes + 64;  // S tile + tag (CB_PART / CB_GATHER page)
    constexpr uint32_t Wt = get_compile_time_arg_val(2);
    constexpr uint32_t rows = get_compile_time_arg_val(3);
    constexpr uint32_t G = get_compile_time_arg_val(4);
    constexpr uint32_t dest_tiles = get_compile_time_arg_val(5);
    constexpr auto out_args = TensorAccessorArgs<6>();

    size_t a = 0;
    const uint32_t out_addr = get_arg_val<uint32_t>(a++);
    const uint32_t ctr_addr = get_arg_val<uint32_t>(a++);
    const uint32_t gsem_addr = get_semaphore(get_arg_val<uint32_t>(a++));
    const uint32_t my_p = get_arg_val<uint32_t>(a++);
    const uint32_t out_page0 = get_arg_val<uint32_t>(a++);      // output page of (my first row, my first column)
    const uint32_t out_row_pitch = get_arg_val<uint32_t>(a++);  // output pages per tile-row (G * Wt)
    const uint32_t salt = get_arg_val<uint32_t>(a++);
    const uint32_t peers_idx = a;                               // G entries: (x << 16) | y of the row group's cores
    const uint32_t tag = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ctr_addr) + 1;

    const auto out = TensorAccessor(out_args, out_addr, out_tile_bytes);
    (void)gsem_addr;

    for (uint32_t r = 0; r < rows; ++r) {
        // ---- S exchange: my S tile -> slot my_p of every core of my row (myself included) ----
        {
            AGRN_ZONE("rw_wait_part");
            cb_wait_front(CB_PART, 1);
        }
        // one write per core (S tile + tag); every core polls the tags of its G slots: no write barrier and no
        // semaphore round trip on the critical path of the last tile-row
        const uint32_t src = get_read_ptr(CB_PART);
        write_tag(src + acc_tile_bytes, tag, salt);
        cb_reserve_back(CB_GATHER, G);
        const uint32_t gather = get_write_ptr(CB_GATHER);
        const uint32_t slot = gather + my_p * acc_page_bytes;
        for (uint32_t k = 0; k < G; ++k) {
            const uint32_t xy = get_arg_val<uint32_t>(peers_idx + k);
            noc_async_write(src, get_noc_addr(xy >> 16, xy & 0xFFFF, slot), acc_tile_bytes + TAG_BYTES);
        }
        noc_async_writes_flushed();
        cb_pop_front(CB_PART, 1);
        {
            AGRN_ZONE("rw_wait_peers");
            for (uint32_t k = 0; k < G; ++k) {
                wait_tag(gather + k * acc_page_bytes + acc_tile_bytes, tag, salt);
            }
        }
        cb_push_back(CB_GATHER, G);

        AGRN_ZONE("rw_output");
        const uint32_t page_row = out_page0 + r * out_row_pitch;
        for (uint32_t j0 = 0; j0 < Wt; j0 += dest_tiles) {
            const uint32_t n = (Wt - j0) < dest_tiles ? (Wt - j0) : dest_tiles;
            cb_wait_front(CB_OUT, n);
            uint32_t rp = get_read_ptr(CB_OUT);
            for (uint32_t k = 0; k < n; ++k) {
                noc_async_write_tile(page_row + j0 + k, out, rp);
                rp += out_tile_bytes;
            }
            noc_async_writes_flushed();
            cb_pop_front(CB_OUT, n);
        }
    }
    noc_async_write_barrier();
    noc_async_atomic_barrier();
    volatile tt_l1_ptr uint32_t* ctr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ctr_addr);
    *ctr = *ctr + 1;
}
