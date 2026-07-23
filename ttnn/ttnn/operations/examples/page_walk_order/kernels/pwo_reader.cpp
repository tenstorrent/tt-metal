// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// page_walk_order reader (NCRISC / NoC0).
//
// Walks every one of N interleaved-DRAM source pages exactly once, in a page-index
// order set by a compile-time `stride`, and accumulates an order-independent integer
// checksum. Interleaved DRAM places page p in bank (p % num_banks), so the walk order
// decides which banks the reader's in-flight reads target:
//
//   stride == num_banks    -> every read in a block hits the SAME bank (serialized).
//   stride == 1            -> consecutive reads spread across all banks (parallel).
//   stride coprime to banks -> bank index steps by (stride % num_banks), also spreads.
//
// The walk is a general coset enumeration: idx = (base + k*stride) mod N, over
// g = gcd(stride, N) cosets each of length N/g. This visits every page exactly once
// for ANY stride, so the read multiset (and thus the checksum) is order-independent —
// only the temporal order changes. Reads are issued a BLOCK at a time under ONE
// barrier so several are outstanding at once and the bank parallelism can manifest;
// the checksum (integer add of every uint16 halfword) is commutative, so it is
// identical across walk orders and matches the host reference. The single tiny output
// write (one 32-byte page holding the checksum) is negligible.

#include "api/dataflow/dataflow_api.h"

constexpr uint32_t cb_pages = 0;  // scratch: `block` source pages read under one barrier
constexpr uint32_t cb_out = 1;    // scratch: one 32-byte checksum output page

void kernel_main() {
    constexpr uint32_t page_bytes = get_compile_time_arg_val(0);
    constexpr uint32_t N = get_compile_time_arg_val(1);          // total source pages
    constexpr uint32_t stride = get_compile_time_arg_val(2);     // page-index step between reads
    constexpr uint32_t g = get_compile_time_arg_val(3);          // gcd(stride, N) = number of cosets
    constexpr uint32_t coset_len = get_compile_time_arg_val(4);  // N / g = pages per coset
    constexpr uint32_t block = get_compile_time_arg_val(5);      // reads kept in flight per barrier
    constexpr uint32_t kernel_iters = get_compile_time_arg_val(6);
    constexpr auto src_args = TensorAccessorArgs<7>();
    constexpr auto out_args = TensorAccessorArgs<src_args.next_compile_time_args_offset()>();

    constexpr uint32_t words_per_page = page_bytes / 2;  // uint16 halfwords per page

    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t out_addr = get_arg_val<uint32_t>(1);

    const auto src_acc = TensorAccessor(src_args, src_addr, page_bytes);
    const auto out_acc = TensorAccessor(out_args, out_addr, 32);

    uint32_t checksum = 0;

    for (uint32_t it = 0; it < kernel_iters; ++it) {
        checksum = 0;
        // Coset-walk state: `p` is the current page index; `coset_pos` counts within the
        // current coset; when a coset finishes, advance `base` and restart at p = base.
        uint32_t base = 0;
        uint32_t p = 0;
        uint32_t coset_pos = 0;
        uint32_t produced = 0;

        while (produced < N) {
            uint32_t this_block = block;
            if (produced + this_block > N) {
                this_block = N - produced;
            }

            // Reuse the fixed scratch region every block (never pushed -> same L1 each time).
            cb_reserve_back(cb_pages, block);
            const uint32_t l1 = get_write_ptr(cb_pages);

            // Issue this_block pipelined reads in the current walk order, then ONE barrier
            // so multiple reads are outstanding and bank parallelism can appear.
            for (uint32_t b = 0; b < this_block; ++b) {
                noc_async_read(src_acc.get_noc_addr(p), l1 + b * page_bytes, page_bytes);
                // Advance the page-index walk.
                p += stride;
                if (p >= N) {
                    p -= N;
                }
                ++coset_pos;
                if (coset_pos == coset_len) {
                    coset_pos = 0;
                    ++base;
                    p = base;
                }
                ++produced;
            }
            noc_async_read_barrier();

            // Order-independent checksum: one halfword (word 0) per page. This is a
            // NEGLIGIBLE compute (this_block adds), so the measured time is the READ
            // walk, not the checksum — but it still confirms every page was fetched to
            // the right slot and that all walk orders read the identical page set.
            volatile tt_l1_ptr uint16_t* wp = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1);
            for (uint32_t b = 0; b < this_block; ++b) {
                checksum += wp[b * words_per_page];
            }
        }
    }

    // Negligible write: one 32-byte page with the checksum in word 0.
    cb_reserve_back(cb_out, 1);
    const uint32_t out_l1 = get_write_ptr(cb_out);
    volatile tt_l1_ptr uint32_t* op = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_l1);
    for (uint32_t i = 0; i < 8; ++i) {
        op[i] = 0;
    }
    op[0] = checksum;
    noc_async_write(out_l1, out_acc.get_noc_addr(0), 32);
    noc_async_write_barrier();
}
