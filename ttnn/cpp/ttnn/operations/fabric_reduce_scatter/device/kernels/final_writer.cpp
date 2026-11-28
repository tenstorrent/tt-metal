// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Final writer (BRISC, NoC1): my chunks of the summed own block into the output; then re-arm counters A / B.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/operations/fabric_reduce_scatter/device/kernels/fabric_reduce_scatter_args.hpp"
#include "ttnn/cpp/ttnn/operations/fabric_reduce_scatter/device/kernels/fabric_reduce_scatter_chunk_walk.hpp"

using namespace fabric_reduce_scatter;

void kernel_main() {
    constexpr uint32_t page_bytes = get_compile_time_arg_val(ct::PAGE_BYTES);
    constexpr uint32_t run_pages = get_compile_time_arg_val(ct::RUN_PAGES);
    constexpr uint32_t num_banks = get_compile_time_arg_val(ct::NUM_BANKS);
    constexpr uint32_t group = get_compile_time_arg_val(ct::GROUP);
    constexpr auto out_args = TensorAccessorArgs<ct::COUNT>();
    constexpr uint32_t chunk_bytes = run_pages * page_bytes;

    const uint32_t out_addr = get_common_arg_val<uint32_t>(final_writer_common::OUT_ADDR);
    const uint32_t blk_pages = get_common_arg_val<uint32_t>(final_writer_common::BLK_PAGES);
    const uint32_t a_addr = get_common_arg_val<uint32_t>(final_writer_common::SEM_A_ADDR);
    const uint32_t b_addr = get_common_arg_val<uint32_t>(final_writer_common::SEM_B_ADDR);
    const uint32_t first_bank = get_arg_val<uint32_t>(final_writer_rt::FIRST_BANK);
    const uint32_t bank_stride = get_arg_val<uint32_t>(final_writer_rt::BANK_STRIDE);
    uint32_t chunks_left = get_arg_val<uint32_t>(final_writer_rt::NUM_CHUNKS);
    const uint32_t expect_a = get_arg_val<uint32_t>(final_writer_rt::EXPECT_A);
    const uint32_t expect_b = get_arg_val<uint32_t>(final_writer_rt::EXPECT_B);
    const auto out = TensorAccessor(out_args, out_addr, page_bytes);

    uint32_t pending = 0;
    for_each_chunk(blk_pages, first_bank, bank_stride, num_banks, run_pages, [&](uint32_t page, uint32_t n, uint32_t) {
        cb_wait_front(cb::OUT, run_pages * (pending + 1));
        noc_async_write(get_read_ptr(cb::OUT) + pending * chunk_bytes, out.get_noc_addr(page), n * page_bytes);
        --chunks_left;
        if (++pending == group || chunks_left == 0) {
            noc_async_write_barrier();
            cb_pop_front(cb::OUT, run_pages * pending);
            pending = 0;
        }
    });
    if (expect_a > 0) {
        volatile tt_l1_ptr uint32_t* c = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(a_addr);
        noc_semaphore_wait_min(c, expect_a);
        noc_semaphore_inc(get_noc_addr(a_addr), 0u - expect_a);
    }
    if (expect_b > 0) {
        volatile tt_l1_ptr uint32_t* c = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(b_addr);
        noc_semaphore_wait_min(c, expect_b);
        noc_semaphore_inc(get_noc_addr(b_addr), 0u - expect_b);
    }
    noc_async_atomic_barrier();
}
