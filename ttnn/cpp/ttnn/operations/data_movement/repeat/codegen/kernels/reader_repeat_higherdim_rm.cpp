// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Repeat-local reader for HIGHER-DIM replication on RM interleaved tensors.
//
// This mirrors the shared reader_repeat_higherdim_rm.cpp mapping, with
// compile-time shortcuts for size-1 repeated dimensions. Tiny repeat cases often
// broadcast a singleton dim; avoiding the full generic div/mod map keeps those
// cases from losing to TTNN's collapsed RM path.
//
// CT args: xfer_size, l1_stride, TensorAccessorArgs(in_t),
//          cb_id, NUM_REPEATS, LOWER_PAGES, REP_DIM_PAGES, BATCH
// RT args: src_addr, num_out_pages, out_start_page
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t num_out_pages = get_arg_val<uint32_t>(1);
    uint32_t out_start_page = get_arg_val<uint32_t>(2);

    constexpr uint32_t xfer_size = get_compile_time_arg_val(0);
    constexpr uint32_t l1_stride = get_compile_time_arg_val(1);
    constexpr auto src_args = TensorAccessorArgs<2>();
    constexpr uint32_t cb_id = get_compile_time_arg_val(src_args.next_compile_time_args_offset());
    constexpr uint32_t NUM_REPEATS = get_compile_time_arg_val(src_args.next_compile_time_args_offset() + 1);
    constexpr uint32_t LOWER_PAGES = get_compile_time_arg_val(src_args.next_compile_time_args_offset() + 2);
    constexpr uint32_t REP_DIM_PAGES = get_compile_time_arg_val(src_args.next_compile_time_args_offset() + 3);
    constexpr uint32_t BATCH = get_compile_time_arg_val(src_args.next_compile_time_args_offset() + 4);

    const auto s = TensorAccessor(src_args, src_addr);

    constexpr uint32_t SRC_LOWER = REP_DIM_PAGES * LOWER_PAGES;
    constexpr uint32_t DST_LOWER = NUM_REPEATS * SRC_LOWER;

    uint32_t out_page = out_start_page;
    uint32_t pages_left = num_out_pages;

    while (pages_left > 0) {
        uint32_t batch = (pages_left < BATCH) ? pages_left : BATCH;
        cb_reserve_back(cb_id, batch);
        uint32_t l1_addr = get_write_ptr(cb_id);

        for (uint32_t t = 0; t < batch; t++) {
            uint32_t src_page;

            if constexpr (REP_DIM_PAGES == 1 && LOWER_PAGES == 1) {
                src_page = out_page / NUM_REPEATS;
            } else if constexpr (REP_DIM_PAGES == 1) {
                uint32_t block = out_page / (NUM_REPEATS * LOWER_PAGES);
                uint32_t within = out_page % (NUM_REPEATS * LOWER_PAGES);
                src_page = block * LOWER_PAGES + (within % LOWER_PAGES);
            } else {
                uint32_t block = out_page / DST_LOWER;
                uint32_t within = out_page % DST_LOWER;
                uint32_t lower_in_rep = within % SRC_LOWER;
                src_page = block * SRC_LOWER + lower_in_rep;
            }

            uint64_t noc_addr = s.get_noc_addr(src_page);
            noc_async_read(noc_addr, l1_addr, xfer_size);
            l1_addr += l1_stride;
            out_page++;
        }
        noc_async_read_barrier();
        cb_push_back(cb_id, batch);
        pages_left -= batch;
    }
}
