// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reader for repeat_interleave on RM interleaved tensors.
//
// Same RM stick addressing as ops/repeat (now common/templates/reader_repeat_higherdim_rm.cpp), but the
// within-block source page uses the per-element (AABB) interleave formula
// instead of the modular (ABAB) repeat formula:
//
//   SRC_LOWER = REP_DIM_PAGES * LOWER_PAGES
//   DST_LOWER = NUM_REPEATS * SRC_LOWER
//   block     = out_page / DST_LOWER          (everything above rep_dim)
//   within    = out_page % DST_LOWER
//   lo        = within % LOWER_PAGES           (offset below rep_dim)
//   out_rep   = within / LOWER_PAGES           (replicated rep_dim index)
//   in_rep    = out_rep / NUM_REPEATS          (AABB collapse, per-element)
//   src       = block*SRC_LOWER + in_rep*LOWER_PAGES + lo
//
// This is valid only when the interleaved dim is a whole-stick (outer or H)
// dim — i.e. NOT the last (W, within-stick) dim. The Python builder routes the
// last-dim case to reader_repeat_interleave_lastdim_rm.cpp instead.
//
// CT args: stick_size, input_page_size, l1_slot_stride, TensorAccessorArgs(in_t),
//          cb_id, NUM_REPEATS, LOWER_PAGES, REP_DIM_PAGES, BATCH
// RT args: src_addr, num_out_pages, out_start_page
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t num_out_pages = get_arg_val<uint32_t>(1);
    uint32_t out_start_page = get_arg_val<uint32_t>(2);

    constexpr uint32_t stick_size = get_compile_time_arg_val(0);
    constexpr uint32_t input_page_size = get_compile_time_arg_val(1);
    constexpr uint32_t l1_slot_stride = get_compile_time_arg_val(2);
    constexpr auto src_args = TensorAccessorArgs<3>();
    constexpr uint32_t cb_id = get_compile_time_arg_val(src_args.next_compile_time_args_offset());
    constexpr uint32_t NUM_REPEATS = get_compile_time_arg_val(src_args.next_compile_time_args_offset() + 1);
    constexpr uint32_t LOWER_PAGES = get_compile_time_arg_val(src_args.next_compile_time_args_offset() + 2);
    constexpr uint32_t REP_DIM_PAGES = get_compile_time_arg_val(src_args.next_compile_time_args_offset() + 3);
    constexpr uint32_t BATCH = get_compile_time_arg_val(src_args.next_compile_time_args_offset() + 4);

    const auto s = TensorAccessor(src_args, src_addr, input_page_size);

    constexpr uint32_t SRC_LOWER = REP_DIM_PAGES * LOWER_PAGES;
    constexpr uint32_t DST_LOWER = NUM_REPEATS * SRC_LOWER;

    uint32_t out_page = out_start_page;
    uint32_t pages_left = num_out_pages;

    while (pages_left > 0) {
        uint32_t batch = (pages_left < BATCH) ? pages_left : BATCH;
        cb_reserve_back(cb_id, batch);
        uint32_t l1_addr = get_write_ptr(cb_id);

        for (uint32_t t = 0; t < batch; t++) {
            uint32_t block = out_page / DST_LOWER;
            uint32_t within = out_page % DST_LOWER;
            uint32_t lo = within % LOWER_PAGES;
            uint32_t out_rep = within / LOWER_PAGES;
            uint32_t in_rep = out_rep / NUM_REPEATS;
            uint32_t src_page = block * SRC_LOWER + in_rep * LOWER_PAGES + lo;

            uint64_t noc_addr = s.get_noc_addr(src_page);
            noc_async_read(noc_addr, l1_addr, stick_size);
            l1_addr += l1_slot_stride;
            out_page++;
        }
        noc_async_read_barrier();
        cb_push_back(cb_id, batch);
        pages_left -= batch;
    }
}
