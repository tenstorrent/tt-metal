// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Repeat-local reader for HIGHER-DIM replication on RM interleaved tensors.
//
// The page map is SEQ_REPEAT's from sequencers.h, with the repeat geometry as
// compile-time args so its per-stick div/mod strength-reduces; a stick transfer is
// short enough that software divides would dominate it.
//
// CT args: xfer_size, l1_stride, TensorAccessorArgs(in_t),
//          cb_id, NUM_REPEATS, LOWER_PAGES, REP_DIM_PAGES, BATCH
// RT args: src_addr, num_out_pages, out_start_page
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/operations/data_movement/common/kernels/codegen/sequencers.h"

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

    Noc noc;
    CircularBuffer cb_in(cb_id);

    SeqRepeatState seq = seq_repeat_init(out_start_page, NUM_REPEATS, LOWER_PAGES, REP_DIM_PAGES);
    uint32_t pages_left = num_out_pages;

    while (pages_left > 0) {
        uint32_t batch = (pages_left < BATCH) ? pages_left : BATCH;
        cb_in.reserve_back(batch);
        uint32_t l1_offset = 0;

        for (uint32_t t = 0; t < batch; t++) {
            const uint32_t src_page = seq_repeat_next(seq);
            noc.async_read(s, cb_in, xfer_size, {.page_id = src_page, .offset_bytes = 0}, {.offset_bytes = l1_offset});
            l1_offset += l1_stride;
        }
        noc.async_read_barrier();
        cb_in.push_back(batch);
        pages_left -= batch;
    }
}
