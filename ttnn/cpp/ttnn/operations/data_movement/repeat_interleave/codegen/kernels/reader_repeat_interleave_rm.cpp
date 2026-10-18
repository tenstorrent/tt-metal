// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reader for repeat_interleave on RM interleaved tensors.
//
// Source-page addressing is the shared SEQ_REPEAT_INTERLEAVE sequencer, the same one the unified
// tile reader selects. This kernel exists for the transport loop, not the mapping: an RM slot is a
// whole stick, so the read is stick_size bytes out of a page whose pitch is the buffer's aligned
// page size, whereas the unified reader transfers the whole page.
//
// Valid only when the interleaved dim is a whole-stick (outer or H) dim. The last (W, within-stick)
// dim needs a different addressing scheme that no sequencer implements.
//
// CT args: stick_size, input_page_size, l1_slot_stride, TensorAccessorArgs(in_t),
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

    Noc noc;
    CircularBuffer cb_in(cb_id);

    auto seq = seq_repeat_interleave_init(out_start_page, NUM_REPEATS, LOWER_PAGES, REP_DIM_PAGES);

    uint32_t pages_left = num_out_pages;

    while (pages_left > 0) {
        uint32_t batch = (pages_left < BATCH) ? pages_left : BATCH;
        cb_in.reserve_back(batch);
        uint32_t l1_offset = 0;

        for (uint32_t t = 0; t < batch; t++) {
            const uint32_t src_page = seq_repeat_interleave_next(seq);
            noc.async_read(s, cb_in, stick_size, {.page_id = src_page, .offset_bytes = 0}, {.offset_bytes = l1_offset});
            l1_offset += l1_slot_stride;
        }
        noc.async_read_barrier();
        cb_in.push_back(batch);
        pages_left -= batch;
    }
}
