// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Two-input non-width reader: alternates blocks of sticks from each input in output order.
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/operations/data_movement/common/kernels/codegen/sequencers.h"

void kernel_main() {
    uint32_t num_sticks = get_arg_val<uint32_t>(0);
    const uint32_t start_tensor = get_arg_val<uint32_t>(1);
    const uint32_t start_tensor_id = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_in = get_compile_time_arg_val(0);
    constexpr uint32_t BATCH = get_compile_time_arg_val(1);
    constexpr uint32_t ppb_0 = get_compile_time_arg_val(2);
    constexpr uint32_t ppb_1 = get_compile_time_arg_val(3);
    constexpr uint32_t CB_PAGE_SIZE = get_compile_time_arg_val(4);
    constexpr uint32_t IN0_PAGE_SIZE = get_compile_time_arg_val(5);
    constexpr uint32_t IN1_PAGE_SIZE = get_compile_time_arg_val(6);

    constexpr uint32_t ta_base = 7;
    constexpr auto ta0_args = TensorAccessorArgs<ta_base>();
    constexpr auto ta1_args = TensorAccessorArgs<ta0_args.next_compile_time_args_offset()>();

    const uint32_t src_addr_0 = get_common_arg_val<uint32_t>(0);
    const uint32_t src_addr_1 = get_common_arg_val<uint32_t>(1);

    SeqConcatState seq = seq_concat_init(
        start_tensor, start_tensor_id, get_arg_val<uint32_t>(3), get_arg_val<uint32_t>(4), ppb_0, ppb_1);

    // DRAM and L1 pages have different alignment, so each accessor needs its own buffer's pitch.
    const auto s0 = TensorAccessor(ta0_args, src_addr_0, IN0_PAGE_SIZE);
    const auto s1 = TensorAccessor(ta1_args, src_addr_1, IN1_PAGE_SIZE);

    Noc noc;
    CircularBuffer input_cb(cb_in);

    uint32_t sticks_left = num_sticks;

    while (sticks_left > 0) {
        uint32_t batch = (sticks_left < BATCH) ? sticks_left : BATCH;
        input_cb.reserve_back(batch);
        uint32_t l1_offset = 0;

        for (uint32_t t = 0; t < batch; t++) {
            const bool from_first = seq.curr_tensor == 0;
            const uint32_t stick_id = seq_concat_next(seq);
            if (from_first) {
                noc.async_read(s0, input_cb, IN0_PAGE_SIZE, {.page_id = stick_id}, {.offset_bytes = l1_offset});
            } else {
                noc.async_read(s1, input_cb, IN1_PAGE_SIZE, {.page_id = stick_id}, {.offset_bytes = l1_offset});
            }
            l1_offset += CB_PAGE_SIZE;
        }
        noc.async_read_barrier();
        input_cb.push_back(batch);
        sticks_left -= batch;
    }
}
