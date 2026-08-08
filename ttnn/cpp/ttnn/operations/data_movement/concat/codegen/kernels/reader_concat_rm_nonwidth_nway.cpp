// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// N-way ROW_MAJOR non-width concat reader.
//
// Every output stick comes entirely from one original input.  The runtime
// cursor walks per-input stick blocks for each outer prefix, avoiding the
// pairwise intermediate cascade used by the two-input reader.
//
// CT: cb_id, BATCH, NUM_INPUTS, CB_PAGE_SIZE, IN_PAGE_SIZE,
//     TensorAccessorArgs(shared interleaved input placement)
// RT: num_sticks, current_input, current_input_stick,
//     base_addr[N], sticks_per_block[N], source_stick_id[N]
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    uint32_t num_sticks = get_arg_val<uint32_t>(0);
    uint32_t current_input = get_arg_val<uint32_t>(1);
    uint32_t current_input_stick = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t BATCH = get_compile_time_arg_val(1);
    constexpr uint32_t NUM_INPUTS = get_compile_time_arg_val(2);
    constexpr uint32_t CB_PAGE_SIZE = get_compile_time_arg_val(3);
    constexpr uint32_t IN_PAGE_SIZE = get_compile_time_arg_val(4);
    constexpr auto src_args = TensorAccessorArgs<5>();

    constexpr uint32_t BASE_OFFSET = 3;
    constexpr uint32_t BLOCK_OFFSET = BASE_OFFSET + NUM_INPUTS;
    constexpr uint32_t STICK_ID_OFFSET = BLOCK_OFFSET + NUM_INPUTS;
    uint32_t sticks_per_block[NUM_INPUTS];
    uint32_t stick_ids[NUM_INPUTS];
    for (uint32_t input = 0; input < NUM_INPUTS; ++input) {
        sticks_per_block[input] = get_arg_val<uint32_t>(BLOCK_OFFSET + input);
        stick_ids[input] = get_arg_val<uint32_t>(STICK_ID_OFFSET + input);
    }

    Noc noc;
    CircularBuffer input_cb(cb_id);

    uint32_t sticks_left = num_sticks;
    while (sticks_left > 0) {
        const uint32_t batch = (sticks_left < BATCH) ? sticks_left : BATCH;
        input_cb.reserve_back(batch);
        uint32_t l1_offset = 0;

        for (uint32_t stick = 0; stick < batch; ++stick) {
            const auto source =
                TensorAccessor(src_args, get_arg_val<uint32_t>(BASE_OFFSET + current_input), IN_PAGE_SIZE);
            noc.async_read(
                source, input_cb, IN_PAGE_SIZE, {.page_id = stick_ids[current_input]++}, {.offset_bytes = l1_offset});
            l1_offset += CB_PAGE_SIZE;

            current_input_stick++;
            if (current_input_stick == sticks_per_block[current_input]) {
                current_input_stick = 0;
                current_input++;
                if (current_input == NUM_INPUTS) {
                    current_input = 0;
                }
            }
        }
        noc.async_read_barrier();
        input_cb.push_back(batch);
        sticks_left -= batch;
    }
}
