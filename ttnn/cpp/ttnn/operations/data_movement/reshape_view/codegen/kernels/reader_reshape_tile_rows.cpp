// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Tile-row permutation TILE reshape reader: stages each batch's plan page and its distinct source tiles once.
// Every batch reserves all in_slots (only n_in are filled) so batches never straddle the wrap of the 2 * in_slots CB.
#include <cstdint>

#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/debug/assert.h"

void kernel_main() {
    const uint32_t input_addr = get_arg_val<uint32_t>(0);
    const uint32_t plan_addr = get_arg_val<uint32_t>(1);
    const uint32_t start_batch = get_arg_val<uint32_t>(2);
    const uint32_t end_batch = get_arg_val<uint32_t>(3);

    constexpr uint32_t plan_page_bytes = get_compile_time_arg_val(0);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t plan_cb = get_compile_time_arg_val(2);
    constexpr uint32_t input_cb = get_compile_time_arg_val(3);
    constexpr uint32_t in_slots = get_compile_time_arg_val(4);
    constexpr uint32_t header_words = get_compile_time_arg_val(5);
    constexpr auto plan_args = TensorAccessorArgs<6>();
    constexpr auto input_args = TensorAccessorArgs<plan_args.next_compile_time_args_offset()>();
    static_assert(in_slots > 0, "a batch reads at least one source tile");
    static_assert((header_words + in_slots) * sizeof(uint32_t) <= plan_page_bytes, "plan page holds the slot list");

    const auto input_accessor = TensorAccessor(input_args, input_addr);
    const auto plan_accessor = TensorAccessor(plan_args, plan_addr);
    Noc noc;
    CircularBuffer plan_buffer(plan_cb);
    CircularBuffer input_buffer(input_cb);

    for (uint32_t batch = start_batch; batch < end_batch; ++batch) {
        plan_buffer.reserve_back(1);
        const uint32_t plan_l1 = plan_buffer.get_write_ptr();
        noc.async_read<NocOptions::DEFAULT, plan_page_bytes>(
            plan_accessor, plan_buffer, plan_page_bytes, {.page_id = batch, .offset_bytes = 0}, {.offset_bytes = 0});
        noc.async_read_barrier();
        CoreLocalMem<volatile uint32_t> plan(plan_l1);
        const uint32_t n_in = plan[0];
        ASSERT(n_in <= in_slots);

        input_buffer.reserve_back(in_slots);
        for (uint32_t slot = 0; slot < n_in; ++slot) {
            noc.async_read<NocOptions::DEFAULT, tile_bytes>(
                input_accessor,
                input_buffer,
                tile_bytes,
                {.page_id = plan[header_words + slot], .offset_bytes = 0},
                {.offset_bytes = slot * tile_bytes});
        }
        plan_buffer.push_back(1);
        noc.async_read_barrier();
        input_buffer.push_back(in_slots);
    }
}
