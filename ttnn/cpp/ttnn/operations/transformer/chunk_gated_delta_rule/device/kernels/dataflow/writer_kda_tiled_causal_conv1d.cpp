// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr uint32_t Ct = get_compile_time_arg_val(0);
    constexpr uint32_t row_bytes = get_compile_time_arg_val(1);
    constexpr auto output_a = TensorAccessorArgs<2>();
    constexpr auto state_a = TensorAccessorArgs<output_a.next_compile_time_args_offset()>();
    const uint32_t mt_start = get_arg_val<uint32_t>(0);
    const uint32_t mt_count = get_arg_val<uint32_t>(1);
    const uint32_t write_state = get_arg_val<uint32_t>(2);
    const uint32_t output_addr = get_arg_val<uint32_t>(3);
    const uint32_t state_addr = get_arg_val<uint32_t>(4);
    constexpr uint32_t tile_bytes = 2048;
    const auto output = TensorAccessor(output_a, output_addr, tile_bytes);
    const auto state = TensorAccessor(state_a, state_addr, row_bytes);
    CircularBuffer output_cb(5);
    CircularBuffer state_cb(7);
    Noc noc;
    for (uint32_t item = 0; item < mt_count; ++item) {
        output_cb.wait_front(Ct);
        const uint32_t tile_base = (mt_start + item) * Ct;
        for (uint32_t ct = 0; ct < Ct; ++ct) {
            noc.async_write(
                output_cb, output, tile_bytes, {.offset_bytes = ct * tile_bytes}, {.page_id = tile_base + ct});
        }
        noc.async_write_barrier();
        output_cb.pop_front(Ct);
    }
    if (write_state) {
        state_cb.wait_front(3);
        for (uint32_t row = 0; row < 3; ++row) {
            noc.async_write(state_cb, state, row_bytes, {.offset_bytes = row * row_bytes}, {.page_id = row});
        }
        noc.async_write_barrier();
        state_cb.pop_front(3);
    }
}
