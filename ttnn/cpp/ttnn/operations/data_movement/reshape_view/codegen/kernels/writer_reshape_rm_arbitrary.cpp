// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc.h"

void kernel_main() {
    const uint32_t dst_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t start_unit = get_arg_val<uint32_t>(0);
    const uint32_t num_units = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_out = get_named_compile_time_arg_val("cb_out");
    constexpr uint32_t new_stick_bytes = get_named_compile_time_arg_val("new_stick_bytes");
    constexpr uint32_t new_page_bytes = get_named_compile_time_arg_val("new_page_bytes");
    constexpr uint32_t slab_bytes = get_named_compile_time_arg_val("slab_bytes");
    constexpr uint32_t slot_bytes = get_named_compile_time_arg_val("slot_bytes");
    constexpr uint32_t slabs_per_stick = get_named_compile_time_arg_val("slabs_per_stick");
    constexpr uint32_t batch = get_named_compile_time_arg_val("batch");
    constexpr auto dst_args = TensorAccessorArgs<0>();

    const auto destination = TensorAccessor(dst_args, dst_addr, new_page_bytes);
    Noc noc;
    CircularBuffer out_buffer(cb_out);

    uint32_t unit = start_unit;
    uint32_t units_left = num_units;
    while (units_left > 0) {
        const uint32_t n = units_left < batch ? units_left : batch;
        out_buffer.wait_front(n);
        for (uint32_t j = 0; j < n; ++j) {
            const uint32_t stick = (unit + j) / slabs_per_stick;
            const uint32_t column = (unit + j - stick * slabs_per_stick) * slab_bytes;
            const uint32_t stick_left = new_stick_bytes - column;
            // A slab never exceeds one NOC burst.
            noc.async_write(
                out_buffer,
                destination,
                stick_left < slab_bytes ? stick_left : slab_bytes,
                {.offset_bytes = j * slot_bytes},
                {.page_id = stick, .offset_bytes = column});
        }
        // The slots are free once the writes have left L1; completion is only awaited at exit.
        noc.async_writes_flushed();
        out_buffer.pop_front(n);
        unit += n;
        units_left -= n;
    }
    noc.async_write_barrier();
}
