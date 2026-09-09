// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Writer paired with reader_reshape_rm_arbitrary.cpp (BATCHED).
//
// Interior slabs are output-aligned.  The final slab writes through the end of
// the aligned physical output page (the reader zeroed its padding), so every
// transaction is aligned, disjoint, and contained within one output page.
//
// Batched to match the reader: waits NABATCH cb_out slots, issues all their
// writes, then ONE noc_async_write_barrier, then pops NABATCH. Same B sequence
// as the reader (B = min(NABATCH, units_left)). Byte-identical; reduces to the
// original at NABATCH=1.
//
// CT: cb_out, new_stick_bytes, new_page_bytes, slab_bytes,
//     slabs_per_output, noc_max_burst_bytes, NABATCH, slab_slot_bytes,
//     TensorAccessorArgs(output)
// RT: dst_addr, start_unit, num_units
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc.h"

void kernel_main() {
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_unit = get_arg_val<uint32_t>(1);
    const uint32_t num_units = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t new_stick_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t new_page_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t slab_bytes = get_compile_time_arg_val(3);
    constexpr uint32_t slabs_per_output = get_compile_time_arg_val(4);
    constexpr uint32_t noc_max_burst_bytes = get_compile_time_arg_val(5);
    constexpr uint32_t NABATCH = get_compile_time_arg_val(6);
    constexpr uint32_t slab_slot_bytes = get_compile_time_arg_val(7);
    constexpr auto dst_args = TensorAccessorArgs<8>();

    const auto destination = TensorAccessor(dst_args, dst_addr, new_page_bytes);

    Noc noc;
    CircularBuffer out_buffer(cb_out);

    uint32_t units_left = num_units;
    uint32_t unit = start_unit;
    while (units_left > 0) {
        const uint32_t B = (units_left < NABATCH) ? units_left : NABATCH;
        out_buffer.wait_front(B);

        for (uint32_t j = 0; j < B; ++j) {
            const uint32_t u = unit + j;
            const uint32_t output_page = u / slabs_per_output;
            const uint32_t slab_index = u - output_page * slabs_per_output;
            const uint32_t output_column = slab_index * slab_bytes;
            const uint32_t logical_bytes =
                ((new_stick_bytes - output_column) < slab_bytes) ? (new_stick_bytes - output_column) : slab_bytes;
            const bool final = output_column + logical_bytes == new_stick_bytes;
            uint32_t remaining = final ? (new_page_bytes - output_column) : logical_bytes;

            const uint32_t source_offset = j * slab_slot_bytes;  // reader slab slot
            uint32_t offset = 0;
            while (remaining > 0) {
                const uint32_t burst = remaining < noc_max_burst_bytes ? remaining : noc_max_burst_bytes;
                noc.async_write(
                    out_buffer,
                    destination,
                    burst,
                    {.offset_bytes = source_offset + offset},
                    {.page_id = output_page, .offset_bytes = output_column + offset});
                offset += burst;
                remaining -= burst;
            }
        }
        noc.async_write_barrier();
        out_buffer.pop_front(B);
        unit += B;
        units_left -= B;
    }
}
