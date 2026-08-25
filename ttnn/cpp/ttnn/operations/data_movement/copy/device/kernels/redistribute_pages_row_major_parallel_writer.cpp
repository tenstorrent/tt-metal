// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

FORCE_INLINE uint32_t u32_min(uint32_t a, uint32_t b) { return a < b ? a : b; }

void kernel_main() {
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_unit = get_arg_val<uint32_t>(1);
    const uint32_t num_units = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t units_per_row = get_compile_time_arg_val(1);
    constexpr uint32_t output_pages_per_row = get_compile_time_arg_val(2);
    constexpr uint32_t output_page_elements = get_compile_time_arg_val(3);
    constexpr uint32_t row_elements = get_compile_time_arg_val(4);
    constexpr uint32_t unit_elements = get_compile_time_arg_val(5);
    constexpr uint32_t element_bytes = get_compile_time_arg_val(6);

    constexpr auto dst_args = TensorAccessorArgs<7>();
    const auto dst = TensorAccessor(dst_args, dst_addr);
    Noc noc;
    DataflowBuffer cb(cb_id);

    for (uint32_t unit = start_unit; unit < start_unit + num_units; ++unit) {
        const uint32_t row = unit / units_per_row;
        const uint32_t column = (unit % units_per_row) * unit_elements;
        const uint32_t elements = u32_min(unit_elements, row_elements - column);
        const uint32_t page_id = row * output_pages_per_row + column / output_page_elements;
        const uint32_t page_offset = (column % output_page_elements) * element_bytes;

        cb.wait_front(1);
        noc.async_write(
            cb, dst, elements * element_bytes, {.offset_bytes = 0}, {.page_id = page_id, .offset_bytes = page_offset});
        noc.async_write_barrier();
        cb.pop_front(1);
    }
}
