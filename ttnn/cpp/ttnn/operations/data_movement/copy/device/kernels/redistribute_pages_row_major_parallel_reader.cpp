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
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_unit = get_arg_val<uint32_t>(1);
    const uint32_t num_units = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t units_per_row = get_compile_time_arg_val(1);
    constexpr uint32_t input_pages_per_row = get_compile_time_arg_val(2);
    constexpr uint32_t input_page_elements = get_compile_time_arg_val(3);
    constexpr uint32_t row_elements = get_compile_time_arg_val(4);
    constexpr uint32_t unit_elements = get_compile_time_arg_val(5);
    constexpr uint32_t element_bytes = get_compile_time_arg_val(6);

    constexpr auto src_args = TensorAccessorArgs<7>();
    const auto src = TensorAccessor(src_args, src_addr);
    Noc noc;
    DataflowBuffer cb(cb_id);

    for (uint32_t unit = start_unit; unit < start_unit + num_units; ++unit) {
        const uint32_t row = unit / units_per_row;
        const uint32_t column = (unit % units_per_row) * unit_elements;
        const uint32_t elements = u32_min(unit_elements, row_elements - column);
        const uint32_t page_id = row * input_pages_per_row + column / input_page_elements;
        const uint32_t page_offset = (column % input_page_elements) * element_bytes;

        cb.reserve_back(1);
        noc.async_read(
            src, cb, elements * element_bytes, {.page_id = page_id, .offset_bytes = page_offset}, {.offset_bytes = 0});
        noc.async_read_barrier();
        cb.push_back(1);
    }
}
