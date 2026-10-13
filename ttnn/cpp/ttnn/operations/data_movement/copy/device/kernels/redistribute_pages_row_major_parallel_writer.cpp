// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

FORCE_INLINE std::uint32_t u32_min(std::uint32_t a, std::uint32_t b) { return a < b ? a : b; }

void kernel_main() {
    const auto start_unit = get_arg(args::start_unit);
    const auto num_units = get_arg(args::num_units);

    constexpr auto units_per_row = get_arg(args::units_per_row);
    constexpr auto output_pages_per_row = get_arg(args::output_pages_per_row);
    constexpr auto output_page_elements = get_arg(args::output_page_elements);
    constexpr auto row_elements = get_arg(args::row_elements);
    constexpr auto unit_elements = get_arg(args::unit_elements);
    constexpr auto element_bytes = get_arg(args::element_bytes);

    const auto dst = TensorAccessor(tensor::dst);
    Noc noc;
    DataflowBuffer unit_buffer(dfb::unit);

    for (std::uint32_t unit = start_unit; unit < start_unit + num_units; ++unit) {
        const std::uint32_t row = unit / units_per_row;
        const std::uint32_t column = (unit % units_per_row) * unit_elements;
        const std::uint32_t elements = u32_min(unit_elements, row_elements - column);
        const std::uint32_t page_id = row * output_pages_per_row + column / output_page_elements;
        const std::uint32_t page_offset = (column % output_page_elements) * element_bytes;

        unit_buffer.wait_front(1);
        noc.async_write(
            unit_buffer,
            dst,
            elements * element_bytes,
            {.offset_bytes = 0},
            {.page_id = page_id, .offset_bytes = page_offset});
        noc.async_write_barrier();
        unit_buffer.pop_front(1);
    }
}
