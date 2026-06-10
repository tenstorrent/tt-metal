// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t values_addr = get_arg_val<uint32_t>(0);
    const uint32_t indices_addr = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_values = get_compile_time_arg_val(0);
    constexpr uint32_t cb_indices = get_compile_time_arg_val(1);
    constexpr uint32_t num_rows = get_compile_time_arg_val(2);
    constexpr uint32_t output_slices_per_row = get_compile_time_arg_val(3);
    constexpr uint32_t values_slice_bytes = get_compile_time_arg_val(4);
    constexpr uint32_t indices_slice_bytes = get_compile_time_arg_val(5);
    constexpr uint32_t values_page_bytes = get_compile_time_arg_val(6);
    constexpr uint32_t indices_page_bytes = get_compile_time_arg_val(7);
    constexpr uint32_t output_tiles_per_row = get_compile_time_arg_val(8);
    constexpr uint32_t rank_page_groups = output_slices_per_row / 16;
    constexpr auto values_args = TensorAccessorArgs<9>();

    const auto values = TensorAccessor(values_args, values_addr, values_page_bytes);
    const auto indices = TensorAccessor(values_args, indices_addr, indices_page_bytes);
    CircularBuffer values_cb(cb_values);
    CircularBuffer indices_cb(cb_indices);
    Noc noc;

    for (uint32_t row = 0; row < num_rows; ++row) {
        values_cb.wait_front(output_tiles_per_row);
        for (uint32_t page = 0; page < output_slices_per_row; ++page) {
            const uint32_t rank_page = ((page & 0xF) * rank_page_groups) | (page >> 4);
            const uint32_t src_offset = page * values_slice_bytes;
            const uint32_t dst_offset = rank_page * values_slice_bytes;
            noc.async_write(
                values_cb,
                values,
                values_slice_bytes,
                {.offset_bytes = src_offset},
                {.page_id = row, .offset_bytes = dst_offset});
        }
        noc.async_write_barrier();
        values_cb.pop_front(output_tiles_per_row);

        indices_cb.wait_front(output_tiles_per_row);
        for (uint32_t page = 0; page < output_slices_per_row; ++page) {
            const uint32_t rank_page = ((page & 0xF) * rank_page_groups) | (page >> 4);
            const uint32_t src_offset = page * indices_slice_bytes;
            const uint32_t dst_offset = rank_page * indices_slice_bytes;
            noc.async_write(
                indices_cb,
                indices,
                indices_slice_bytes,
                {.offset_bytes = src_offset},
                {.page_id = row, .offset_bytes = dst_offset});
        }
        noc.async_write_barrier();
        indices_cb.pop_front(output_tiles_per_row);
    }
}
