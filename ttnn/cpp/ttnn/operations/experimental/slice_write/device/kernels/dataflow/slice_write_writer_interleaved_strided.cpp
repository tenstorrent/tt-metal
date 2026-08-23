// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/core_local_mem.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/scratchpad.h"
#include "api/tensor/tensor_accessor.h"
#include "experimental/kernel_args.h"

#include "slice_write_writer_common.hpp"

template <uint32_t alignment_offset, uint32_t page_begins_offset, uint32_t element_size, uint32_t output_row_stride>
TT_KERNEL void writer(
    uint32_t output_stick_size,
    uint32_t input_stick_size,
    uint32_t stick_size_offset,
    uint32_t num_dims,
    uint32_t start_id,
    uint32_t num_sticks_per_core,
    uint32_t num_sticks_per_core_read,
    uint32_t num_read_per_barrier) {
    auto read_geometry = slice_write::load_strided_geometry(num_dims);
    auto write_geometry = read_geometry;
    const uint32_t output_base =
        get_common_arg_val<uint32_t>(decltype(tensor::output)::addr_crta_offset / sizeof(uint32_t));
    const auto output_accessor = TensorAccessor(decltype(tensor::output)::args, output_base, output_stick_size);

    Noc noc;
    DataflowBuffer input(dfb::input);
    // This memory is filled and drained by the writer itself, so it is a raw
    // scratchpad rather than a producer/consumer DFB self-loop.
    Scratchpad<uint32_t> output_rows(scratch::output_rows);
    const uint32_t output_rows_base = output_rows.get_base_address();

    uint32_t dst_stick_id = start_id;
    uint32_t src_stick_id = start_id;
    uint32_t sticks_read = 0;
    uint32_t sticks_written = 0;
    // Batch destination-row reads before each barrier when the last dimension is strided.
    // Each row is then patched in L1 and written back, preserving elements outside the slice.
    for (uint32_t iter = 0; iter < num_sticks_per_core_read && sticks_written < num_sticks_per_core; ++iter) {
        uint32_t output_offset = 0;
        for (uint32_t i = 0; i < num_read_per_barrier && sticks_read < num_sticks_per_core; ++i) {
            sticks_read++;
            noc.async_read(
                output_accessor,
                CoreLocalMem<uint32_t>(output_rows_base + output_offset),
                output_stick_size,
                {.page_id = src_stick_id},
                {});
            output_offset += output_row_stride;
            src_stick_id += read_geometry.reverse_stride[1];
            slice_write::advance(num_dims, read_geometry, src_stick_id);
        }
        noc.async_read_barrier();

        input.wait_front(num_read_per_barrier);
        output_offset = 0;
        uint32_t src_offset = 0;
        for (uint32_t i = 0; i < num_read_per_barrier && sticks_written < num_sticks_per_core; ++i) {
            sticks_written++;
            volatile tt_l1_ptr uint8_t* out_stick =
                reinterpret_cast<volatile tt_l1_ptr uint8_t*>(output_rows_base + output_offset + page_begins_offset);
            volatile tt_l1_ptr uint8_t* in_stick =
                reinterpret_cast<volatile tt_l1_ptr uint8_t*>(input.get_read_ptr() + src_offset + alignment_offset);
            uint32_t out_index = 0;
            // Copy element bytes explicitly so this path works for every supported datum width.
            for (uint32_t j = 0; j < input_stick_size / element_size; ++j) {
                for (uint32_t byte = 0; byte < element_size; ++byte) {
                    out_stick[out_index * element_size + byte] = in_stick[j * element_size + byte];
                }
                out_index += write_geometry.reverse_stride[0];
            }
            noc.async_write(
                CoreLocalMem<uint32_t>(output_rows_base + output_offset),
                output_accessor,
                output_stick_size,
                {},
                {.page_id = dst_stick_id});
            output_offset += output_row_stride;
            src_offset += stick_size_offset;
            dst_stick_id += write_geometry.reverse_stride[1];
            slice_write::advance(num_dims, write_geometry, dst_stick_id);
        }
        noc.async_write_barrier();
        input.pop_front(num_read_per_barrier);
    }
}
