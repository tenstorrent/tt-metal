// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <stdint.h>

#include "api/core_local_mem.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/tensor_accessor.h"
#include "experimental/kernel_args.h"

#include "slice_write_writer_common.hpp"

template <uint32_t alignment_offset, uint32_t page_begins_offset>
TT_KERNEL void writer(
    uint32_t output_stick_size,
    uint32_t input_stick_size,
    uint32_t stick_size_offset,
    uint32_t num_dims,
    uint32_t start_id,
    uint32_t num_sticks_per_core,
    uint32_t num_sticks_per_core_read,
    uint32_t num_read_per_barrier) {
    auto geometry = slice_write::load_strided_geometry(num_dims);
    const uint32_t output_base =
        get_common_arg_val<uint32_t>(decltype(tensor::output)::addr_crta_offset / sizeof(uint32_t));
    const auto output_accessor = TensorAccessor(decltype(tensor::output)::args, output_base, output_stick_size);
    const uint32_t noc_write_size = std::min(output_stick_size, input_stick_size);

    Noc noc;
    DataflowBuffer input(dfb::input);

    uint32_t dst_stick_id = start_id;
    uint32_t sticks_written = 0;
    for (uint32_t iter = 0; iter < num_sticks_per_core_read && sticks_written < num_sticks_per_core; ++iter) {
        input.wait_front(num_read_per_barrier);
        uint32_t src_offset = 0;
        for (uint32_t i = 0; i < num_read_per_barrier && sticks_written < num_sticks_per_core; ++i) {
            sticks_written++;
            noc.async_write(
                CoreLocalMem<uint32_t>(input.get_read_ptr() + src_offset + alignment_offset),
                output_accessor,
                noc_write_size,
                {},
                {.page_id = dst_stick_id, .offset_bytes = page_begins_offset});
            src_offset += stick_size_offset;
            dst_stick_id += geometry.reverse_stride[1];
            slice_write::advance(num_dims, geometry, dst_stick_id);
        }
        noc.async_write_barrier();
        input.pop_front(num_read_per_barrier);
    }
}
