// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/tensor_accessor.h"
#include "slice_write_writer_common.hpp"

template <uint32_t unpad_input_width>
TT_KERNEL void writer(
    uint32_t output_byte_offset,
    uint32_t output_stick_size,
    uint32_t input_stick_size,
    uint32_t stick_size_offset,
    uint32_t num_dims,
    uint32_t start_id,
    uint32_t num_sticks_per_core,
    uint32_t num_sticks_per_core_read,
    uint32_t num_read_per_barrier,
    uint32_t padding_width_units) {
    auto geometry = slice_write::load_geometry(num_dims);
    const uint32_t output_base =
        get_common_arg_val<uint32_t>(decltype(tensor::output)::addr_crta_offset / sizeof(uint32_t));
    const auto output_accessor =
        TensorAccessor(decltype(tensor::output)::args, output_base + output_byte_offset, output_stick_size);
    const uint32_t noc_write_size = std::min(output_stick_size, input_stick_size);

    Noc noc;
    DataflowBuffer input(dfb::input);

    uint32_t dst_stick_id = start_id;
    uint32_t sticks_read = 0;
    for (uint32_t iter = 0; iter < num_sticks_per_core_read && sticks_read < num_sticks_per_core; ++iter) {
        input.wait_front(num_read_per_barrier);
        uint32_t src_offset = 0;

        for (uint32_t i = 0; i < num_read_per_barrier && sticks_read < num_sticks_per_core; ++i) {
            sticks_read++;
            if constexpr (unpad_input_width != 0) {
                if ((geometry.id[0] + padding_width_units + 1) <= geometry.num_unpadded[0]) {
                    noc.async_write(
                        input,
                        output_accessor,
                        noc_write_size,
                        {.offset_bytes = src_offset},
                        {.page_id = dst_stick_id});
                }
            } else {
                noc.async_write(
                    input, output_accessor, noc_write_size, {.offset_bytes = src_offset}, {.page_id = dst_stick_id});
            }
            src_offset += stick_size_offset;
            dst_stick_id++;
            slice_write::advance(num_dims, geometry, dst_stick_id);
        }
        noc.async_write_barrier();
        input.pop_front(num_read_per_barrier);
    }
}
