// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/tensor_accessor.h"
#include "experimental/kernel_args.h"
#include "ttnn/operations/data_movement/common/kernels/common.hpp"

template <uint32_t page_offset>
TT_KERNEL void reader(
    uint32_t stick_size,
    uint32_t stick_size_offset,
    uint32_t num_sticks_per_core,
    uint32_t num_sticks_per_core_read,
    uint32_t num_read_per_barrier,
    uint32_t start_id) {
    const uint32_t input_base =
        get_common_arg_val<uint32_t>(decltype(tensor::input)::addr_crta_offset / sizeof(uint32_t));
    // Preserve the legacy runtime page-size override so same-spec cache hits bind the
    // current allocation without inheriting stale aligned-page metadata.
    const auto input_accessor = TensorAccessor(decltype(tensor::input)::args, input_base, stick_size);

    Noc noc;
    DataflowBuffer input(dfb::input);

    uint32_t i_stick = start_id;
    uint32_t sticks_read = 0;
    for (uint32_t iter = 0; iter < num_sticks_per_core_read && sticks_read < num_sticks_per_core; ++iter) {
        input.reserve_back(num_read_per_barrier);
        uint32_t l1_offset = 0;

        for (uint32_t i = 0; i < num_read_per_barrier && sticks_read < num_sticks_per_core; ++i) {
            sticks_read++;
            noc.async_read(input_accessor, input, stick_size, {.page_id = i_stick}, {.offset_bytes = l1_offset});
            if constexpr (page_offset != 0) {
                // Align the source within its DFB entry. The writer consumes from the
                // matching offset, preserving overlap-safe last-dimension placement.
                noc.async_read_barrier();
                const uint32_t l1_write_addr = input.get_write_ptr() + l1_offset;
                tt::data_movement::common::tt_memmove<false, false, false, 0>(
                    noc, l1_write_addr + page_offset, l1_write_addr, stick_size);
            }
            l1_offset += stick_size_offset;
            i_stick++;
        }
        noc.async_read_barrier();
        input.push_back(num_read_per_barrier);
    }
}
