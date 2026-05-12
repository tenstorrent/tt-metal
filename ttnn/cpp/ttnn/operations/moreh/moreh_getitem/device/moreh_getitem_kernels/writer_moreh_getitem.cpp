// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "experimental/noc.h"
#include "experimental/circular_buffer.h"
#include "experimental/tensor.h"

void kernel_main() {
    uint32_t i = 0;
    // buffer
    uint32_t dst_addr = get_arg_val<uint32_t>(i++);

    // output
    uint32_t output_stick_size = get_arg_val<uint32_t>(i++);

    // etc
    uint32_t start_id = get_arg_val<uint32_t>(i++);
    uint32_t num_sticks = get_arg_val<uint32_t>(i++);

    constexpr uint32_t cb_id_out = tt::CBIndex::c_0;

    constexpr auto dst_args = TensorAccessorArgs<0>();
    // Third argument page_size from runtime args overrides TensorAccessorArgs::AlignedPageSize, which may be stale on
    // program cache hits.
    const auto s0 = TensorAccessor(dst_args, dst_addr, output_stick_size);

    experimental::Noc noc;
    experimental::CircularBuffer cb_out_obj(cb_id_out);

    uint32_t end_id = start_id + num_sticks;
    for (uint32_t i = start_id; i < end_id; ++i) {
        cb_out_obj.wait_front(1);
        noc.async_write(cb_out_obj, s0, output_stick_size, {.offset_bytes = 0}, {.page_id = i});
        noc.async_write_barrier();
        cb_out_obj.pop_front(1);
    }
}
