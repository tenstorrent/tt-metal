// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc.h"
#include "ttnn/operations/data_movement/common/kernels/codegen/sequencers.h"

void kernel_main() {
    uint32_t dst_addr = get_common_arg_val<uint32_t>(0);
    uint32_t num_reads = get_arg_val<uint32_t>(0);
    uint32_t num_sticks_per_read = get_arg_val<uint32_t>(1);
    uint32_t num_sticks_per_cb_push = get_arg_val<uint32_t>(2);
    uint32_t start_stick = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_out0 = get_named_compile_time_arg_val("cb_out0");
    constexpr uint32_t new_stick_size = get_named_compile_time_arg_val("new_stick_size");
    constexpr uint32_t aligned_page_size = get_named_compile_time_arg_val("aligned_page_size");
    constexpr auto dst_args = TensorAccessorArgs<0>();

    const auto s = TensorAccessor(dst_args, dst_addr, aligned_page_size);

    Noc noc;
    CircularBuffer out_buffer(cb_out0);

    auto st = seq_identity_init(start_stick);
    for (uint32_t iter = 0; iter < num_reads; ++iter) {
        out_buffer.wait_front(num_sticks_per_cb_push);
        uint32_t l1_read_offset = 0;

        for (uint32_t i = 0; i < num_sticks_per_read; ++i) {
            const uint32_t i_stick = seq_identity_next(st);
            noc.async_write(
                out_buffer,
                s,
                new_stick_size,
                {.offset_bytes = l1_read_offset},
                {.page_id = i_stick, .offset_bytes = 0});
            l1_read_offset += new_stick_size;
        }
        // Source reads are done once the writes are issued; a full barrier is only needed at exit.
        noc.async_writes_flushed();
        out_buffer.pop_front(num_sticks_per_cb_push);
    }
    noc.async_write_barrier();
}
