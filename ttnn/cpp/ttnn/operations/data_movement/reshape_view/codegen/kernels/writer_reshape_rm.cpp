// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc.h"
#include "ttnn/operations/data_movement/common/kernels/codegen/sequencers.h"

void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t num_reads = get_arg_val<uint32_t>(1);
    uint32_t num_sticks_per_read = get_arg_val<uint32_t>(2);
    uint32_t num_sticks_per_cb_push = get_arg_val<uint32_t>(3);
    uint32_t start_stick = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_out0 = get_compile_time_arg_val(0);
    constexpr uint32_t new_stick_size = get_compile_time_arg_val(1);
    constexpr uint32_t aligned_page_size = get_compile_time_arg_val(2);
    constexpr auto dst_args = TensorAccessorArgs<3>();
    constexpr uint32_t noc_max_burst_bytes = get_compile_time_arg_val(dst_args.next_compile_time_args_offset());

    const auto s = TensorAccessor(dst_args, dst_addr, aligned_page_size);

    Noc noc;
    CircularBuffer out_buffer(cb_out0);

    auto st = seq_identity_init(start_stick);
    for (uint32_t iter = 0; iter < num_reads; ++iter) {
        out_buffer.wait_front(num_sticks_per_cb_push);
        uint32_t l1_read_offset = 0;

        for (uint32_t i = 0; i < num_sticks_per_read; ++i) {
            const uint32_t i_stick = seq_identity_next(st);
            // A stick can exceed the NOC transaction ceiling, so it goes out in bursts.
            uint32_t remaining = new_stick_size;
            uint32_t offset = 0;
            while (remaining > 0) {
                uint32_t burst = remaining < noc_max_burst_bytes ? remaining : noc_max_burst_bytes;
                noc.async_write(
                    out_buffer,
                    s,
                    burst,
                    {.offset_bytes = l1_read_offset + offset},
                    {.page_id = i_stick, .offset_bytes = offset});
                remaining -= burst;
                offset += burst;
            }
            l1_read_offset += new_stick_size;
        }
        // Source reads are done once the writes are issued; a full barrier is only needed at exit.
        noc.async_writes_flushed();
        out_buffer.pop_front(num_sticks_per_cb_push);
    }
    noc.async_write_barrier();
}
