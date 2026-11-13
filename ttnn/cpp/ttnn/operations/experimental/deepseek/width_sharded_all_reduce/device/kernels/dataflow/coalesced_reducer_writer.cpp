// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"

// Scatters this reducer core's reduced faces from cb_out to the output tensor shards.
void kernel_main() {
    constexpr uint32_t cb_out_id = get_compile_time_arg_val(0);
    constexpr uint32_t group_pages = get_compile_time_arg_val(1);

    const uint32_t output_addr = get_common_arg_val<uint32_t>(0);

    size_t arg_idx = 0;
    const uint32_t num_out_chunks = get_arg_val<uint32_t>(arg_idx++);
    tt_l1_ptr uint32_t* out_x = reinterpret_cast<tt_l1_ptr uint32_t*>(get_arg_addr(arg_idx));
    arg_idx += num_out_chunks;
    tt_l1_ptr uint32_t* out_y = reinterpret_cast<tt_l1_ptr uint32_t*>(get_arg_addr(arg_idx));
    arg_idx += num_out_chunks;
    tt_l1_ptr uint32_t* out_off = reinterpret_cast<tt_l1_ptr uint32_t*>(get_arg_addr(arg_idx));
    arg_idx += num_out_chunks;
    tt_l1_ptr uint32_t* out_len = reinterpret_cast<tt_l1_ptr uint32_t*>(get_arg_addr(arg_idx));
    arg_idx += num_out_chunks;

    Noc noc;
    CircularBuffer cb_out(cb_out_id);
    cb_out.wait_front(group_pages);
    uint32_t read_ptr = cb_out.get_read_ptr();
    for (uint32_t c = 0; c < num_out_chunks; ++c) {
        const uint32_t len = out_len[c];
        noc_async_write(read_ptr, get_noc_addr(out_x[c], out_y[c], output_addr + out_off[c]), len);
        read_ptr += len;
    }
    noc.async_write_barrier();
    cb_out.pop_front(group_pages);
}
