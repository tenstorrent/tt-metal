// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {
    const uint32_t dst_addr = get_arg_val<uint32_t>(1);
    const uint32_t num_total_W = get_arg_val<uint32_t>(3);
    const uint32_t num_total_Z = get_arg_val<uint32_t>(5);
    const uint32_t num_total_Y = get_arg_val<uint32_t>(7);
    const uint32_t num_total_X = get_arg_val<uint32_t>(9);
    const uint32_t padded_X_nbytes = get_arg_val<uint32_t>(11);
    const uint32_t start_dst_stick_id = get_arg_val<uint32_t>(17);
    const uint32_t start_dst_stick_wi = get_arg_val<uint32_t>(19);
    const uint32_t num_local_Y = get_arg_val<uint32_t>(21);
    const uint32_t num_local_unpadded_Y = get_arg_val<uint32_t>(22);
    const uint32_t full_padded_X_nbytes = get_arg_val<uint32_t>(24);
    const uint32_t dst_stick_offset = get_arg_val<uint32_t>(25);  // == start_src_stick_wi * elem_size
    const uint32_t num_local_W = get_arg_val<uint32_t>(26);

    constexpr uint32_t page_size = get_compile_time_arg_val(1);
    constexpr auto src_args = TensorAccessorArgs<2>();
    constexpr auto dst_args = TensorAccessorArgs<src_args.next_compile_time_args_offset()>();

    constexpr uint32_t cb_id = tt::CBIndex::c_0;

    const auto s1 = TensorAccessor(dst_args, dst_addr, page_size);

    uint32_t dst_stick_id = start_dst_stick_id;
    uint32_t dst_stick_wi = start_dst_stick_wi;
    for (uint32_t w = 0; w < num_local_W; ++w) {
        for (uint32_t z = 0; z < num_total_Z; ++z) {
            for (uint32_t y = 0; y < num_local_Y; ++y) {
                // DPRINT << "WR: " << w << ", " << z << ", " << y << ENDL();
                cb_wait_front(cb_id, 1);
                uint32_t l1_addr = get_read_ptr(cb_id);
                uint64_t dst_noc_addr = get_noc_addr(dst_stick_id, s1, dst_stick_offset);
                noc_async_write(l1_addr, dst_noc_addr, padded_X_nbytes);
                noc_async_write_barrier();
                ++dst_stick_id;
                cb_pop_front(cb_id, 1);
            }
        }
    }
}
