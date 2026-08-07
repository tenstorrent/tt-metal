// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
#include "api/debug/dprint.h"

void kernel_main() {
    uint32_t dst_addr1 = get_arg_val<uint32_t>(0);
    uint32_t dst_addr2 = get_arg_val<uint32_t>(1);
    uint32_t num_tiles = get_arg_val<uint32_t>(2);
    uint32_t start_id = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_id_out = get_compile_time_arg_val(0);
    constexpr auto dst1_args = TensorAccessorArgs<1>();
    constexpr auto dst2_args = TensorAccessorArgs<dst1_args.next_compile_time_args_offset()>();

    // single-tile ublocks
    constexpr uint32_t onetile = 1;

    Noc noc;
    CircularBuffer cb_out(cb_id_out);

    const auto s1 = TensorAccessor(dst1_args, dst_addr1);
    const auto s2 = TensorAccessor(dst2_args, dst_addr2);

    uint32_t end_id = start_id + num_tiles;
    for (uint32_t i = start_id; i < end_id; ++i) {
        cb_out.wait_front(onetile);

        if (dst_addr1 != 0) {
            noc.async_write(cb_out, s1, s1.get_aligned_page_size(), {.offset_bytes = 0}, {.page_id = i});
            noc.async_write_barrier();
        }

        if (dst_addr2 != 0) {
            noc.async_write(cb_out, s2, s2.get_aligned_page_size(), {.offset_bytes = 0}, {.page_id = i});
            noc.async_write_barrier();
        }

        cb_out.pop_front(onetile);
    }
}
