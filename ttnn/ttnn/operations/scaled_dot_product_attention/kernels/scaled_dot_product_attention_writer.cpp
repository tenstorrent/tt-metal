// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
void kernel_main() {
    constexpr uint32_t cb_out=tt::CBIndex::c_17;
    uint32_t dst_addr=get_arg_val<uint32_t>(0), total_tiles=get_arg_val<uint32_t>(1);
    constexpr auto dst_args=TensorAccessorArgs<0>();
    uint32_t tile_bytes=get_tile_size(cb_out);
    const auto accessor=TensorAccessor(dst_args,dst_addr,tile_bytes);
    for (uint32_t i=0; i<total_tiles; ++i) {
        cb_wait_front(cb_out,1);
        noc_async_write_tile(i,accessor,get_read_ptr(cb_out));
        noc_async_write_barrier();
        cb_pop_front(cb_out,1);
    }
}
