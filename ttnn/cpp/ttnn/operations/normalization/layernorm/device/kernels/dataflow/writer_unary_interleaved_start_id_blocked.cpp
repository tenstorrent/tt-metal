// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "ttnn/operations/normalization/kernel_util/generic/blocked_range.h"

namespace generic = norm::kernel_util::generic;

void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t Wt = get_arg_val<uint32_t>(1);
    uint32_t num_tile_rows = get_arg_val<uint32_t>(2);
    uint32_t tile_offset = get_arg_val<uint32_t>(3);

    constexpr uint32_t blk = get_compile_time_arg_val(0);  // needed for correctness of softmax/LN kernels
    constexpr auto dst_args = TensorAccessorArgs<1>();

    constexpr uint32_t cb_id_out0 = tt::CBIndex::c_16;
    constexpr uint32_t onetile = 1;
    const uint32_t tile_bytes = get_tile_size(cb_id_out0);

    const auto s = TensorAccessor(dst_args, dst_addr, tile_bytes);

    uint32_t tile_id = tile_offset;
    for (uint32_t h = 0; h < num_tile_rows; h++) {
        for (auto block : generic::blocks(Wt, blk)) {
            cb_wait_front(cb_id_out0, block.size());
            uint32_t l1_read_addr = get_read_ptr(cb_id_out0);
            for (auto i : block.local()) {
                noc_async_write_tile(tile_id, s, l1_read_addr);
                tile_id++;
                l1_read_addr += tile_bytes;
            }
            noc_async_write_barrier();
            cb_pop_front(cb_id_out0, block.size());
        }
    }
}
