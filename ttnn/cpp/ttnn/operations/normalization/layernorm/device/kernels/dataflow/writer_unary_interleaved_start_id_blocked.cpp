// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "ttnn/operations/normalization/kernel_util/generic/blocked_range.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
#include "api/debug/dprint.h"

namespace generic = norm::kernel_util::generic;

void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t Wt = get_arg_val<uint32_t>(1);
    uint32_t num_tile_rows = get_arg_val<uint32_t>(2);
    uint32_t tile_offset = get_arg_val<uint32_t>(3);

    constexpr uint32_t blk = get_compile_time_arg_val(0);  // needed for correctness of softmax/LN kernels
    constexpr auto dst_args = TensorAccessorArgs<1>();

    // CB index - configurable via named compile-time args for kernel chaining support
    constexpr uint32_t cb_id_out0 = get_named_compile_time_arg_val("cb_out");
    constexpr uint32_t onetile = 1;
    const uint32_t tile_bytes = get_tile_size(cb_id_out0);

    Noc noc;
    CircularBuffer cb_out0(cb_id_out0);

    const auto s = TensorAccessor(dst_args, dst_addr);

    DPRINT << "[BW " << (uint32_t)my_x[0] << "," << (uint32_t)my_y[0] << "] enter cb=" << cb_id_out0
           << " fsz=" << get_local_cb_interface(cb_id_out0).fifo_size << " rows=" << num_tile_rows << " Wt=" << Wt
           << " blk=" << blk << ENDL();

    uint32_t tile_id = tile_offset;
    for (uint32_t h = 0; h < num_tile_rows; h++) {
        for (auto block : generic::blocks(Wt, blk)) {
            DPRINT << "[BW " << (uint32_t)my_x[0] << "," << (uint32_t)my_y[0] << "] wait_front "
                   << block.full_block_size() << ENDL();
            cb_out0.wait_front(block.full_block_size());
            DPRINT << "[BW " << (uint32_t)my_x[0] << "," << (uint32_t)my_y[0] << "] got tiles" << ENDL();
            uint32_t idx = 0;
            for (auto i : block.local()) {
                noc.async_write(cb_out0, s, tile_bytes, {.offset_bytes = idx * tile_bytes}, {.page_id = tile_id});
                tile_id++;
                idx++;
            }
            noc.async_write_barrier();
            cb_out0.pop_front(block.full_block_size());
        }
    }
    DPRINT << "[BW " << (uint32_t)my_x[0] << "," << (uint32_t)my_y[0] << "] done" << ENDL();
}
