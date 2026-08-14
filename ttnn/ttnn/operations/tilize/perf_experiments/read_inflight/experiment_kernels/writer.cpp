// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// read_inflight bake-off writer — HELD CONSTANT across every arm.
//
//   P_LOCAL_SHARD (1): the output CB is ALIASED on this core's resident L1
//       shard, so there is nothing to send. The writer only drains the CB, so
//       the CB still has exactly one consumer. This is the focus topology.
//   P_ACCESSOR (0): interleaved destination — one whole tile page per transfer,
//       one write barrier per block (master.md B5 + B7), same as the op.

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_out = 16;

    constexpr uint32_t placement = get_compile_time_arg_val(0);  // 0 accessor / 1 local shard
    constexpr uint32_t wt_chunk = get_compile_time_arg_val(1);
    constexpr uint32_t wt = get_compile_time_arg_val(2);
    constexpr uint32_t out_tile_bytes = get_compile_time_arg_val(3);
    constexpr auto dst_args = TensorAccessorArgs<4>();

    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_blocks = get_arg_val<uint32_t>(1);
    const uint32_t tile_row0 = get_arg_val<uint32_t>(2);
    const uint32_t tile_col0 = get_arg_val<uint32_t>(3);

    if (num_blocks == 0) {
        return;
    }

    if constexpr (placement == 1) {
        for (uint32_t b = 0; b < num_blocks; ++b) {
            cb_wait_front(cb_out, wt_chunk);
            cb_pop_front(cb_out, wt_chunk);
        }
        return;
    }

    const auto acc = TensorAccessor(dst_args, dst_addr);
    for (uint32_t b = 0; b < num_blocks; ++b) {
        cb_wait_front(cb_out, wt_chunk);
        uint32_t l1 = get_read_ptr(cb_out);
        const uint32_t tile0 = (tile_row0 + b) * wt + tile_col0;
        for (uint32_t k = 0; k < wt_chunk; ++k) {
            noc_async_write(l1, acc.get_noc_addr(tile0 + k), out_tile_bytes);
            l1 += out_tile_bytes;
        }
        noc_async_write_barrier();
        cb_pop_front(cb_out, wt_chunk);
    }
}
