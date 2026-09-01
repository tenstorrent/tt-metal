// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

// Reader for the fused partial-RoPE path. X and the output are resident L1 shards; cos / sin /
// trans_mat are DRAM-interleaved. This core owns one input tile-row, so it streams its matching
// rope_Wt cos + sin tiles (starting at cos_sin_start_tile) and the single (replicated) trans_mat
// tile into their CBs for the compute kernel.
void kernel_main() {
    uint32_t argrt = 0;
    const uint32_t cos_addr = get_arg_val<uint32_t>(argrt++);
    const uint32_t sin_addr = get_arg_val<uint32_t>(argrt++);
    const uint32_t trans_mat_addr = get_arg_val<uint32_t>(argrt++);
    const uint32_t cos_sin_start_tile = get_arg_val<uint32_t>(argrt++);

    constexpr uint32_t cos_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t sin_cb_id = get_compile_time_arg_val(1);
    constexpr uint32_t trans_mat_cb_id = get_compile_time_arg_val(2);
    constexpr uint32_t rope_Wt = get_compile_time_arg_val(3);
    constexpr auto cos_args = TensorAccessorArgs<4>();
    constexpr auto sin_args = TensorAccessorArgs<cos_args.next_compile_time_args_offset()>();
    constexpr auto trans_mat_args = TensorAccessorArgs<sin_args.next_compile_time_args_offset()>();

    constexpr uint32_t onetile = 1;

    Noc noc;
    CircularBuffer cos_cb(cos_cb_id);
    CircularBuffer sin_cb(sin_cb_id);
    CircularBuffer trans_mat_cb(trans_mat_cb_id);

    const uint32_t cos_tile_bytes = get_tile_size(cos_cb_id);
    const uint32_t sin_tile_bytes = get_tile_size(sin_cb_id);
    const uint32_t trans_mat_tile_bytes = get_tile_size(trans_mat_cb_id);
    const auto s_cos = TensorAccessor(cos_args, cos_addr);
    const auto s_sin = TensorAccessor(sin_args, sin_addr);
    const auto s_trans_mat = TensorAccessor(trans_mat_args, trans_mat_addr);

    // trans_mat: single replicated tile (page 0), reused for the whole compute.
    trans_mat_cb.reserve_back(onetile);
    noc.async_read(
        s_trans_mat, CoreLocalMem<uint32_t>(trans_mat_cb.get_write_ptr()), trans_mat_tile_bytes, {.page_id = 0}, {});

    // cos / sin: this core's rope tile-row.
    cos_cb.reserve_back(rope_Wt);
    sin_cb.reserve_back(rope_Wt);
    uint32_t cos_l1 = cos_cb.get_write_ptr();
    uint32_t sin_l1 = sin_cb.get_write_ptr();
    for (uint32_t j = 0; j < rope_Wt; ++j) {
        noc.async_read(s_cos, CoreLocalMem<uint32_t>(cos_l1), cos_tile_bytes, {.page_id = cos_sin_start_tile + j}, {});
        noc.async_read(s_sin, CoreLocalMem<uint32_t>(sin_l1), sin_tile_bytes, {.page_id = cos_sin_start_tile + j}, {});
        cos_l1 += cos_tile_bytes;
        sin_l1 += sin_tile_bytes;
    }

    noc.async_read_barrier();
    trans_mat_cb.push_back(onetile);
    cos_cb.push_back(rope_Wt);
    sin_cb.push_back(rope_Wt);
}
