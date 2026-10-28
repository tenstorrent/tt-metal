// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/cpp/ttnn/operations/experimental/transformer/fused_partial_rope/device/kernels/dataflow/fused_partial_rope_reader_common.hpp"

// Reader for the fused partial-RoPE path. X and the output are resident L1 shards; cos / sin /
// trans_mat are DRAM-interleaved. This core streams its cos + sin slice and the single (replicated)
// trans_mat tile into their CBs for the compute kernel:
//   TILE cos/sin:       rope_Wt tiles starting at page `cos_sin_start` (one tile-row).
//   ROW_MAJOR cos/sin:  `cos_rows` row pages starting at page `cos_sin_start`; each Rd-wide row is
//                       byte-identical to rope_Wt 1x32 tiles.
void kernel_main() {
    uint32_t argrt = 0;
    const uint32_t cos_addr = get_arg_val<uint32_t>(argrt++);
    const uint32_t sin_addr = get_arg_val<uint32_t>(argrt++);
    const uint32_t trans_mat_addr = get_arg_val<uint32_t>(argrt++);
    const uint32_t cos_sin_start = get_arg_val<uint32_t>(argrt++);

    constexpr uint32_t cos_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t sin_cb_id = get_compile_time_arg_val(1);
    constexpr uint32_t trans_mat_cb_id = get_compile_time_arg_val(2);
    constexpr uint32_t rope_Wt = get_compile_time_arg_val(3);
    constexpr bool cos_rm = get_compile_time_arg_val(4) != 0;
    constexpr uint32_t cos_rows = get_compile_time_arg_val(5);
    constexpr bool trans_mat_rm = get_compile_time_arg_val(6) != 0;
    constexpr auto cos_args = TensorAccessorArgs<7>();
    constexpr auto sin_args = TensorAccessorArgs<cos_args.next_compile_time_args_offset()>();
    constexpr auto trans_mat_args = TensorAccessorArgs<sin_args.next_compile_time_args_offset()>();

    constexpr uint32_t onetile = 1;
    constexpr uint32_t cos_sin_tiles = rope_Wt * cos_rows;

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

    read_trans_mat<trans_mat_rm>(noc, trans_mat_cb, s_trans_mat, trans_mat_tile_bytes);

    cos_cb.reserve_back(cos_sin_tiles);
    sin_cb.reserve_back(cos_sin_tiles);
    uint32_t cos_l1 = cos_cb.get_write_ptr();
    uint32_t sin_l1 = sin_cb.get_write_ptr();
    if constexpr (cos_rm) {
        const uint32_t cos_row_bytes = rope_Wt * cos_tile_bytes;
        const uint32_t sin_row_bytes = rope_Wt * sin_tile_bytes;
        for (uint32_t r = 0; r < cos_rows; ++r) {
            noc.async_read(s_cos, CoreLocalMem<uint32_t>(cos_l1), cos_row_bytes, {.page_id = cos_sin_start + r}, {});
            noc.async_read(s_sin, CoreLocalMem<uint32_t>(sin_l1), sin_row_bytes, {.page_id = cos_sin_start + r}, {});
            cos_l1 += cos_row_bytes;
            sin_l1 += sin_row_bytes;
        }
    } else {
        for (uint32_t j = 0; j < rope_Wt; ++j) {
            noc.async_read(s_cos, CoreLocalMem<uint32_t>(cos_l1), cos_tile_bytes, {.page_id = cos_sin_start + j}, {});
            noc.async_read(s_sin, CoreLocalMem<uint32_t>(sin_l1), sin_tile_bytes, {.page_id = cos_sin_start + j}, {});
            cos_l1 += cos_tile_bytes;
            sin_l1 += sin_tile_bytes;
        }
    }

    noc.async_read_barrier();
    trans_mat_cb.push_back(onetile);
    cos_cb.push_back(cos_sin_tiles);
    sin_cb.push_back(cos_sin_tiles);
}
