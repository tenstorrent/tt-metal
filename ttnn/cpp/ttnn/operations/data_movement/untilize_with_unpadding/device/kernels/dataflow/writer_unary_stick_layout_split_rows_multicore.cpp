// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/operations/data_movement/common/kernels/common.hpp"
#include "experimental/kernel_args.h"

void kernel_main() {
    // Constexpr
    constexpr uint32_t tile_height = 32;

    const uint32_t padded_X_size = get_arg(args::padded_X_size);
    const uint32_t start_stick_id = get_arg(args::start_stick_id);
    const uint32_t n_block_reps = get_arg(args::n_block_reps);

    constexpr bool FLOAT32_DTYPE = get_arg(args::float32_dtype) == 1;
    constexpr auto unpadded_X_size = get_arg(args::unpadded_X_size);

    const uint32_t num_tiles_per_row = padded_X_size >> (FLOAT32_DTYPE ? 7 : 6);

    // The output page the accessor walks is the per-shard page for BLOCK/WIDTH-sharded output (a
    // logical row spans multiple shards) and the full unpadded row otherwise. That page size now
    // rides on the tensor binding, so noc_async_write_sharded's multi-shard row split is fed
    // automatically -- same mechanism as the ROW_MAJOR-input factory.
    const auto s = TensorAccessor(tensor::dst);

    Noc noc;
    DataflowBuffer dfb_out0(dfb::out);

    auto pop_blocks = [&](uint32_t num_blocks) {
        for (uint32_t i = 0; i < num_blocks; i++) {
            dfb_out0.wait_front(num_tiles_per_row);
            dfb_out0.pop_front(num_tiles_per_row);
        }
    };

    auto write_block = [&](uint32_t base_stick_id, uint32_t num_rows) {
        uint32_t padding_rows = (tile_height - num_rows) & 31;
        bool has_rows = (num_rows + padding_rows) > 0;

        dfb_out0.wait_front(num_tiles_per_row * has_rows);
        uint32_t l1_read_addr = dfb_out0.get_read_ptr();
        for (uint32_t k = 0; k < num_rows; k++) {
            // Splits the write across shards for B/W-sharded outputs; falls through to a single
            // noc_async_write for interleaved / HEIGHT-sharded.
            tt::data_movement::common::noc_async_write_sharded(
                noc, l1_read_addr, s, base_stick_id + k, /*offset=*/0, /*size=*/unpadded_X_size);

            noc.async_write_barrier();
            l1_read_addr += padded_X_size;
        }
        dfb_out0.pop_front(num_tiles_per_row * has_rows);
    };

    uint32_t stick_id = start_stick_id;
    // Vararg block: n_block_reps groups of 5, each {n_data, n_mixed, n_pads, times, repeat_count}.
    // The group count varies per node with the block assignment, so this stays positional.
    uint32_t rt_arg_idx = 0;
    uint32_t count = 1;
    constexpr int32_t n_mixed_idx = 1;
    constexpr int32_t n_pad_idx = 2;
    constexpr int32_t times_idx = 3;
    constexpr uint32_t repeat_ct_idx = 4;
    constexpr int32_t num_rt_idx = 5;

    for (uint32_t block_rep_idx = 0; block_rep_idx < n_block_reps; ++block_rep_idx) {
        const uint32_t repeat_count = get_vararg(rt_arg_idx + repeat_ct_idx);
        const uint32_t n_data = get_vararg(rt_arg_idx);                 // number of full tile-rows
        const uint32_t n_mixed = get_vararg(rt_arg_idx + n_mixed_idx);  // number of rows in a partially filled tile-row
        const uint32_t n_pads = get_vararg(rt_arg_idx + n_pad_idx);     // number of padding tile-rows
        const uint32_t times = get_vararg(rt_arg_idx + times_idx);  // number of times the pattern of tile-rows repeats
        if (count == repeat_count) {
            rt_arg_idx = rt_arg_idx + num_rt_idx;
            count = 1;
        } else {
            count++;
        }

        for (uint32_t t = 0; t < times; ++t) {
            for (uint32_t y_t = 0; y_t < n_data; y_t++) {
                write_block(stick_id, tile_height);
                stick_id += tile_height;
            }

            write_block(stick_id, n_mixed);
            stick_id += n_mixed;

            pop_blocks(n_pads);
        }
    }
}
