// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 op-private copy of writer_unary_stick_layout_split_rows_multicore.cpp (default multi-core
// interleaved untilize-with-unpadding writer). The legacy kernel is still consumed positionally by the
// un-migrated variants, so the migrated multi-core interleaved factory carries its own copy here. Only the
// binding mechanism changed:
//   - the output CB id comes from the DFB consumer token (dfb::cb_id_out0)
//   - the destination address comes from the TensorAccessor binding (ta::dst_args)
//   - the FLOAT32_DTYPE flag and unpadded_X_size come from named compile-time args (args::)
//   - padded_X_size / start_stick_id / n_block_reps are named runtime args (args::)
//   - the variable-length tail of per-block-rep quintuples (n_data, n_mixed, n_pads, times, repeat_count)
//     is read as positional runtime varargs (get_arg_addr), which begin right after the three named
//     runtime args. The block-rep write/pad loop is preserved verbatim.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr uint32_t cb_id_out0 = dfb::cb_id_out0;
    constexpr uint32_t tile_height = 32;

    const uint32_t padded_X_size = get_arg(args::padded_X_size);
    const uint32_t start_stick_id = get_arg(args::start_stick_id);
    const uint32_t n_block_reps = get_arg(args::n_block_reps);

    constexpr bool FLOAT32_DTYPE = get_arg(args::float32_dtype) == 1;
    constexpr uint32_t unpadded_X_size = get_arg(args::unpadded_X_size);

    // The three named runtime args occupy positions 0..2; the block-rep varargs begin at position 3.
    volatile tt_l1_ptr uint32_t* block_reps = (tt_l1_ptr uint32_t*)(get_arg_addr(3));

    const uint32_t num_tiles_per_row = padded_X_size >> (FLOAT32_DTYPE ? 7 : 6);

    const auto s = TensorAccessor(ta::dst_args);

    Noc noc;
    CircularBuffer cb_out0(cb_id_out0);

    auto pop_blocks = [&](uint32_t num_blocks) {
        for (uint32_t i = 0; i < num_blocks; i++) {
            cb_out0.wait_front(num_tiles_per_row);
            cb_out0.pop_front(num_tiles_per_row);
        }
    };

    auto write_block = [&](uint32_t base_stick_id, uint32_t num_rows) {
        uint32_t padding_rows = (tile_height - num_rows) & 31;
        bool has_rows = (num_rows + padding_rows) > 0;

        cb_out0.wait_front(num_tiles_per_row * has_rows);
        uint32_t l1_read_addr = cb_out0.get_read_ptr();
        for (uint32_t k = 0; k < num_rows; k++) {
            CoreLocalMem<uint32_t> src(l1_read_addr);
            noc.async_write(
                src, s, unpadded_X_size, {.offset_bytes = 0}, {.page_id = base_stick_id + k, .offset_bytes = 0});

            noc.async_write_barrier();
            l1_read_addr += padded_X_size;
        }
        cb_out0.pop_front(num_tiles_per_row * has_rows);
    };

    uint32_t stick_id = start_stick_id;
    uint32_t rt_arg_idx = 0;
    uint32_t count = 1;
    constexpr int32_t n_mixed_idx = 1;
    constexpr int32_t n_pad_idx = 2;
    constexpr int32_t times_idx = 3;
    constexpr uint32_t repeat_ct_idx = 4;
    constexpr int32_t num_rt_idx = 5;

    for (uint32_t block_rep_idx = 0; block_rep_idx < n_block_reps; ++block_rep_idx) {
        const uint32_t repeat_count = block_reps[rt_arg_idx + repeat_ct_idx];
        const uint32_t n_data = block_reps[rt_arg_idx];                 // number of full tile-rows
        const uint32_t n_mixed = block_reps[rt_arg_idx + n_mixed_idx];  // number of rows in a partially filled tile-row
        const uint32_t n_pads = block_reps[rt_arg_idx + n_pad_idx];     // number of padding tile-rows
        const uint32_t times = block_reps[rt_arg_idx + times_idx];  // number of times the pattern of tile-rows repeats
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
