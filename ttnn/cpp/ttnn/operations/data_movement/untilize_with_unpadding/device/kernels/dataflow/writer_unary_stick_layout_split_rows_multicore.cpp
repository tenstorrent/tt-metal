// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 (ProgramSpec) port. This kernel lives in this op's directory and is used only by
// UntilizeWithUnpaddingMultiCoreInterleavedProgramFactory, so it is ported in place (not forked).
// Logic, loop bounds and numeric paths are UNCHANGED; only the access mechanism moves to named
// bindings:
//   dst address          -> ta::dst (TensorAccessor)
//   CB id 16             -> dfb::out
//   FLOAT32_DTYPE / unpadded_X_size CTAs -> named CTAs (get_arg(args::...))
//   padded_X_size / start_stick_id / n_block_reps fixed RTAs -> named RTAs
//   per-core block-rep 5-tuples (variable length) -> runtime varargs (get_vararg)

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    // Constexpr
    constexpr uint32_t tile_height = 32;

    const uint32_t padded_X_size = get_arg(args::padded_X_size);
    const uint32_t start_stick_id = get_arg(args::start_stick_id);
    const uint32_t n_block_reps = get_arg(args::n_block_reps);

    constexpr bool FLOAT32_DTYPE = get_arg(args::float32_dtype) == 1;
    constexpr uint32_t unpadded_X_size = get_arg(args::unpadded_X_size);

    const uint32_t num_tiles_per_row = padded_X_size >> (FLOAT32_DTYPE ? 7 : 6);

    const auto s = TensorAccessor(ta::dst);

    Noc noc;
    DataflowBuffer cb_out0(dfb::out);

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
    // The per-core block-rep tuples are positional runtime varargs (run-length-compressed).
    // get_vararg(i) indexes from 0 (the named RTA section is invisibly skipped by the helper).
    uint32_t vararg_idx = 0;
    uint32_t count = 1;
    constexpr int32_t n_mixed_idx = 1;
    constexpr int32_t n_pad_idx = 2;
    constexpr int32_t times_idx = 3;
    constexpr uint32_t repeat_ct_idx = 4;
    constexpr int32_t num_rt_idx = 5;

    for (uint32_t block_rep_idx = 0; block_rep_idx < n_block_reps; ++block_rep_idx) {
        const uint32_t repeat_count = get_vararg(vararg_idx + repeat_ct_idx);
        const uint32_t n_data = get_vararg(vararg_idx);                 // number of full tile-rows
        const uint32_t n_mixed = get_vararg(vararg_idx + n_mixed_idx);  // number of rows in a partially filled tile-row
        const uint32_t n_pads = get_vararg(vararg_idx + n_pad_idx);     // number of padding tile-rows
        const uint32_t times = get_vararg(vararg_idx + times_idx);  // number of times the pattern of tile-rows repeats
        if (count == repeat_count) {
            vararg_idx = vararg_idx + num_rt_idx;
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
