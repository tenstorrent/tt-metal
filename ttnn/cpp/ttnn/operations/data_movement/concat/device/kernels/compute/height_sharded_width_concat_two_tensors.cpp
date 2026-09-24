// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/transpose.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

template <uint32_t BatchSize = 1>
FORCE_INLINE void transpose(uint32_t dfb_in_id, uint32_t dfb_out_id, DataflowBuffer& dfb_in, DataflowBuffer& dfb_out) {
    dfb_in.wait_front(BatchSize);

    tile_regs_acquire();
    tile_regs_wait();

    dfb_out.reserve_back(BatchSize);

    transpose_init(dfb_in_id);
    for (uint32_t i = 0; i < BatchSize; i++) {
        transpose_tile(dfb_in_id, i, i);
        pack_tile(i, dfb_out_id);
    }

    tile_regs_commit();
    tile_regs_release();

    dfb_out.push_back(BatchSize);
    dfb_in.pop_front(BatchSize);
}

void kernel_main() {
    constexpr uint32_t input0_num_tiles_height = get_arg(args::input0_num_tiles_height);
    constexpr uint32_t input0_num_tiles_width = get_arg(args::input0_num_tiles_width);
    constexpr uint32_t input1_num_tiles_height = get_arg(args::input1_num_tiles_height);
    constexpr uint32_t input1_num_tiles_width = get_arg(args::input1_num_tiles_width);

    constexpr uint32_t tile_size = get_arg(args::tile_size);
    constexpr uint32_t groups = get_arg(args::groups);
    constexpr uint32_t MAX_BATCH_SIZE = get_arg(args::max_batch_size);

    // input0 / input1 arrive from the reader; the transposed halves go back to it on the transpose
    // buffers, and the concatenated result comes back on concat to be transposed out for the writer.
    DataflowBuffer input0_dfb(dfb::input0);
    DataflowBuffer input1_dfb(dfb::input1);
    DataflowBuffer input0_transpose_dfb(dfb::input0_transpose);
    DataflowBuffer input1_transpose_dfb(dfb::input1_transpose);
    DataflowBuffer concat_dfb(dfb::concat);
    DataflowBuffer output_transpose_dfb(dfb::output_transpose);

    compute_kernel_hw_startup(dfb::input0, dfb::input0_transpose);
    transpose_init(dfb::input0);

    constexpr uint32_t output_num_tiles_width = input0_num_tiles_width + input1_num_tiles_width;

    for (uint32_t i = 0; i < input0_num_tiles_height; i++) {
        reconfig_data_format_srca(dfb::input0);
        pack_reconfig_data_format(dfb::input0_transpose);
        if constexpr (input0_num_tiles_width <= MAX_BATCH_SIZE) {
            transpose<input0_num_tiles_width>(dfb::input0, dfb::input0_transpose, input0_dfb, input0_transpose_dfb);
        } else {
            for (uint32_t j = 0; j < input0_num_tiles_width; j++) {
                transpose(dfb::input0, dfb::input0_transpose, input0_dfb, input0_transpose_dfb);
            }
        }
        if constexpr (input1_num_tiles_width <= MAX_BATCH_SIZE) {
            transpose<input1_num_tiles_width>(dfb::input1, dfb::input1_transpose, input1_dfb, input1_transpose_dfb);
        } else {
            for (uint32_t j = 0; j < input1_num_tiles_width; j++) {
                transpose(dfb::input1, dfb::input1_transpose, input1_dfb, input1_transpose_dfb);
            }
        }

        reconfig_data_format_srca(dfb::concat);
        pack_reconfig_data_format(dfb::output_transpose);
        if constexpr (output_num_tiles_width <= MAX_BATCH_SIZE) {
            transpose<output_num_tiles_width>(dfb::concat, dfb::output_transpose, concat_dfb, output_transpose_dfb);
        } else {
            for (uint32_t j = 0; j < output_num_tiles_width; j++) {
                transpose(dfb::concat, dfb::output_transpose, concat_dfb, output_transpose_dfb);
            }
        }
    }
}
