// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/pack_untilize.h"
#include "api/compute/transpose.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

template <int BATCH_SIZE, std::uint32_t input_id, std::uint32_t transpose_id>
FORCE_INLINE void transpose(DataflowBuffer& input, DataflowBuffer& transpose_output) {
    input.wait_front(BATCH_SIZE);

    tile_regs_acquire();
    for (std::uint32_t i = 0; i < BATCH_SIZE; i++) {
        transpose_tile(input_id, i, i);
    }
    tile_regs_commit();
    input.pop_front(BATCH_SIZE);

    transpose_output.reserve_back(BATCH_SIZE);
    tile_regs_wait();
    pack_untilize_dest<1>(transpose_id, BATCH_SIZE);
    tile_regs_release();

    transpose_output.push_back(BATCH_SIZE);
}
TT_KERNEL void convert_to_chw(uint32_t total_tiles) {
    constexpr int BATCH_SIZE = 8;
    const std::uint32_t num_batches = total_tiles / BATCH_SIZE;
    const std::uint32_t leftover = total_tiles % BATCH_SIZE;
    DataflowBuffer input(dfb::input);
    DataflowBuffer transpose_output(dfb::transpose);

    compute_kernel_hw_startup(dfb::input, dfb::transpose);
    pack_untilize_init(dfb::input, dfb::transpose);
    transpose_init(dfb::input);

    pack_untilize_dest_init<1>(dfb::transpose);

    for (std::uint32_t i = 0; i < num_batches; i++) {
        transpose<BATCH_SIZE, dfb::input, dfb::transpose>(input, transpose_output);
    }

    for (std::uint32_t idx = 0; idx < leftover; idx++) {
        transpose<1, dfb::input, dfb::transpose>(input, transpose_output);
    }
    pack_untilize_uninit(dfb::transpose);
}
