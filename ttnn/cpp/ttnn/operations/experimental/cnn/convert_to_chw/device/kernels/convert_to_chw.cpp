// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/pack_untilize.h"
#include "api/compute/transpose.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

template <int BATCH_SIZE>
FORCE_INLINE void transpose(
    DataflowBuffer& input, DataflowBuffer& transpose_output, uint32_t input_id, uint32_t transpose_output_id) {
    input.wait_front(BATCH_SIZE);

    tile_regs_acquire();
    for (std::uint32_t i = 0; i < BATCH_SIZE; i++) {
        transpose_tile(input_id, i, i);
    }
    tile_regs_commit();
    input.pop_front(BATCH_SIZE);

    transpose_output.reserve_back(BATCH_SIZE);
    tile_regs_wait();
    pack_untilize_dest<1>(transpose_output_id, BATCH_SIZE);
    tile_regs_release();

    transpose_output.push_back(BATCH_SIZE);
}
TT_KERNEL void compute(uint32_t tiles_per_core) {
    constexpr int BATCH_SIZE = 8;
    const std::uint32_t num_batches = tiles_per_core / BATCH_SIZE;
    const std::uint32_t leftover = tiles_per_core % BATCH_SIZE;
    DataflowBuffer input(dfb::input);
    DataflowBuffer transpose_output(dfb::transpose);
    const uint32_t input_id = input.get_id();
    const uint32_t transpose_output_id = transpose_output.get_id();

    compute_kernel_hw_startup(input_id, transpose_output_id);
    pack_untilize_init(input_id, transpose_output_id);
    transpose_init(input_id);

    pack_untilize_dest_init<1>(transpose_output_id);

    for (std::uint32_t i = 0; i < num_batches; i++) {
        transpose<BATCH_SIZE>(input, transpose_output, input_id, transpose_output_id);
    }

    for (std::uint32_t idx = 0; idx < leftover; idx++) {
        transpose<1>(input, transpose_output, input_id, transpose_output_id);
    }
    pack_untilize_uninit(transpose_output_id);
}
