// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Untilize kernel, one per untilizer core. Turns each batch of tiles its core's dataflow kernel stages into
// a batch of row-major tokens, in the buffer that core's readers take rows out of.
//
// A batch is one tile-row of the dispatched buffer: UNT_BATCH_ROWS tokens, `full_ct_dim` tiles wide. The
// rows come out a block of tiles at a time so the input window stays small -- a whole tile-row is 458 kB at
// the production shape and would not fit L1 on top of the output ring.
//
// How many batches there are is data-dependent, so the dataflow kernel works it out and leaves it in
// cb_batches before the first one arrives. Everything else this kernel needs is producer-consumer ordering
// on the two circular buffers.

#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/cb_api.h"
#include "api/compute/pack_untilize.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t cb_in_id = get_compile_time_arg_val(0);
    constexpr uint32_t cb_out_id = get_compile_time_arg_val(1);
    constexpr uint32_t cb_batches_id = get_compile_time_arg_val(2);
    constexpr uint32_t full_ct_dim = get_compile_time_arg_val(3);
    constexpr uint32_t block_ct_dim = get_compile_time_arg_val(4);
    constexpr uint32_t batch_rows = get_compile_time_arg_val(5);
    constexpr uint32_t num_blocks = full_ct_dim / block_ct_dim;

    CircularBuffer cb_in(cb_in_id);
    CircularBuffer cb_out(cb_out_id);
    CircularBuffer cb_batches(cb_batches_id);

    compute_kernel_hw_startup(cb_in_id, cb_out_id);
    pack_untilize_init<block_ct_dim, full_ct_dim>(cb_in_id, cb_out_id);

    // read_tile_value has UNPACK read L1 and broadcast to MATH and PACK, so all three TRISCs walk the same
    // number of batches. It is never popped: the dataflow kernel pushes it once and it stays resident.
    cb_batches.wait_front(1);
    const uint32_t num_batches = read_tile_value(cb_batches_id, 0, 0);

    for (uint32_t b = 0; b < num_batches; b++) {
        cb_out.reserve_back(batch_rows);
        for (uint32_t block = 0; block < num_blocks; block++) {
            cb_in.wait_front(block_ct_dim);
            pack_untilize_block<block_ct_dim, full_ct_dim>(cb_in_id, 1, cb_out_id, block);
            cb_in.pop_front(block_ct_dim);
        }
        cb_out.push_back(batch_rows);
    }

    pack_untilize_uninit(cb_out_id);
}
