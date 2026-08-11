// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/pack_untilize.h"
#include "api/compute/transpose.h"
#include "api/dataflow/circular_buffer.h"

template <int BATCH_SIZE>
FORCE_INLINE void transpose(std::uint32_t cb_in, std::uint32_t cb_out) {
    CircularBuffer cb_in_cb(cb_in);
    CircularBuffer cb_out_cb(cb_out);
    cb_in_cb.wait_front(BATCH_SIZE);

    tile_regs_acquire();
    for (std::uint32_t i = 0; i < BATCH_SIZE; i++) {
        transpose_tile(cb_in, i, i);
    }
    tile_regs_commit();
    cb_in_cb.pop_front(BATCH_SIZE);

    cb_out_cb.reserve_back(BATCH_SIZE);
    tile_regs_wait();
    pack_untilize_dest<1>(cb_out, BATCH_SIZE);
    tile_regs_release();

    cb_out_cb.push_back(BATCH_SIZE);
}
void kernel_main() {
    constexpr int BATCH_SIZE = 8;
    const std::uint32_t total_tiles = get_arg_val<std::uint32_t>(0);
    const std::uint32_t num_batches = total_tiles / BATCH_SIZE;
    const std::uint32_t leftover = total_tiles % BATCH_SIZE;
    constexpr std::uint32_t cb_in = get_compile_time_arg_val(0);
    constexpr std::uint32_t cb_transpose_in = get_compile_time_arg_val(1);

    compute_kernel_hw_startup(cb_in, cb_transpose_in);
    pack_untilize_init(cb_in, cb_transpose_in);
    transpose_init(cb_in);

    pack_untilize_dest_init<1>(cb_transpose_in);

    for (std::uint32_t i = 0; i < num_batches; i++) {
        transpose<BATCH_SIZE>(cb_in, cb_transpose_in);
    }

    for (std::uint32_t idx = 0; idx < leftover; idx++) {
        transpose<1>(cb_in, cb_transpose_in);
    }
    pack_untilize_uninit(cb_transpose_in);
}
