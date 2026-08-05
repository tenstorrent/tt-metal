// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/pack_untilize.h"
#include "api/compute/transpose.h"
#include "api/compute/tilize.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "api/dataflow/circular_buffer.h"

template <std::uint32_t BatchSize = 1>
FORCE_INLINE void transpose(std::uint32_t cb_in, std::uint32_t cb_out) {
    CircularBuffer cb_in_cb(cb_in);
    CircularBuffer cb_out_cb(cb_out);
    cb_in_cb.wait_front(BatchSize);

    tile_regs_acquire();
    tile_regs_wait();

    transpose_init(cb_in);
    for (std::uint32_t i = 0; i < BatchSize; i++) {
        transpose_tile(cb_in, i, i);
    }

    cb_out_cb.reserve_back(BatchSize);
    pack_untilize_dest<1>(cb_out, BatchSize);

    tile_regs_commit();
    tile_regs_release();

    cb_out_cb.push_back(BatchSize);
    cb_in_cb.pop_front(BatchSize);
}

void kernel_main() {
    constexpr std::uint32_t cb_in_batch = get_compile_time_arg_val(0);
    constexpr std::uint32_t cb_tiled_in = get_compile_time_arg_val(1);
    constexpr std::uint32_t cb_transpose_in0 = get_compile_time_arg_val(2);
    constexpr std::uint32_t cb_transpose_in1 = get_compile_time_arg_val(3);
    constexpr std::uint32_t total_tiles_per_block = get_compile_time_arg_val(4);
    constexpr std::uint32_t total_sticks_per_block = get_compile_time_arg_val(5);
    constexpr std::uint32_t total_num_blocks = get_compile_time_arg_val(6);

    compute_kernel_hw_startup(cb_in_batch, cb_tiled_in);

    for (std::uint32_t block_idx = 0; block_idx < total_num_blocks; block_idx++) {
        compute_kernel_lib::tilize<
            total_tiles_per_block,
            cb_in_batch,
            cb_tiled_in,
            compute_kernel_lib::tilize_config::InitUninitMode::InitAndUninit,
            compute_kernel_lib::tilize_config::WaitMode::WaitBlock,
            compute_kernel_lib::tilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure>(
            1, total_sticks_per_block);

        pack_untilize_init(cb_in_batch, cb_transpose_in0);
        transpose_init(cb_in_batch);
        pack_untilize_dest_init<1>(cb_in_batch);

        for (std::uint32_t idx = 0; idx < total_tiles_per_block; idx++) {
            const std::uint32_t cb_transpose_in = idx % 2 == 0 ? cb_transpose_in0 : cb_transpose_in1;
            transpose<1>(cb_tiled_in, cb_transpose_in);
        }
        pack_untilize_uninit(cb_transpose_in0);
    }
}
