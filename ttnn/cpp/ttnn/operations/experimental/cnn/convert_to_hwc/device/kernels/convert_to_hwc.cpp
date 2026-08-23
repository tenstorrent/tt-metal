// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/pack_untilize.h"
#include "api/compute/transpose.h"
#include "api/compute/tilize.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

template <std::uint32_t BatchSize, std::uint32_t input_id, std::uint32_t output_id>
FORCE_INLINE void transpose(DataflowBuffer& input, DataflowBuffer& output) {
    input.wait_front(BatchSize);

    tile_regs_acquire();
    tile_regs_wait();

    transpose_init(input_id);
    for (std::uint32_t i = 0; i < BatchSize; i++) {
        transpose_tile(input_id, i, i);
    }

    output.reserve_back(BatchSize);
    pack_untilize_dest<1>(output_id, BatchSize);

    tile_regs_commit();
    tile_regs_release();

    output.push_back(BatchSize);
    input.pop_front(BatchSize);
}

template <std::uint32_t total_tiles_per_block, std::uint32_t total_sticks_per_block, std::uint32_t total_num_blocks>
TT_KERNEL void convert_to_hwc() {
    DataflowBuffer tiled(dfb::tiled);
    DataflowBuffer transpose0(dfb::transpose0);
    DataflowBuffer transpose1(dfb::transpose1);

    compute_kernel_hw_startup(dfb::batch, dfb::tiled);

    for (std::uint32_t block_idx = 0; block_idx < total_num_blocks; block_idx++) {
        compute_kernel_lib::tilize<
            total_tiles_per_block,
            dfb::batch,
            dfb::tiled,
            compute_kernel_lib::tilize_config::InitUninitMode::InitAndUninit,
            compute_kernel_lib::tilize_config::WaitMode::WaitBlock,
            compute_kernel_lib::tilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure>(
            1, total_sticks_per_block);

        pack_untilize_init(dfb::batch, dfb::transpose0);
        transpose_init(dfb::batch);
        pack_untilize_dest_init<1>(dfb::batch);

        for (std::uint32_t idx = 0; idx < total_tiles_per_block; idx++) {
            if (idx % 2 == 0) {
                transpose<1, dfb::tiled, dfb::transpose0>(tiled, transpose0);
            } else {
                transpose<1, dfb::tiled, dfb::transpose1>(tiled, transpose1);
            }
        }
        pack_untilize_uninit(dfb::transpose0);
    }
}
