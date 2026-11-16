// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/compute_kernel_api.h"
#include "api/compute/experimental/custom_mm.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/pack.h"
#include "api/dataflow/circular_buffer.h"
#include "api/debug/kernel_profiler.hpp"
#include "../persistent.h"

// The persistent op's sharded router, a partial core (see dataflow/router_partial.cpp): this core's K slice of
// its chip's gate column -- k_slice tiles of cb_rw, rows [k_start, +k_slice) of the column -- against the same
// tiles of the row in cb_rin, one custom_mm (LoFi, 16-bit DEST, as the unsharded router's), packed as one 1x32
// partial of the chip's 32 experts' gate logits into cb_part. The reducer core sums the chip's partials.
//
// Compile-time args:
//   0: cb_rin (the row, k_tiles 1x32 tiles)   1: k_tiles   2: cb_rw (this core's k_slice gate tiles)
//   3: cb_part   4: k_slice
//
// Runtime args: 0: k_start (this core's first K tile)
//
// FE_PERSISTENT: once per iteration; popping the row stays last (see persistent.h).

void kernel_main() {
    constexpr uint32_t cb_rin_id = get_compile_time_arg_val(0);
    constexpr uint32_t k_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t cb_rw_id = get_compile_time_arg_val(2);
    constexpr uint32_t cb_part_id = get_compile_time_arg_val(3);
    constexpr uint32_t k_slice = get_compile_time_arg_val(4);
    const uint32_t k_start = get_arg_val<uint32_t>(0);

    CircularBuffer w_cb(cb_rw_id);

    constexpr bool transpose = false;
    constexpr auto layout = CustomMmDestLayout::DefaultSplitAcc;
    // SrcOrder::Reverse: SrcA holds the weight (in1), SrcB the activation (in0).
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_rin_id, cb_rw_id, cb_part_id);

    // The tile shape the unpacker and packer are set up for (FE_NATIVE_ONE_ROW: a one-row iteration's 1x32 aliases).
    bool shape_one_row = false;
    for (uint32_t iter = 0; fe_wait_go(iter); ++iter) {
        fe_rewind_cbs();
        const bool one_row = fe_one_row();
        const uint32_t cb_rin = fe_cb(cb_rin_id, one_row);
        const uint32_t cb_part = fe_cb(cb_part_id, one_row);
        if (one_row != shape_one_row) {
            // SrcB (the row) and the packer's tile shape; custom_mm's init below takes SrcB's face rows.
            reconfig_full_operand<SrcOrder::Reverse>(cb_rin, cb_rw_id);
            pack_reconfig_data_format<true>(cb_part);
            shape_one_row = one_row;
        }
        CircularBuffer in_cb(cb_rin);
        CircularBuffer out_cb(cb_part);
        out_cb.reserve_back(1);
        in_cb.wait_front(k_tiles);
        w_cb.wait_front(k_slice);
        tile_regs_acquire();
        custom_mm_block_init_short<transpose, layout>(cb_rin, cb_rw_id, cb_part, 1);
        {
            DeviceZoneScopedN("FE_R_PART_MM");
            custom_mm_block<layout, true>(cb_rin, cb_rw_id, k_start, 0, 0, k_slice, 1);
        }
        custom_mm_block_uninit<layout>(cb_part);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_part);
        tile_regs_release();
        out_cb.push_back(1);
        w_cb.pop_front(k_slice);
        in_cb.pop_front(k_tiles);
    }
}
