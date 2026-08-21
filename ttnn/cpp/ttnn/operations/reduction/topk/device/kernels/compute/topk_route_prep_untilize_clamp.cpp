// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// topk_route_prep compute: fused untilize + clamp for ttnn.topk's large-k routing
// composite. Skeleton mirrors the standard copy_tile -> DEST -> pack_untilize_dest
// pattern (see e.g. data_movement/permute/device/kernels/compute/
// transpose_xw_rm_single_tile_size.cpp), with a unary_max pass injected while the
// tiles sit in DEST.
//
// Per block of bw tiles (bw = min(DEST capacity, tiles left in the tile-row); the
// factory passes the two possible widths as compile args): copy each tile into
// DEST, floor it at CLAMP_BITS — the fp32 bit pattern of the lowest finite bf16
// (0xFF7F0000, i.e. -3.3895313892515355e38; see the factory for the derivation and
// topk.cpp's clamp-trick contract for why — then pack_untilize the block into one
// output-CB page (32 sticks of bw*32 elements).

#include <cstdint>

#include "api/compute/compute_kernel_api.h"  // unary_max_tile(_init)
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/pack_untilize.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/dataflow_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp"  // compute_kernel_lib::DEST_AUTO_LIMIT

namespace {

template <uint32_t bw, uint32_t cb_in, uint32_t cb_out>
ALWI void clamp_untilize_block(DataflowBuffer& dfb_in, DataflowBuffer& dfb_out) {
    dfb_out.reserve_back(1);  // one output page holds the whole untilized block

    tile_regs_acquire();
    // Consume per tile at CB index 0: indexed reads past index 0 do NO wrap
    // arithmetic (llk_unpack_A computes rd_ptr + index*page and the front can
    // straddle fifo_limit once mixed bw_full/bw_last pops desynchronize the
    // read pointer on width_tiles % bw_full != 0 tensors) -- index 0 after a
    // per-tile pop is the one always-wrap-safe access pattern.
    for (uint32_t j = 0; j < bw; ++j) {
        dfb_in.wait_front(1);
        copy_tile(cb_in, 0, j);
        dfb_in.pop_front(1);
    }
    for (uint32_t j = 0; j < bw; ++j) {
        unary_max_tile(j, CLAMP_BITS);
    }
    tile_regs_commit();

    tile_regs_wait();
    pack_untilize_dest<bw>(cb_out);
    tile_regs_release();

    dfb_out.push_back(1);
}

}  // namespace

void kernel_main() {
    constexpr uint32_t bw_full = get_compile_time_arg_val(0);
    constexpr uint32_t bw_last = get_compile_time_arg_val(1);  // remainder block width, in [1, bw_full]
    constexpr uint32_t cb_in = get_compile_time_arg_val(2);
    constexpr uint32_t cb_out = get_compile_time_arg_val(3);

    // The factory sizes bw_full for bf16 half-sync DEST; keep it honest at compile time.
    static_assert(bw_full >= 1 && bw_full <= compute_kernel_lib::DEST_AUTO_LIMIT, "block width exceeds DEST capacity");
    static_assert(bw_last >= 1 && bw_last <= bw_full, "remainder block width out of range");

    const uint32_t nblocks = get_arg_val<uint32_t>(0);
    const uint32_t start_block = get_arg_val<uint32_t>(1);
    const uint32_t nblocks_per_row = get_arg_val<uint32_t>(2);

    DataflowBuffer dfb_in(cb_in);
    DataflowBuffer dfb_out(cb_out);

    compute_kernel_hw_startup(cb_in, cb_out);
    unary_op_init_common(cb_in, cb_out);
    unary_max_tile_init();
    pack_untilize_dest_init<bw_full>(cb_out);

    // Blocks are tile-row-major; only the last block of each tile-row is bw_last wide.
    uint32_t pos = start_block % nblocks_per_row;
    for (uint32_t b = 0; b < nblocks; ++b) {
        const bool last_in_row = (pos == nblocks_per_row - 1);
        if constexpr (bw_last == bw_full) {
            clamp_untilize_block<bw_full, cb_in, cb_out>(dfb_in, dfb_out);
        } else {
            if (last_in_row) {
                // The packer is configured per block width; swap it around the remainder block.
                pack_untilize_uninit(cb_out);
                pack_untilize_dest_init<bw_last>(cb_out);
                clamp_untilize_block<bw_last, cb_in, cb_out>(dfb_in, dfb_out);
                pack_untilize_uninit(cb_out);
                pack_untilize_dest_init<bw_full>(cb_out);
            } else {
                clamp_untilize_block<bw_full, cb_in, cb_out>(dfb_in, dfb_out);
            }
        }
        pos = last_in_row ? 0 : pos + 1;
    }

    pack_untilize_uninit(cb_out);
}
