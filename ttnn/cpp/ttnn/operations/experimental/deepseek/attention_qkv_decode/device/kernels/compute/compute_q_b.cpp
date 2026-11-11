// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/experimental/custom_mm.h"
#include "api/compute/experimental/pack_block.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"
#include "qkv_epilogue.hpp"

// q = rms_scale(q_a) * ((q_a * gamma) @ w), Nc tiles, the head's trailing rope tiles rotated (do_rope).
//
// The row carries q_a unnormalized, and q_a times gamma from scaled_q_a_first: the RMSNorm scale, one
// scalar, commutes with the matmul, so it is applied to the Nc results rather than to q_a's tiles.
using namespace ckernel;
void kernel_main() {
    constexpr uint32_t q_lora_tiles = get_arg(args::q_lora_tiles);
    constexpr uint32_t row_tiles = get_arg(args::row_tiles);
    constexpr uint32_t scaled_q_a_first = get_arg(args::scaled_q_a_first);
    constexpr uint32_t nc_tiles = get_arg(args::nc_tiles);
    constexpr uint32_t inv_n_bits = get_arg(args::inv_n_bits);
    constexpr uint32_t eps_bits = get_arg(args::eps_bits);
    constexpr uint32_t rope_tiles = 2;
    const bool do_rope = get_arg(args::do_rope) != 0;

    constexpr uint32_t row_cb_id = dfb::qkv_row;
    constexpr uint32_t w_cb_id = dfb::w;
    constexpr uint32_t mm_out_cb_id = dfb::mm_out;
    constexpr uint32_t scale_cb_id = dfb::scale;
    constexpr uint32_t tmp_cb_id = dfb::tmp;
    constexpr uint32_t out_cb_id = dfb::out;
    constexpr auto layout = CustomMmDestLayout::DenseSplitAcc;
    static_assert(nc_tiles % 2 == 0 && nc_tiles >= rope_tiles && nc_tiles <= 16);

    CircularBuffer row_cb(row_cb_id);
#ifdef W_PIPE
    // Constructing w from its relay token snaps it to the pipe's cursor, which carries over from the
    // previous launch; a CircularBuffer over the bare id would restart at the ring's base.
    DataflowBuffer w_relay(dfb::w);
#endif
    CircularBuffer w_cb(w_cb_id);
    CircularBuffer mm_out_cb(mm_out_cb_id);
    CircularBuffer scale_cb(scale_cb_id);
    CircularBuffer tmp_cb(tmp_cb_id);
    CircularBuffer out_cb(out_cb_id);

    compute_kernel_hw_startup<SrcOrder::Reverse>(row_cb_id, w_cb_id, mm_out_cb_id);
    row_cb.wait_front(row_tiles);
    w_cb.wait_front(q_lora_tiles * nc_tiles);
    custom_mm_block_init_short<false, layout>(row_cb_id, w_cb_id, mm_out_cb_id, nc_tiles);
    pack_block_contiguous_init(mm_out_cb_id);
    mm_out_cb.reserve_back(nc_tiles);
    tile_regs_acquire();
    custom_mm_block<layout, true>(row_cb_id, w_cb_id, scaled_q_a_first, 0, 0, q_lora_tiles, nc_tiles);
    tile_regs_commit();
    tile_regs_wait();
    pack_block_contiguous(0, mm_out_cb_id, nc_tiles);
    tile_regs_release();
    custom_mm_block_uninit<layout>(mm_out_cb_id);
    mm_out_cb.push_back(nc_tiles);
    w_cb.pop_front(q_lora_tiles * nc_tiles);

    // custom_mm_block_uninit leaves a 32x32 packer stride behind; reprogram for 1x32 tiles.
    compute_kernel_hw_startup(row_cb_id, mm_out_cb_id, out_cb_id);
    rms_scale(row_cb_id, dfb::stat, scale_cb_id, 0, q_lora_tiles, inv_n_bits, eps_bits);
    row_cb.pop_front(row_tiles);

    scale_cb.wait_front(1);
    mm_out_cb.wait_front(nc_tiles);
    out_cb.reserve_back(nc_tiles);
    const uint32_t plain_tiles = do_rope ? nc_tiles - rope_tiles : nc_tiles;
    for (uint32_t t = 0; t < plain_tiles; t += 2) {
        scale_tiles<2, false>(mm_out_cb_id, t, scale_cb_id, 0, out_cb_id);
    }
    if (do_rope) {
        tmp_cb.reserve_back(rope_tiles);
        scale_tiles<rope_tiles, false>(mm_out_cb_id, nc_tiles - rope_tiles, scale_cb_id, 0, tmp_cb_id);
        tmp_cb.push_back(rope_tiles);
        rope_tiles_to<rope_tiles>(tmp_cb_id, dfb::cs, out_cb_id);
    }
    out_cb.push_back(nc_tiles);
    mm_out_cb.pop_front(nc_tiles);
    scale_cb.pop_front(1);
}
