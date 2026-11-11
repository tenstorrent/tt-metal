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
#ifdef IS_KV
#include "qkv_epilogue.hpp"
#else
#include "api/compute/experimental/rmsnorm.h"
#include "api/compute/tile_move_copy.h"
#endif

// tokens @ w, one 1x32 tile, for the reader to gather on the hub: a kv core's as x, a q_a core's as
// x and then x times this core's tile of q_a_norm's gamma.
//
// IS_KV: once the gathered row arrives, this core's tile of kv is normalized over all of kv's tiles in
// it, times gamma, and rotated (do_rope: it is one of kv's trailing rope tiles), for the writer to put
// in the ring.
using namespace ckernel;
void kernel_main() {
    constexpr uint32_t k_tiles = get_arg(args::k_tiles);
    constexpr uint32_t in0_cb_id = dfb::in0;
    constexpr uint32_t w_cb_id = dfb::w;
    constexpr uint32_t x_cb_id = dfb::x;
#ifdef IS_KV
    constexpr uint32_t mm_cb_id = x_cb_id;
#else
    constexpr uint32_t mm_cb_id = dfb::mm_out;
#endif
    constexpr auto layout = CustomMmDestLayout::DenseSplitAcc;

    CircularBuffer in0_cb(in0_cb_id);
#ifdef W_PIPE
    // Constructing w from its relay token snaps it to the pipe's cursor, which carries over from the
    // previous launch; a CircularBuffer over the bare id would restart at the ring's base.
    DataflowBuffer w_relay(dfb::w);
#endif
    CircularBuffer w_cb(w_cb_id);
    CircularBuffer mm_cb(mm_cb_id);

    compute_kernel_hw_startup<SrcOrder::Reverse>(in0_cb_id, w_cb_id, mm_cb_id);
    in0_cb.wait_front(k_tiles);
    w_cb.wait_front(k_tiles);
    custom_mm_block_init_short<false, layout>(in0_cb_id, w_cb_id, mm_cb_id, 1);
    pack_block_contiguous_init(mm_cb_id);
    mm_cb.reserve_back(1);
    tile_regs_acquire();
    custom_mm_block<layout, true>(in0_cb_id, w_cb_id, 0, 0, 0, k_tiles, 1);
    tile_regs_commit();
    tile_regs_wait();
    pack_block_contiguous(0, mm_cb_id, 1);
    tile_regs_release();
    custom_mm_block_uninit<layout>(mm_cb_id);
    mm_cb.push_back(1);
    in0_cb.pop_front(k_tiles);
    w_cb.pop_front(k_tiles);

#ifndef IS_KV
    constexpr uint32_t gamma_cb_id = dfb::q_a_gamma;
    CircularBuffer gamma_cb(gamma_cb_id);
    CircularBuffer x_cb(x_cb_id);
    // custom_mm_block_uninit leaves a 32x32 packer stride behind; reprogram for 1x32 tiles.
    compute_kernel_hw_startup(mm_cb_id, gamma_cb_id, x_cb_id);
    mm_cb.wait_front(1);
    gamma_cb.wait_front(1);
    x_cb.reserve_back(2);
    tile_regs_acquire();
    copy_init(mm_cb_id);
    copy_tile(mm_cb_id, 0, 0);
    copy_tile(mm_cb_id, 0, 1);
    // HiFi4: the program's LoFi would truncate an operand of the product.
    reconfig_data_format_srca(mm_cb_id, gamma_cb_id);
    mul_reuse_dest_init_fidelity<EltwiseBinaryReuseDestType::DEST_TO_SRCA, MathFidelity::HiFi4>(gamma_cb_id);
    mul_reuse_dest_tiles_fidelity<EltwiseBinaryReuseDestType::DEST_TO_SRCA, MathFidelity::HiFi4>(gamma_cb_id, 0, 1);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, x_cb_id);
    pack_tile(1, x_cb_id);
    tile_regs_release();
    x_cb.push_back(2);
    gamma_cb.pop_front(1);
    mm_cb.pop_front(1);
#endif

#ifdef IS_KV
    constexpr uint32_t row_tiles = get_arg(args::row_tiles);
    constexpr uint32_t kv_first_tile = get_arg(args::kv_first_tile);
    constexpr uint32_t kv_tiles = get_arg(args::kv_tiles);
    constexpr uint32_t inv_n_bits = get_arg(args::inv_n_bits);
    constexpr uint32_t eps_bits = get_arg(args::eps_bits);
    const uint32_t kv_index = get_arg(args::kv_index);
    const bool do_rope = get_arg(args::do_rope) != 0;

    constexpr uint32_t row_cb_id = dfb::qkv_row;
    constexpr uint32_t gamma_cb_id = dfb::gamma;
    constexpr uint32_t scale_cb_id = dfb::scale;
    constexpr uint32_t tmp_cb_id = dfb::tmp;
    constexpr uint32_t out_cb_id = dfb::out;
    CircularBuffer row_cb(row_cb_id);
    CircularBuffer gamma_cb(gamma_cb_id);
    CircularBuffer scale_cb(scale_cb_id);
    CircularBuffer tmp_cb(tmp_cb_id);
    CircularBuffer out_cb(out_cb_id);

    // custom_mm_block_uninit leaves a 32x32 packer stride behind; reprogram for 1x32 tiles.
    compute_kernel_hw_startup(row_cb_id, gamma_cb_id, out_cb_id);
    row_cb.wait_front(row_tiles);
    gamma_cb.wait_front(1);
    rms_scale(row_cb_id, dfb::stat, scale_cb_id, kv_first_tile, kv_tiles, inv_n_bits, eps_bits);
    scale_cb.wait_front(1);
    const uint32_t norm_cb_id = do_rope ? tmp_cb_id : out_cb_id;
    CircularBuffer norm_cb(norm_cb_id);
    norm_cb.reserve_back(1);
    scale_tiles<1, true>(row_cb_id, kv_first_tile + kv_index, scale_cb_id, gamma_cb_id, norm_cb_id);
    norm_cb.push_back(1);
    if (do_rope) {
        out_cb.reserve_back(1);
        rope_tiles_to<1>(tmp_cb_id, dfb::cs, out_cb_id);
        out_cb.push_back(1);
    }
    scale_cb.pop_front(1);
    gamma_cb.pop_front(1);
    row_cb.pop_front(row_tiles);
#endif
}
