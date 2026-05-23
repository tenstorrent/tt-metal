// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// v3 fused SwiGLU compute kernel — PACKER_L1_ACC variant.
//
// Pattern (modelled on bmm_large_block_zm_fused_bias_activation.cpp without
// FUSE_BIAS):
//   * Each matmul phase has num_blocks K-blocks. dst is acquired/released
//     once per (sb_m, sb_n) subblock pair within each K-block.
//   * For K-blocks [0..num_blocks-2]:
//        - The K-loop accumulates the block's per-output-tile dot products
//          into a fresh dst (acquire_dst clears via dst_section_done).
//        - The packer is configured with L1_ACC=0 on block 0 (overwrite) and
//          L1_ACC=1 on block 1+ (add to existing L1). Each subblock packs
//          into mm_partials_cb, then the kernel pops what it just pushed so
//          the CB ring stays bounded. Because the CB is sized exactly to
//          one full block, the next K-block's packer writes land back in
//          the SAME L1 slots, accumulating physically into the same region.
//   * For the LAST K-block:
//        - L1_ACC stays enabled. The block's dst is added into the partials.
//        - Subblocks push into mm_partials WITHOUT popping, so after the
//          K-loop the partials CB has out_block_num_tiles tiles available
//          holding the final accumulated sum.
//   * After the K-loop, a second pass copies each subblock from partials_cb
//     into dst, optionally applies SILU on the pack, and packs into the
//     final CB (cb_gate_intermed / cb_up_intermed / cb_out).
//
// This avoids the dst-reload pattern from v2 (which was producing Inf
// outputs) by letting the packer handle cross-K-block accumulation.

#include <cstdint>

#include "api/compute/matmul.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_fused_activation.hpp"

namespace {

template <
    uint32_t in0_block_w,
    uint32_t in0_num_subblocks,
    uint32_t in0_block_num_tiles,
    uint32_t in0_subblock_num_tiles,
    uint32_t in1_num_subblocks,
    uint32_t in1_block_num_tiles,
    uint32_t in1_per_core_w,
    uint32_t num_blocks,
    uint32_t out_subblock_h,
    uint32_t out_subblock_w,
    uint32_t out_subblock_num_tiles,
    uint32_t out_block_num_tiles,
    bool apply_silu_on_final>
FORCE_INLINE void matmul_phase_v3(
    uint32_t in0_cb_id, uint32_t in1_cb_id, uint32_t partials_cb_id, uint32_t final_cb_id) {
    // Reconfig packer for partials format (previous phase's final_cb format
    // would otherwise leak). pack_reconfig_data_format (the reconfig variant)
    // does NOT reset L1_ACC — we do that explicitly below.
    PACK((pack_reconfig_data_format(partials_cb_id)));
#ifdef PACKER_L1_ACC
    PACK((llk_pack_reconfig_l1_acc(0)));  // block 0 must overwrite, not accumulate
#endif
    for (uint32_t block = 0; block < num_blocks; ++block) {
        const bool last_out = (block == num_blocks - 1);

        cb_wait_front(in0_cb_id, in0_block_num_tiles);
        cb_wait_front(in1_cb_id, in1_block_num_tiles);

        int in0_index_subblock_offset = 0;
        for (uint32_t sb_m = 0; sb_m < in0_num_subblocks; ++sb_m) {
            int in1_index_subblock_offset = 0;
            for (uint32_t sb_n = 0; sb_n < in1_num_subblocks; ++sb_n) {
                tile_regs_acquire();

                // matmul_tiles per-tile pattern (v2-style).
                int dst_index = 0;
                int in0_index_h_offset = 0;
                for (uint32_t h = 0; h < out_subblock_h; ++h) {
                    for (uint32_t w = 0; w < out_subblock_w; ++w) {
                        int in1_index_inner_dim_offset = 0;
                        for (uint32_t inner_dim = 0; inner_dim < in0_block_w; ++inner_dim) {
                            int in0_index = in0_index_subblock_offset + in0_index_h_offset + inner_dim;
                            int in1_index = in1_index_subblock_offset + in1_index_inner_dim_offset + w;
                            matmul_tiles(in0_cb_id, in1_cb_id, in0_index, in1_index, dst_index);
                            in1_index_inner_dim_offset += in1_per_core_w;
                        }
                        ++dst_index;
                    }
                    in0_index_h_offset += in0_block_w;
                }

                tile_regs_commit();
                tile_regs_wait();

                cb_reserve_back(partials_cb_id, out_subblock_num_tiles);
                for (uint32_t i = 0; i < out_subblock_num_tiles; ++i) {
                    pack_tile(i, partials_cb_id);
                }
                cb_push_back(partials_cb_id, out_subblock_num_tiles);

                tile_regs_release();
                in1_index_subblock_offset += out_subblock_w;
            }
            in0_index_subblock_offset += in0_subblock_num_tiles;
        }

        // Pop what we just pushed on all but the LAST block. This keeps the
        // partials CB ring at the same WP=RP modulo so the next K-block's
        // packer L1_ACC writes hit the same physical L1 slots.
        if (!last_out) {
            for (uint32_t s = 0; s < out_block_num_tiles; s += out_subblock_num_tiles) {
                cb_wait_front(partials_cb_id, out_subblock_num_tiles);
                cb_pop_front(partials_cb_id, out_subblock_num_tiles);
            }
        }

#ifdef PACKER_L1_ACC
        // After block 0 finishes, flip L1_ACC on so blocks 1..N-1 accumulate.
        if (block == 0) {
            PACK((llk_pack_reconfig_l1_acc(1)));
        }
#endif

        cb_pop_front(in0_cb_id, in0_block_num_tiles);
        cb_pop_front(in1_cb_id, in1_block_num_tiles);
    }

    // After the K-loop: partials_cb_id has out_block_num_tiles tiles holding
    // the final accumulated sum. Move them through dst into final_cb_id,
    // applying silu on the way if requested.
#ifdef PACKER_L1_ACC
    PACK((llk_pack_reconfig_l1_acc(0)));  // future packs (to final_cb) must overwrite
#endif
    // Packer was configured for partials_cb format during matmul. The final
    // pack lands in final_cb (different format) — reconfigure both packer
    // data format and SrcA before the copy/pack loop.
    PACK((pack_reconfig_data_format(final_cb_id)));
    // matmul puts in1 → SrcA, in0 → SrcB. Reconfigure SrcA from in1 to
    // partials so copy_tile reads partials.
    copy_tile_to_dst_init_short_with_dt(in1_cb_id, partials_cb_id);

    for (uint32_t sb = 0; sb < (out_block_num_tiles / out_subblock_num_tiles); ++sb) {
        tile_regs_acquire();
        cb_wait_front(partials_cb_id, out_subblock_num_tiles);
        for (uint32_t i = 0; i < out_subblock_num_tiles; ++i) {
            copy_tile(partials_cb_id, i, i);
        }
        cb_pop_front(partials_cb_id, out_subblock_num_tiles);

        tile_regs_commit();

        if constexpr (apply_silu_on_final) {
            apply_activation_from_pack<KernelActivation::SILU>(out_subblock_num_tiles);
        } else {
            tile_regs_wait();
        }

        cb_reserve_back(final_cb_id, out_subblock_num_tiles);
        for (uint32_t i = 0; i < out_subblock_num_tiles; ++i) {
            pack_tile(i, final_cb_id);
        }
        cb_push_back(final_cb_id, out_subblock_num_tiles);

        tile_regs_release();
    }
}

template <uint32_t out_block_num_tiles, uint32_t out_subblock_num_tiles>
FORCE_INLINE void multiply_phase_v3(uint32_t gate_cb_id, uint32_t up_cb_id, uint32_t activated_cb_id) {
    cb_wait_front(gate_cb_id, out_block_num_tiles);
    cb_wait_front(up_cb_id, out_block_num_tiles);

    // Reconfigure packer for activated format and unpacker for both
    // gate_cb (SrcA) and up_cb (SrcB). After phase 2's second pass the
    // SrcA was configured for partials_gu but SrcB still points at the
    // old cb_in0_x (bf8) from matmul — mul_tiles_init's full_init only
    // reprograms the unpack MOP, not the data formats. Without the
    // explicit reconfig SrcB reads bf16 up_intermed bytes as bf8 and the
    // multiply collapses to denormal magnitudes.
    PACK((pack_reconfig_data_format(activated_cb_id)));
    reconfig_data_format(gate_cb_id, up_cb_id);
    mul_tiles_init(gate_cb_id, up_cb_id);

    constexpr uint32_t num_subblocks = out_block_num_tiles / out_subblock_num_tiles;
    uint32_t base = 0;
    for (uint32_t sb = 0; sb < num_subblocks; ++sb) {
        tile_regs_acquire();
        for (uint32_t i = 0; i < out_subblock_num_tiles; ++i) {
            mul_tiles(gate_cb_id, up_cb_id, base + i, base + i, i);
        }
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(activated_cb_id, out_subblock_num_tiles);
        for (uint32_t i = 0; i < out_subblock_num_tiles; ++i) {
            pack_tile(i, activated_cb_id);
        }
        cb_push_back(activated_cb_id, out_subblock_num_tiles);
        tile_regs_release();
        base += out_subblock_num_tiles;
    }
    cb_pop_front(gate_cb_id, out_block_num_tiles);
    cb_pop_front(up_cb_id, out_block_num_tiles);
}

}  // namespace

void kernel_main() {
    // Phase 1 (gate)
    constexpr uint32_t g_in0_block_w = get_compile_time_arg_val(0);
    constexpr uint32_t g_in0_num_subblocks = get_compile_time_arg_val(1);
    constexpr uint32_t g_in0_block_num_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t g_in0_subblock_num_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t g_in1_num_subblocks = get_compile_time_arg_val(4);
    constexpr uint32_t g_in1_block_num_tiles = get_compile_time_arg_val(5);
    constexpr uint32_t g_in1_per_core_w = get_compile_time_arg_val(6);
    constexpr uint32_t g_num_blocks = get_compile_time_arg_val(7);
    // Phase 2 (up)
    constexpr uint32_t u_in0_block_w = get_compile_time_arg_val(8);
    constexpr uint32_t u_in0_num_subblocks = get_compile_time_arg_val(9);
    constexpr uint32_t u_in0_block_num_tiles = get_compile_time_arg_val(10);
    constexpr uint32_t u_in0_subblock_num_tiles = get_compile_time_arg_val(11);
    constexpr uint32_t u_in1_num_subblocks = get_compile_time_arg_val(12);
    constexpr uint32_t u_in1_block_num_tiles = get_compile_time_arg_val(13);
    constexpr uint32_t u_in1_per_core_w = get_compile_time_arg_val(14);
    constexpr uint32_t u_num_blocks = get_compile_time_arg_val(15);
    // Phase 4 (down)
    constexpr uint32_t d_in0_block_w = get_compile_time_arg_val(16);
    constexpr uint32_t d_in0_num_subblocks = get_compile_time_arg_val(17);
    constexpr uint32_t d_in0_block_num_tiles = get_compile_time_arg_val(18);
    constexpr uint32_t d_in0_subblock_num_tiles = get_compile_time_arg_val(19);
    constexpr uint32_t d_in1_num_subblocks = get_compile_time_arg_val(20);
    constexpr uint32_t d_in1_block_num_tiles = get_compile_time_arg_val(21);
    constexpr uint32_t d_in1_per_core_w = get_compile_time_arg_val(22);
    constexpr uint32_t d_num_blocks = get_compile_time_arg_val(23);
    // Subblock dims
    constexpr uint32_t gu_out_subblock_h = get_compile_time_arg_val(24);
    constexpr uint32_t gu_out_subblock_w = get_compile_time_arg_val(25);
    constexpr uint32_t gu_out_subblock_num_tiles = gu_out_subblock_h * gu_out_subblock_w;
    constexpr uint32_t gu_out_block_num_tiles = get_compile_time_arg_val(26);
    constexpr uint32_t d_out_subblock_h = get_compile_time_arg_val(27);
    constexpr uint32_t d_out_subblock_w = get_compile_time_arg_val(28);
    constexpr uint32_t d_out_subblock_num_tiles = d_out_subblock_h * d_out_subblock_w;
    constexpr uint32_t d_out_block_num_tiles = get_compile_time_arg_val(29);

    // CBs
    constexpr uint32_t cb_in0_x = get_named_compile_time_arg_val("cb_in0_x");
    constexpr uint32_t cb_in1_gate = get_named_compile_time_arg_val("cb_in1_gate");
    constexpr uint32_t cb_in1_up = get_named_compile_time_arg_val("cb_in1_up");
    constexpr uint32_t cb_in1_down = get_named_compile_time_arg_val("cb_in1_down");
    constexpr uint32_t cb_gate_intermed = get_named_compile_time_arg_val("cb_gate_intermed");
    constexpr uint32_t cb_up_intermed = get_named_compile_time_arg_val("cb_up_intermed");
    constexpr uint32_t cb_activated = get_named_compile_time_arg_val("cb_activated");
    constexpr uint32_t cb_in0_down_full = get_named_compile_time_arg_val("cb_in0_down_full");
    constexpr uint32_t cb_partials_gu = get_named_compile_time_arg_val("cb_mm_partials_gu");
    constexpr uint32_t cb_partials_d = get_named_compile_time_arg_val("cb_mm_partials_d");
    constexpr uint32_t cb_out = get_named_compile_time_arg_val("cb_out");

    silu_tile_init_pack();
    mm_init(cb_in0_x, cb_in1_gate, cb_partials_gu);

    // Phase 1: gate matmul with silu fused on the final pack into gate_intermed.
    matmul_phase_v3<
        g_in0_block_w,
        g_in0_num_subblocks,
        g_in0_block_num_tiles,
        g_in0_subblock_num_tiles,
        g_in1_num_subblocks,
        g_in1_block_num_tiles,
        g_in1_per_core_w,
        g_num_blocks,
        gu_out_subblock_h,
        gu_out_subblock_w,
        gu_out_subblock_num_tiles,
        gu_out_block_num_tiles,
        /*apply_silu_on_final=*/true>(cb_in0_x, cb_in1_gate, cb_partials_gu, cb_gate_intermed);

    // Phase 2: up matmul (no activation), output to up_intermed.
    // Use full mm_init to fully reset packer/unpacker/math state — short
    // variant may leak phase-1 state (e.g. packer config still pointing
    // at gate_intermed) which corrupts phase 2's pack to partials.
    mm_init(cb_in0_x, cb_in1_up, cb_partials_gu);
    matmul_phase_v3<
        u_in0_block_w,
        u_in0_num_subblocks,
        u_in0_block_num_tiles,
        u_in0_subblock_num_tiles,
        u_in1_num_subblocks,
        u_in1_block_num_tiles,
        u_in1_per_core_w,
        u_num_blocks,
        gu_out_subblock_h,
        gu_out_subblock_w,
        gu_out_subblock_num_tiles,
        gu_out_block_num_tiles,
        /*apply_silu_on_final=*/false>(cb_in0_x, cb_in1_up, cb_partials_gu, cb_up_intermed);

    // Phase 3: elementwise multiply, output to activated.
    multiply_phase_v3<gu_out_block_num_tiles, gu_out_subblock_num_tiles>(
        cb_gate_intermed, cb_up_intermed, cb_activated);

    // Phase 4: down matmul, output to cb_out.
    mm_init(cb_in0_down_full, cb_in1_down, cb_partials_d);
    matmul_phase_v3<
        d_in0_block_w,
        d_in0_num_subblocks,
        d_in0_block_num_tiles,
        d_in0_subblock_num_tiles,
        d_in1_num_subblocks,
        d_in1_block_num_tiles,
        d_in1_per_core_w,
        d_num_blocks,
        d_out_subblock_h,
        d_out_subblock_w,
        d_out_subblock_num_tiles,
        d_out_block_num_tiles,
        /*apply_silu_on_final=*/false>(cb_in0_down_full, cb_in1_down, cb_partials_d, cb_out);
}
