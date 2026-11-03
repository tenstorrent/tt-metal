// SPDX-FileCopyrightText: 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// pack_thread_cost bench: packer-thread (TRISC2) GELU + pack of one matmul output subblock.
//
// Raw LLK: bypasses compute_kernel_lib::apply_activation_from_pack + ActivationApplyHelper (per-tile
// _llk_math_eltwise_unary_sfpu_params_) + pack_block in matmul_block's pack stage. The helper has no hook between
// the SFPU pass and the pack, so the per-tile STALLWAIT(SFPU, MATH) (an FPU-idle wait meant for math-thread SFPU; on
// the pack thread the data was already handed over by the MATH_PACK semaphore), the per-tile DEST address setup, the
// per-subblock packer reconfig and the GELU/pack ordering can only be changed here.
//
// Knobs (0/1): PT_NOSTALL drop STALLWAIT(SFPU, MATH); PT_BATCH one DEST address setup per subblock (faces walk the
// contiguous tiles); PT_NORECONFIG drop the per-subblock pack_reconfig_data_format (single bf16 pack target, set at
// hw startup); PT_INTERLEAVE pack tile t right after GELU(t) so the packer overlaps GELU(t+1).
// Diagnostics (perf-only, wrong output): PT_NOGELU skip the SFPU body; PT_NOPACK skip the pack.
#pragma once

#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "linear_gelu_gelu_sfpu.h"

#ifndef PT_NOSTALL
#define PT_NOSTALL 0
#endif
#ifndef PT_BATCH
#define PT_BATCH 0
#endif
#ifndef PT_NORECONFIG
#define PT_NORECONFIG 0
#endif
#ifndef PT_INTERLEAVE
#define PT_INTERLEAVE 0
#endif
#ifndef PT_NOGELU
#define PT_NOGELU 0
#endif
#ifndef PT_NOPACK
#define PT_NOPACK 0
#endif

namespace pt {

#ifdef TRISC_PACK
FORCE_INLINE void sfpu_start(uint32_t dst_index) {
    ckernel::math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_index);
#if !PT_NOSTALL
    TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::MATH);
#endif
}

FORCE_INLINE void gelu_tile_faces() {
#pragma GCC unroll 0
    for (uint32_t face = 0; face < 4; ++face) {
#if !PT_NOGELU
        ckernel::sfpu::linear_gelu_lm::gelu_face();
#endif
        ckernel::math::inc_dst_addr<8>();
        ckernel::math::inc_dst_addr<8>();
    }
}

FORCE_INLINE void gelu_tile(uint32_t tile) {
#if PT_BATCH
    (void)tile;
    gelu_tile_faces();
#else
    sfpu_start(tile);
    gelu_tile_faces();
    ckernel::math::clear_dst_reg_addr();
#endif
}

FORCE_INLINE void pack_tiles(uint32_t out_cb, uint32_t first, uint32_t count) {
#if !PT_NOPACK
    llk_matmul_pack<DST_ACCUM_MODE, false, PackMode::Default>(first, out_cb, count);
#endif
}
#endif

FORCE_INLINE void gelu_pack_subblock(uint32_t out_cb, uint32_t num_tiles) {
#ifdef TRISC_PACK
    TTI_SEMWAIT(
        p_stall::STALL_TDMA | p_stall::STALL_CFG, semaphore::t6_sem(semaphore::MATH_PACK), p_stall::STALL_ON_ZERO);
    TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, ckernel::packer::get_packer_dest_offset());
#if PT_BATCH
    sfpu_start(0);
#endif
#if PT_INTERLEAVE
#if !PT_NORECONFIG
    pack_reconfig_data_format(out_cb);
#endif
    for (uint32_t tile = 0; tile < num_tiles; ++tile) {
        gelu_tile(tile);
        TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU);
        pack_tiles(out_cb, tile, 1);
    }
#else
    for (uint32_t tile = 0; tile < num_tiles; ++tile) {
        gelu_tile(tile);
    }
    TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU);
#if !PT_NORECONFIG
    pack_reconfig_data_format(out_cb);
#endif
    pack_tiles(out_cb, 0, num_tiles);
#endif
#if PT_BATCH
    ckernel::math::clear_dst_reg_addr();
#endif
#endif
}

}  // namespace pt
