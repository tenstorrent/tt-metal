// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Fused scaled-FP8 KV dequant for sparse_sdpa (Blackhole): tilize the FP8 latent and multiply it by its per-row,
// per-128-column scale in a single FPU pass.
//
// The plain path is two passes per latent tile: tilize (srcA -> ELWADD with a zeroed srcB -> fp32 dest -> bfp8
// pack), then unpack that bfp8 tile again and mul_tiles_bcast_cols by the FP32 scale tile (-> second bfp8 pack).
// The BH 8-bit tilize already routes srcA through the FPU, so here srcB carries the scale instead of zeros and
// the FPU op is ELWMUL with SRCB_BCAST_COL: one unpack, one FPU pass and one pack per tile.
//
//   srcA rows  0..15 = tile face 0, 16..31 = face 1, 32..47 = face 2, 48..63 = face 3   (8-bit tilize layout)
//   srcB rows  0..15 = scale face 0, 16..31 = face 0, 32..47 = face 2, 48..63 = face 2 (column 0 = row scale)
//
// so one 8-instruction ELWMUL sweep (8 rows each, srcA/srcB/dest all +8) scales every row by its own scale.
// FP8 unpacks to FP16 in srcA and the FPU needs one exponent family, so the scale tile is FP16 (see
// sparse_sdpa_common.hpp for the 2^FUSED_DEQUANT_EXP_OFFSET range offset and how it is undone). The scale CB is
// declared UInt16 on the host (the JIT rejects mixing an FP16 CB with the BF16 ones); srcB is reprogrammed to
// Float16 here.
#pragma once

#include "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/sparse_sdpa_common.hpp"

namespace sparse_sdpa::fused_dequant {

#ifdef TRISC_UNPACK
// One latent tile: srcB <- scale faces 0,0,2,2 (one dvalid), srcA <- 8-bit tilize of the FP8 row-major field
// (the inline sequence of _llk_unpack_tilize_'s 8-bit path, dvalid on face 3).
inline void unpack_tile(uint32_t address_a, uint32_t address_b) {
    volatile uint tt_reg_ptr* cfg = get_cfg_pointer();
    TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111);
    wait_for_next_context(2);
    _llk_unpack_configure_addresses_(address_a, address_b, cfg);
    semaphore_post(semaphore::UNPACK_SYNC);
    TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG);

    // SETADCZW mask 0b0101 = {ch0_z, ch1_z}: ch0_z picks the L1 face, ch1_z the srcB face slot.
    TTI_SETADCZW(p_setadc::UNP_B, 0, 0, 0, 0, 0b0101);
    TTI_UNPACR(SrcB, 0, 0, 0, 0, 1, 0, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    TTI_SETADCZW(p_setadc::UNP_B, 0, 1, 0, 0, 0b0101);
    TTI_UNPACR(SrcB, 0, 0, 0, 0, 1, 0, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    TTI_SETADCZW(p_setadc::UNP_B, 0, 2, 0, 2, 0b0101);
    TTI_UNPACR(SrcB, 0, 0, 0, 0, 1, 0, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    TTI_SETADCZW(p_setadc::UNP_B, 0, 3, 0, 2, 0b0101);
    TTI_UNPACR(SrcB, 0, 0, 0, 0, 1, 1 /*dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);

    TTI_UNPACR(SrcA, 0b00010001, 0, 0, 0, 1, 0, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    TTI_UNPACR(SrcA, 0b00010001, 0, 0, 0, 1, 0, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    if (0 == unp_cfg_context) {
        TTI_CFGSHIFTMASK(1, 0b011, 32 - 1, 0, 0b11, THCON_SEC0_REG3_Base_address_ADDR32);
    } else {
        TTI_CFGSHIFTMASK(1, 0b011, 32 - 1, 0, 0b11, THCON_SEC0_REG3_Base_cntx1_address_ADDR32);
    }
    TTI_NOP;
    TTI_UNPACR(SrcA, 0b00010001, 0, 0, 0, 1, 0, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    TTI_UNPACR(SrcA, 0b00010001, 0, 0, 0, 1, 1 /*dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    t6_semaphore_get(semaphore::UNPACK_SYNC);
    switch_config_context(unp_cfg_context);
}
#endif

#ifdef TRISC_MATH
// One LoFi pass regardless of the op's math fidelity: exact for the FP8 srcA (4 significant bits) times a
// power-of-two scale (the production cast, round_scale_to_power_of_two=True). An arbitrary scale keeps only
// srcB's LoFi bits (~7), about the precision the two-pass path's BFP8 intermediate kept.
inline void math_init() {
    addr_mod_t{.srca = {.incr = 8}, .srcb = {.incr = 8}, .dest = {.incr = 8}}.set(ADDR_MOD_2);
    ckernel_template tmp(1, 8, TT_OP_ELWMUL(0, 0, p_elwise::SRCB_BCAST_COL, ADDR_MOD_2, 0));
    tmp.set_end_op(TT_OP_SETRWC(p_setrwc::CLR_AB, 0, 0, 0, 0, p_setrwc::SET_AB));
    tmp.program();
    math::reset_counters(p_setrwc::SET_ABD_F);
    // FP8 zeros widen into srcA with a nonzero residual unless the src zero flags flush them (as the FP8
    // datacopy does).
    math::_configure_src_zero_flag_(false);
}

inline void math_tile(uint32_t dst_index) {
    math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_index);
    ckernel_template::run();
    math::clear_dst_reg_addr();
}
#endif

// Dequantize the latent field of one 32-row packed slab (front of cb_packed) into tile row `row_tile` of
// cb_out, consuming `scale_blocks` FP16 scale tiles from the front of cb_scale (not popped here).
template <
    uint32_t cb_packed,
    uint32_t cb_scale,
    uint32_t cb_out,
    uint32_t latent_physical_ct,
    uint32_t scale_blocks,
    uint32_t tiles_per_scale_block,
    uint32_t out_row_tiles>
ALWI void latent_slab(uint32_t row_tile) {
    static_assert(tiles_per_scale_block <= 4, "one scale block per FP32 half-DEST");
    reconfig_data_format_srca(cb_packed);
    UNPACK((_llk_unpack_reconfig_data_format_srcb_impl_<DST_ACCUM_MODE, p_dim_stride_target::FACE_ROW_MAJOR>(
        static_cast<uint32_t>(DataFormat::Float16),
        static_cast<uint32_t>(DataFormat::Float16),
        get_local_cb_interface(cb_scale).fifo_page_size,
        FACE_R_DIM,
        4)));
    MATH((_llk_math_reconfig_data_format_srcb_<DST_ACCUM_MODE>(static_cast<uint32_t>(DataFormat::Float16))));
    pack_reconfig_data_format(cb_out);
    PACK((llk_pack_init<PackMode::Default>(cb_out)));
    UNPACK((llk_unpack_tilize_init(cb_packed, latent_physical_ct)));
    UNPACK((TTI_SETADCXX(p_setadc::UNP_B, FACE_R_DIM * FACE_C_DIM - 1, 0x0)));
    MATH((math_init()));

    uint32_t base_a = 0, base_b = 0, page_b = 0;
    UNPACK(base_a = get_local_cb_interface(cb_packed).fifo_rd_ptr - 1;
           base_b = get_local_cb_interface(cb_scale).fifo_rd_ptr - 1;
           page_b = get_local_cb_interface(cb_scale).fifo_page_size;);
    for (uint32_t block = 0; block < scale_blocks; ++block) {
        tile_regs_acquire();
        for (uint32_t tile = 0; tile < tiles_per_scale_block; ++tile) {
            // 8-bit row-major field: one tile column is 32 bytes = 2 L1 address units.
            const uint32_t column = block * tiles_per_scale_block + tile;
            UNPACK((unpack_tile(base_a + column * 2, base_b + block * page_b)));
            MATH((math_tile(tile)));
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t tile = 0; tile < tiles_per_scale_block; ++tile) {
            pack_tile<true>(tile, cb_out, row_tile * out_row_tiles + block * tiles_per_scale_block + tile);
        }
        tile_regs_release();
    }
    UNPACK((llk_unpack_tilize_uninit(cb_packed)));
}

// Multiply Q's latent columns (the first latent_tiles of each tile row) by 2^-FUSED_DEQUANT_EXP_OFFSET in place,
// so Q_latent . (2^OFFSET K_latent) == Q_latent . K_latent exactly. cb_q holds exactly one token, so after its
// push the write pointer is back at the front and pack_tile<true> rewrites the tiles in place.
template <uint32_t cb_q, uint32_t q_row_tiles, uint32_t row_tiles, uint32_t latent_tiles>
ALWI void prescale_q_latent() {
    constexpr uint32_t tiles_per_acquire = 4;
    static_assert(latent_tiles % tiles_per_acquire == 0, "Q latent prescale packs four tiles per DEST acquire");
    constexpr uint32_t inv_offset_fp32 = static_cast<uint32_t>(127 - FUSED_DEQUANT_EXP_OFFSET) << 23;
    CircularBuffer(cb_q).wait_front(q_row_tiles * row_tiles);
    reconfig_data_format_srca(cb_q);
    pack_reconfig_data_format(cb_q);
    copy_init(cb_q);
    binop_with_scalar_tile_init();
    for (uint32_t r = 0; r < q_row_tiles; ++r) {
        for (uint32_t t0 = 0; t0 < latent_tiles; t0 += tiles_per_acquire) {
            tile_regs_acquire();
            for (uint32_t i = 0; i < tiles_per_acquire; ++i) {
                copy_tile(cb_q, r * row_tiles + t0 + i, i);
                mul_unary_tile(i, inv_offset_fp32);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t i = 0; i < tiles_per_acquire; ++i) {
                pack_tile<true>(i, cb_q, r * row_tiles + t0 + i);
            }
            tile_regs_release();
        }
    }
}

}  // namespace sparse_sdpa::fused_dequant
