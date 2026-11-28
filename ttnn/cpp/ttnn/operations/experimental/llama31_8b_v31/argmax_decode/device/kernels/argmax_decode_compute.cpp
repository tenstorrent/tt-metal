// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"

#ifdef TRISC_UNPACK
namespace {

inline void adm_unpack_init() {
    cfg_reg_rmw_tensix<THCON_SEC0_REG2_Haloize_mode_RMW>(0);
    config_unpacker_x_end<p_setadc::UNP_A>(FACE_R_DIM);
    ckernel_template tmp(
        4,
        1,
        TT_OP_UNPACR_NOP(
            SrcB, 0, 0, p_unpacr_nop::SET_DVALID, 0, p_unpacr_nop::WAIT_LIKE_UNPACR, 0, 0, p_unpacr_nop::UNP_ZEROSRC));
    tmp.set_start_op(TT_OP_UNPACR(SrcA, 0b1, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1));
    tmp.program();
}

inline void adm_unpack_tiles(const uint32_t cb, const uint32_t num_tiles) {
    const uint32_t id = get_operand_id(cb);
    const uint32_t base = get_local_cb_interface(id).fifo_rd_ptr - 1;
    const uint32_t step = get_local_cb_interface(id).fifo_page_size;
    volatile uint32_t tt_reg_ptr* cfg = get_cfg_pointer();
    for (uint32_t t = 0; t < num_tiles; ++t) {
        TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111);
        wait_for_next_context(2);
        cfg[unp_cfg_context == 0 ? THCON_SEC0_REG3_Base_address_ADDR32 : THCON_SEC0_REG3_Base_cntx1_address_ADDR32] =
            base + t * step;
        semaphore_post(semaphore::UNPACK_SYNC);
        TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG);
        ckernel_template::run();
        t6_semaphore_get(semaphore::UNPACK_SYNC);
        switch_config_context(unp_cfg_context);
    }
}

}  // namespace
#endif

#ifdef TRISC_MATH
namespace {

inline void adm_math_init() {
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}}.set(ADDR_MOD_3);
    addr_mod_t{.srca = {.incr = 1}, .srcb = {.incr = 0}, .dest = {.incr = 1}}.set(ADDR_MOD_0);
    addr_mod_t{.srca = {.incr = 8}, .srcb = {.incr = 0}, .dest = {.incr = 8}}.set(ADDR_MOD_2);
    ckernel_template tmp(4, 2, TT_OP_MOVA2D(p_mov::DEST_32B_LOW, 0, ADDR_MOD_2, p_mova2d::MOV_8_ROWS, 0));
    tmp.set_end_op(TT_OP_SETRWC(p_setrwc::CLR_AB, 0, 0, 0, 0, p_setrwc::SET_AB));
    tmp.program();
    TTI_SETC16(CLR_DVALID_SrcA_Disable_ADDR32, 0);
    math::reset_counters(p_setrwc::SET_ABD_F);
    math::_configure_copy_zero_flag_state_(static_cast<uint32_t>(DataFormat::UInt16));
    TTI_SFPCONFIG(0, 0xF, 1);
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}}.set(ADDR_MOD_7);
    math::reset_counters(p_setrwc::SET_ABD_F);
}

inline __attribute__((always_inline)) void adm_key() {
    TTI_SFPLOAD(3, 0, 7, 0);
    TTI_SFPLOADI(0, 2, 65535);
    TTI_SFPAND(3, 0, 3, 1);
    TTI_SFPLOADI(1, 2, 32767);
    TTI_SFPAND(3, 1, 1, 1);
    TTI_SFPLOADI(4, 2, 32768);
    TTI_SFPMOV(0, 1, 0, 2);
    TTI_SFPIADD(0, 4, 0, 4);
    TTI_SFPLOADI(4, 2, 32768);
    TTI_SFPIADD(0, 3, 4, 10);
    TTI_SFPLOADI(3, 2, 32768);
    TTI_SFPMOV(0, 1, 0, 0);
    TTI_SFPIADD(0, 3, 0, 6);
    TTI_SFPENCC(3, 0, 0, 10);
    TTI_SFPLOADI(3, 2, 32640);
    TTI_SFPIADD(0, 3, 1, 2);
    TTI_SFPLOADI(0, 0, 1);
    TTI_SFPENCC(3, 0, 0, 10);
    TTI_SFPSHFT(14, 0, 0, 7);
    TTI_SFPIADD(0, 2, 0, 4);
}

inline void adm_face_first() {
#pragma GCC unroll 0
    for (uint32_t r = 0; r < 8; ++r) {
        adm_key();
        TTI_SFPSTORE(0, 0, 7, 64);
        TTI_INCRWC(0, 2, 0, 0);
    }
    TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
    TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
}

inline void adm_face_cmp() {
#pragma GCC unroll 0
    for (uint32_t r = 0; r < 8; ++r) {
        adm_key();
        TTI_SFPLOAD(1, 0, 7, 64);
        TTI_SFPMOV(0, 0, 3, 2);
        TTI_SFPIADD(0, 1, 3, 2);
        TTI_SFPSTORE(0, 0, 7, 64);
        TTI_SFPENCC(3, 0, 0, 10);
        TTI_INCRWC(0, 2, 0, 0);
    }
    TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
    TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
}

inline void adm_math_tile(const uint32_t t) {
    math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(0);
    ckernel_template::run();
    math::clear_dst_reg_addr();
    math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(0);
    TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::MATH);
    const uint32_t low = 0x00800000u + (0x3FFFu - t);
    TT_SFPLOADI(2, 2, low & 0xFFFFu);
    TT_SFPLOADI(2, 8, low >> 16);
#pragma GCC unroll 0
    for (uint32_t f = 0; f < 4; ++f) {
        if (t == 0) {
            adm_face_first();
        } else {
            adm_face_cmp();
        }
    }
    math::clear_dst_reg_addr();
}

inline void adm_dest_acquire() {
    TTI_SEMWAIT(
        p_stall::STALL_MATH | p_stall::STALL_SFPU | p_stall::STALL_SYNC,
        semaphore::t6_sem(semaphore::MATH_PACK),
        p_stall::STALL_ON_MAX);
}
inline void adm_dest_commit() {
    TTI_STALLWAIT(p_stall::STALL_SYNC, p_stall::MATH | p_stall::WAIT_SFPU);
    TTI_SEMPOST(semaphore::t6_sem(semaphore::MATH_PACK));
    math_sync_tile_dst_index = 0;
    dest_offset_id = 1 - dest_offset_id;
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::MATH | p_stall::SFPU1);
    TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, dest_offset_id ? DEST_REGISTER_HALF_SIZE : 0);
}
inline void adm_math_tiles(const uint32_t num_tiles) {
    for (uint32_t t = 0; t < num_tiles; ++t) {
        adm_math_tile(t);
    }
}

}  // namespace
#endif

#ifdef TRISC_PACK
namespace {

inline void adm_dest_wait() {
    TTI_SEMWAIT(
        p_stall::STALL_TDMA | p_stall::STALL_SYNC, semaphore::t6_sem(semaphore::MATH_PACK), p_stall::STALL_ON_ZERO);
}
inline void adm_dest_release() {
    TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::PACK);
    TT_ZEROACC(p_zeroacc::CLR_HALF, 1, 0, ADDR_MOD_1, dest_offset_id % 2);
    TTI_SEMGET(semaphore::t6_sem(semaphore::MATH_PACK));
    dest_offset_id = 1 - dest_offset_id;
    TT_WRCFG(
        dest_offset_id ? p_gpr_pack::DEST_OFFSET_HI : p_gpr_pack::DEST_OFFSET_LO,
        p_cfg::WRCFG_32b,
        DEST_TARGET_REG_CFG_PACK_SEC0_Offset_ADDR32);
    TTI_DMANOP;
    TTI_DMANOP;
}
inline void adm_pack_keys(const uint32_t cb) {
    const uint32_t address = get_output_tile_address<false, PackMode::Default>(get_output_id(cb), 0);
    TTI_SETADC(p_setadc::PAC, p_setadc::CH_0, p_setadc::SET_W, 1);
    program_packer_destination(address);
    ckernel_template::run();
    TTI_SETADCZW(p_setadc::PAC, 0, 0, 0, 0, 0b0101);
}

}  // namespace
#endif

static_assert(DST_SYNC_MODE == DstSync::SyncHalf && DST_ACCUM_MODE);

void kernel_main() {
    const uint32_t num_tiles = get_arg_val<uint32_t>(0);
    constexpr uint32_t cb_in = get_compile_time_arg_val(0);
    constexpr uint32_t cb_keys = get_compile_time_arg_val(1);

    compute_kernel_hw_startup(cb_in, cb_keys);
    CircularBuffer in(cb_in);
    CircularBuffer keys(cb_keys);
    if (num_tiles == 0) {
        return;
    }
    in.wait_front(num_tiles);
    UNPACK((adm_unpack_init()));
    MATH((adm_math_init()));

    MATH((adm_dest_acquire()));
    UNPACK((adm_unpack_tiles(cb_in, num_tiles)));
    MATH((adm_math_tiles(num_tiles)));
    MATH((adm_dest_commit()));

    keys.reserve_back(1);
    PACK((adm_dest_wait()));
    PACK((adm_pack_keys(cb_keys)));
    PACK((adm_dest_release()));
    keys.push_back(1);
}
