// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/common.h"
#include "ckernel_ops.h"
#include "ckernel_template.h"

#ifdef TRISC_UNPACK
using namespace ckernel::unpacker;
inline void unpack_setup() {
    cfg_reg_rmw_tensix<THCON_SEC0_REG2_Haloize_mode_RMW>(0);
    config_unpacker_x_end<p_setadc::UNP_A>(16);
    ckernel_template tmp(
        4,
        1,
        TT_OP_UNPACR_NOP(
            SrcB, 0, 0, p_unpacr_nop::SET_DVALID, 0, p_unpacr_nop::WAIT_LIKE_UNPACR, 0, 0, p_unpacr_nop::UNP_ZEROSRC));
    tmp.set_start_op(TT_OP_UNPACR(SrcA, 0b1, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1));
    tmp.program();
}
inline void unpack_to_srca(const uint32_t cb) {
    const uint32_t address = get_local_cb_interface(cb).fifo_rd_ptr - 1;
    TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111);
    volatile uint32_t tt_reg_ptr* cfg = get_cfg_pointer();
    wait_for_next_context(2);
    cfg[(unp_cfg_context == 0) ? THCON_SEC0_REG3_Base_address_ADDR32 : THCON_SEC0_REG3_Base_cntx1_address_ADDR32] =
        address;
    semaphore_post(semaphore::UNPACK_SYNC);
    TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG);
    ckernel_template::run();
    t6_semaphore_get(semaphore::UNPACK_SYNC);
    switch_config_context(unp_cfg_context);
}
#endif

#ifdef TRISC_PACK
using namespace ckernel::packer;
inline void pack_dest0(const uint32_t out) {
    TTI_SEMWAIT(
        p_stall::STALL_TDMA | p_stall::STALL_SYNC, semaphore::t6_sem(semaphore::MATH_PACK), p_stall::STALL_ON_ZERO);
    auto& cb = get_local_cb_interface(out);
    const uint32_t address = cb.fifo_wr_ptr + cb.fifo_wr_tile_ptr - 1;
    cb.fifo_wr_tile_ptr += cb.fifo_page_size;
    TTI_SETADC(p_setadc::PAC, p_setadc::CH_0, p_setadc::SET_W, 0);
    program_packer_destination(address);
    ckernel_template::run();
    TTI_SETADCZW(p_setadc::PAC, 0, 0, 0, 0, 0b0101);
    TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::PACK);
    TT_ZEROACC(p_zeroacc::CLR_HALF, DST_ACCUM_MODE, 0, ADDR_MOD_1, dest_offset_id % 2);
    t6_semaphore_get<p_stall::NONE>(semaphore::MATH_PACK);
    flip_packer_dest_offset_id();
    select_packer_dest_registers<DstSync::SyncHalf>();
}
#endif

#ifdef TRISC_MATH
using namespace ckernel::math;
inline void math_copy_setup(const uint32_t a) {
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}}.set(ADDR_MOD_3);
    addr_mod_t{.srca = {.incr = 1}, .srcb = {.incr = 0}, .dest = {.incr = 1}}.set(ADDR_MOD_0);
    addr_mod_t{.srca = {.incr = 8}, .srcb = {.incr = 0}, .dest = {.incr = 8}}.set(ADDR_MOD_2);
    ckernel_template tmp(4, 2, TT_OP_MOVA2D(p_mov::DEST_NORM, 0, ADDR_MOD_2, p_mova2d::MOV_8_ROWS, 0));
    tmp.set_end_op(TT_OP_SETRWC(p_setrwc::CLR_AB, 0, 0, 0, 0, p_setrwc::SET_AB));
    tmp.program();
    TTI_SETC16(CLR_DVALID_SrcA_Disable_ADDR32, 0);
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_ABD_F);
    _configure_copy_zero_flag_state_(unpack_dst_format[a]);
}
inline void srca_to_dest(const uint32_t dst) {
    TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, (dst << 6) + get_dest_buffer_base());
    ckernel_template::run();
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
}
inline void sfpu_at_dest0() {
    TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, get_dest_buffer_base());
    TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::MATH);
}
inline void math_give_dest() {
    set_math_semaphores();
    math_sync_tile_dst_index = 0;
    dest_section_flip();
}
inline void math_sfpu_setup() {
    TTI_SFPCONFIG(0, 15, 1);
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}}.set(ADDR_MOD_7);
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_ABD_F);
    TTI_SFPLOADI(0, 0, 16384);
    TTI_SFPCONFIG(0, 12, 0);
    TTI_SFPLOADI(0, 2, 43579);
    TTI_SFPLOADI(0, 8, 16312);
    TTI_SFPCONFIG(0, 13, 0);
    TTI_SFPLOADI(2, 2, 14469);
    TTI_SFPLOADI(2, 8, 16256);
    TTI_SFPLOADI(4, 2, 23258);
    TTI_SFPLOADI(4, 8, 13224);
    TTI_SFPLOADI(1, 2, 42008);
    TTI_SFPLOADI(1, 8, 10156);
    TTI_REPLAY(0, 16, 0, 1);
    TTI_SFPLOAD(3, 0, 7, 0);
    TTI_SFPLOADI(0, 0, 17150);
    TTI_SFPMAD(3, 13, 0, 0, 1);
    TTI_SFPLOADI(5, 0, 17279);
    TTI_SFPSWAP(0, 0, 9, 1);
    TTI_SFPSWAP(0, 5, 0, 1);
    TTI_SFPEXEXP(0, 0, 5, 0);
    TTI_SFPEXMAN(0, 0, 0, 0);
    TTI_SFPSHFT(0, 5, 0, 0);
    TTI_SFPEXEXP(0, 0, 5, 1);
    TTI_SFPEXMAN(0, 0, 0, 1);
    TTI_SFPCAST(0, 0, 0);
    TTI_SFPMAD(0, 1, 4, 6, 0);
    TTI_SFPMAD(0, 6, 2, 0, 0);
    TTI_SFPSETEXP(0, 0, 5, 0);
    TTI_SFPADD(10, 5, 10, 5, 0);
}
inline void math_swiglu() {
    TTI_SEMWAIT(
        p_stall::STALL_MATH | p_stall::STALL_SFPU | p_stall::STALL_SYNC,
        semaphore::t6_sem(semaphore::MATH_PACK),
        p_stall::STALL_ON_MAX);
    srca_to_dest(0);
    srca_to_dest(1);
    sfpu_at_dest0();
    for (uint32_t f = 0; f < 4; ++f) {
        for (uint32_t r = 0; r < 8; ++r) {
            TTI_REPLAY(0, 16, 0, 0);
            TTI_SFPARECIP(0, 5, 0, 0);
            TTI_SFPMAD(5, 0, 12, 5, 2);
            TTI_SFPGT(0, 5, 9, 1);
            TTI_SFPLOADI(6, 0, 32768);
            TTI_SFPMAD(5, 0, 6, 0, 1);
            TTI_SFPENCC(3, 0, 0, 10);
            TTI_SFPMUL(3, 0, 9, 3, 0);
            TTI_SFP_STOCH_RND(0, 0, 9, 3, 3, 1);
            TTI_SFPLOAD(7, 0, 7, 64);
            TTI_SFPMUL(3, 7, 9, 0, 0);
            TTI_SFPSHFT(4080, 0, 5, 5);
            TTI_SFPLOADI(6, 2, 1);
            TTI_SFPAND(5, 6, 5, 1);
            TTI_SFPLOADI(6, 2, 32767);
            TTI_SFPIADD(0, 6, 0, 4);
            TTI_SFPIADD(0, 5, 0, 4);
            TTI_SFPLOADI(5, 0, 65535);
            TTI_SFPAND(0, 5, 0, 1);
            TTI_SFPSETCC(0, 3, 0, 2);
            TTI_SFPSETCC(0, 7, 0, 2);
            TTI_SFPCOMPC(0, 0, 0, 0);
            TTI_SFPMOV(0, 9, 0, 0);
            TTI_SFPENCC(3, 0, 0, 10);
            TTI_SFPSTORE(0, 0, 7, 0);
            TTI_INCRWC(0, 2, 0, 0);
        }
        TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
        TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
    }
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
    math_give_dest();
}
#endif

void kernel_main() {
    constexpr uint32_t a = get_compile_time_arg_val(0);
    constexpr uint32_t b = get_compile_time_arg_val(1);
    constexpr uint32_t out = get_compile_time_arg_val(2);
    const uint32_t n = get_arg_val<uint32_t>(0);
    compute_kernel_hw_startup(a, b, out);
    UNPACK((unpack_setup()));
    MATH((math_copy_setup(a)));
    MATH((math_sfpu_setup()));
    for (uint32_t p = 0; p < n; ++p) {
        cb_wait_front(a, 1);
        cb_wait_front(b, 1);
        cb_reserve_back(out, 1);
        UNPACK((unpack_to_srca(a)));
        UNPACK((unpack_to_srca(b)));
        MATH((math_swiglu()));
        PACK((pack_dest0(out)));
        cb_push_back(out, 1);
        cb_pop_front(a, 1);
        cb_pop_front(b, 1);
    }
}
