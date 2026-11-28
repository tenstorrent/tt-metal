// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "ckernel_ops.h"
#include "ckernel_template.h"

#if !defined(ARCH_BLACKHOLE)
#error "kv_rows_update compute kernel: unsupported configuration"
#endif

namespace {

#ifdef TRISC_UNPACK
inline void unpack_rows_setup() {
    using namespace ckernel;
    using namespace ckernel::unpacker;
    cfg_reg_rmw_tensix<THCON_SEC0_REG2_Haloize_mode_RMW>(0);
    config_unpacker_x_end<p_setadc::UNP_A>(FACE_R_DIM);
    constexpr std::uint32_t face_a = TT_OP_UNPACR(SrcA, 0b1, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    constexpr std::uint32_t mark_b = TT_OP_UNPACR_NOP(
        SrcB, 0, 0, p_unpacr_nop::SET_DVALID, 0, p_unpacr_nop::WAIT_LIKE_UNPACR, 0, 0, p_unpacr_nop::UNP_ZEROSRC);
    ckernel_template tmp(4, 1, mark_b);
    tmp.set_start_op(face_a);
    tmp.program();
}

inline void unpack_rows(std::uint32_t in, std::uint32_t tile) {
    using namespace ckernel;
    using namespace ckernel::unpacker;
    const std::uint32_t address =
        get_local_cb_interface(in).fifo_rd_ptr - 1 + get_local_cb_interface(in).fifo_page_size * tile;
    TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111);
    volatile std::uint32_t tt_reg_ptr* cfg = get_cfg_pointer();
    wait_for_next_context(2);
    cfg[unp_cfg_context == 0 ? THCON_SEC0_REG3_Base_address_ADDR32 : THCON_SEC0_REG3_Base_cntx1_address_ADDR32] = address;
    semaphore_post(semaphore::UNPACK_SYNC);
    TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG);
    ckernel_template::run();
    t6_semaphore_get(semaphore::UNPACK_SYNC);
    switch_config_context(unp_cfg_context);
}
#endif

#ifdef TRISC_MATH
inline void math_rows_setup() {
    using namespace ckernel;
    using namespace ckernel::math;
    addr_mod_t{
        .srca = {.incr = 8},
        .srcb = {.incr = 0},
        .dest = {.incr = 8},
    }
        .set(ADDR_MOD_2);
    ckernel_template tmp(4, 2, TT_OP_MOVA2D(p_mov::DEST_NORM, 0, ADDR_MOD_2, p_mova2d::MOV_8_ROWS, 0));
    tmp.set_end_op(TT_OP_SETRWC(p_setrwc::CLR_AB, 0, 0, 0, 0, p_setrwc::SET_AB));
    tmp.program();
    TTI_SETC16(CLR_DVALID_SrcA_Disable_ADDR32, 0);
    reset_counters(p_setrwc::SET_ABD_F);
}

inline void math_rows(std::uint32_t dst_index) {
    using namespace ckernel;
    using namespace ckernel::math;
    set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_index);
    ckernel_template::run();
    clear_dst_reg_addr();
}

inline void math_section_done() {
    using namespace ckernel;
    using namespace ckernel::math;
    set_math_semaphores();
    math_sync_tile_dst_index = 0;
    dest_section_flip();
}
#endif

#ifdef TRISC_PACK
inline void pack_rows(std::uint32_t dst_index, std::uint32_t out) {
    using namespace ckernel;
    using namespace ckernel::packer;
    const std::uint32_t address =
        get_local_cb_interface(out).fifo_wr_ptr + get_local_cb_interface(out).fifo_wr_tile_ptr - 1;
    get_local_cb_interface(out).fifo_wr_tile_ptr += get_local_cb_interface(out).fifo_page_size;
    set_dst_write_addr(dst_index);
    program_packer_destination(address);
    ckernel_template::run();
    TTI_SETADCZW(p_setadc::PAC, 0, 0, 0, 0, 0b0101);
}

inline void pack_section_done() {
    using namespace ckernel;
    using namespace ckernel::packer;
    TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::PACK);
    TT_ZEROACC(p_zeroacc::CLR_HALF, 0, 0, ADDR_MOD_1, dest_offset_id % 2);
    t6_semaphore_get<p_stall::NONE>(semaphore::MATH_PACK);
    flip_packer_dest_offset_id();
    select_packer_dest_registers<DstSync::SyncHalf>();
}
#endif

}  // namespace

void kernel_main() {
    constexpr uint32_t NG = get_compile_time_arg_val(0);
    constexpr uint32_t cin = get_compile_time_arg_val(1);
    constexpr uint32_t cpk = get_compile_time_arg_val(2);
    compute_kernel_hw_startup(cin, cin, cpk);
    UNPACK((unpack_rows_setup()));
    MATH((math_rows_setup()));
    cb_wait_front(cin, NG);
    cb_reserve_back(cpk, NG);
    for (uint32_t g = 0; g < NG; ++g) {
        MATH((ckernel::math::math_dest_wait()));
        UNPACK((unpack_rows(cin, g)));
        MATH((math_rows(0)));
        MATH((math_section_done()));
        PACK((TTI_SEMWAIT(
            ckernel::p_stall::STALL_TDMA | ckernel::p_stall::STALL_SYNC,
            ckernel::semaphore::t6_sem(ckernel::semaphore::MATH_PACK),
            ckernel::p_stall::STALL_ON_ZERO)));
        PACK((pack_rows(0, cpk)));
        PACK((pack_section_done()));
    }
    cb_push_back(cpk, NG);
    cb_pop_front(cin, NG);
}
