// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

#if defined(FUSE_BIAS) || defined(SFPU_ACTIVATION) || defined(PACK_RELU) || defined(IN0_TRANSPOSE_TILE) || \
    defined(IN1_TRANSPOSE_TILE) || defined(MM_PARTIALS_RELOAD_ALIAS) || defined(SKIP_COMPUTE) ||           \
    defined(ARCH_QUASAR) || !defined(PACKER_L1_ACC)
#error "configuration not supported by this kernel"
#endif

#ifdef TRISC_MATH
namespace {
inline void mm_math_dest_take() {
    TTI_SEMWAIT(
        p_stall::STALL_MATH | p_stall::STALL_SFPU | p_stall::STALL_SYNC,
        semaphore::t6_sem(semaphore::MATH_PACK),
        p_stall::STALL_ON_MAX);
}
inline void mm_math_dest_give() {
    TTI_STALLWAIT(p_stall::STALL_SYNC, p_stall::MATH | p_stall::WAIT_SFPU);
    TTI_SEMPOST(semaphore::t6_sem(semaphore::MATH_PACK));
    math_sync_tile_dst_index = 0;
    dest_offset_id = 1 - dest_offset_id;
    const std::uint32_t half = (0 != dest_offset_id) ? DEST_REGISTER_HALF_SIZE : 0x0;
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::MATH | p_stall::SFPU1);
    TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, half);
}
}  // namespace
#endif

#ifdef TRISC_PACK
namespace {
inline void mm_pack_dest_take() {
    TTI_SEMWAIT(
        p_stall::STALL_TDMA | p_stall::STALL_SYNC, semaphore::t6_sem(semaphore::MATH_PACK), p_stall::STALL_ON_ZERO);
}
inline void mm_pack_dest_give() {
    TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::PACK);
    TT_ZEROACC(p_zeroacc::CLR_HALF, DST_ACCUM_MODE, 0, ADDR_MOD_1, dest_offset_id % 2);
    TTI_SEMGET(semaphore::t6_sem(semaphore::MATH_PACK));
    dest_offset_id = 1 - dest_offset_id;
    TT_WRCFG(
        dest_offset_id ? p_gpr_pack::DEST_OFFSET_HI : p_gpr_pack::DEST_OFFSET_LO,
        p_cfg::WRCFG_32b,
        DEST_TARGET_REG_CFG_PACK_SEC0_Offset_ADDR32);
    TTI_DMANOP;
    TTI_DMANOP;
}
}  // namespace
#endif

#ifdef TRISC_UNPACK
namespace {

inline uint32_t mm_rd(const uint32_t id) { return get_local_cb_interface(id).fifo_rd_ptr - 1; }
inline uint32_t mm_page(const uint32_t id) { return get_local_cb_interface(id).fifo_page_size; }
inline bool mm_dest_direct(const uint32_t id) { return is_32bit_input(unpack_src_format[id], unpack_dst_format[id]); }

inline void mm_unp_stream(const uint32_t in1_cb, const uint32_t ct, const uint32_t rt, const uint32_t kt) {
    const bool along_w = ct >= rt;
    cfg_reg_rmw_tensix<THCON_SEC0_REG2_Haloize_mode_RMW>(0);
    TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111);
    TTI_SETADCXX(p_setadc::UNP_A, 4 * FACE_R_DIM * FACE_C_DIM - 1, 0x0);
    TTI_SETADCXX(p_setadc::UNP_B, 4 * FACE_R_DIM * FACE_C_DIM - 1, 0x0);
    TT_SETDMAREG(0, LOWER_HALFWORD(kt), 0, LO_16(p_gpr_unpack::KT_DIM));
    if (along_w) {
        load_replay_buf(0, 12, [] {
            TTI_UNPACR(SrcA, 0, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
            TTI_RDCFG(p_gpr_unpack::TMP0, THCON_SEC0_REG3_Base_address_ADDR32);
            TTI_ADDDMAREG(0, p_gpr_unpack::TMP0, p_gpr_unpack::TMP0, p_gpr_unpack::TILE_SIZE_A);
            TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::THCON);
            TTI_WRCFG(p_gpr_unpack::TMP0, 0, THCON_SEC0_REG3_Base_address_ADDR32);
            TTI_NOP;
            TTI_UNPACR(SrcA, 0, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
            TTI_RDCFG(p_gpr_unpack::TMP0, THCON_SEC0_REG3_Base_cntx1_address_ADDR32);
            TTI_ADDDMAREG(0, p_gpr_unpack::TMP0, p_gpr_unpack::TMP0, p_gpr_unpack::TILE_SIZE_A);
            TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::THCON);
            TTI_WRCFG(p_gpr_unpack::TMP0, 0, THCON_SEC0_REG3_Base_cntx1_address_ADDR32);
            TTI_NOP;
        });
    } else {
        load_replay_buf(0, 12, [] {
            TTI_UNPACR(SrcB, 0, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
            TTI_RDCFG(p_gpr_unpack::TMP0, THCON_SEC1_REG3_Base_address_ADDR32);
            TTI_ADDDMAREG(0, p_gpr_unpack::TMP0, p_gpr_unpack::TMP0, p_gpr_unpack::TMP_LO);
            TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::THCON);
            TTI_WRCFG(p_gpr_unpack::TMP0, 0, THCON_SEC1_REG3_Base_address_ADDR32);
            TTI_NOP;
            TTI_UNPACR(SrcB, 0, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
            TTI_RDCFG(p_gpr_unpack::TMP0, THCON_SEC1_REG3_Base_cntx1_address_ADDR32);
            TTI_ADDDMAREG(0, p_gpr_unpack::TMP0, p_gpr_unpack::TMP0, p_gpr_unpack::TMP_LO);
            TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::THCON);
            TTI_WRCFG(p_gpr_unpack::TMP0, 0, THCON_SEC1_REG3_Base_cntx1_address_ADDR32);
            TTI_NOP;
        });
    }
    ckernel_unpack_template(false, false, lltt::replay_insn(0, 6), 0, 0, 0, lltt::replay_insn(6, 6), 0, 0).program();
}

inline void mm_unp_kstep(
    const uint32_t in0_cb,
    const uint32_t in1_cb,
    const uint32_t i0,
    const uint32_t i1,
    const uint32_t ct,
    const uint32_t rt,
    const uint32_t kt) {
    volatile uint32_t tt_reg_ptr* cfg = get_cfg_pointer();
    const uint32_t id0 = get_operand_id(in0_cb);
    const uint32_t id1 = get_operand_id(in1_cb);
    const uint32_t p0 = mm_page(id0);
    const uint32_t p1 = mm_page(id1);
    const uint32_t b0 = mm_rd(id0) + p0 * i0;
    const uint32_t a0 = mm_rd(id1) + p1 * i1;
    const bool along_w = ct >= rt;
    const uint32_t outer = along_w ? rt : ct;
    const uint32_t b_step = along_w ? p0 * kt : 0;
    const uint32_t a_step = along_w ? 0 : p1;
    if (!along_w) {
        TTI_MULDMAREG(0, p_gpr_unpack::TMP_LO, p_gpr_unpack::TILE_SIZE_B, p_gpr_unpack::KT_DIM);
    }
    for (uint32_t t = 0; t < outer; t++) {
        wait_for_next_context(2);
        if (unp_cfg_context == 0) {
            cfg[THCON_SEC0_REG3_Base_address_ADDR32] = a0 + t * a_step;
            cfg[THCON_SEC1_REG3_Base_address_ADDR32] = b0 + t * b_step;
        } else {
            cfg[THCON_SEC0_REG3_Base_cntx1_address_ADDR32] = a0 + t * a_step;
            cfg[THCON_SEC1_REG3_Base_cntx1_address_ADDR32] = b0 + t * b_step;
        }
        semaphore_post(semaphore::UNPACK_SYNC);
        TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG);
        if (along_w) {
            TTI_UNPACR(SrcB, 0, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
        } else {
            TTI_UNPACR(SrcA, 0, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
        }
        TT_MOP(0, (along_w ? ct : rt) - 1, unp_cfg_context == 0 ? 0 : 0xff);
        t6_semaphore_get(semaphore::UNPACK_SYNC);
        switch_config_context(unp_cfg_context);
    }
}

inline void mm_unp_srca_to(const uint32_t from_cb, const uint32_t to_cb) {
    const uint32_t f = get_operand_id(from_cb);
    const uint32_t id = get_operand_id(to_cb);
    if (unpack_src_format[f] == unpack_src_format[id] && unpack_dst_format[f] == unpack_dst_format[id]) {
        return;
    }
    const uint32_t sf = unpack_src_format[id];
    const uint32_t df = unpack_dst_format[id];
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::UNPACK0);
    cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG0_SrcAUnsigned_RMW>(sf == to_underlying(DataFormat::UInt8) ? 1 : 0);
    cfg_reg_rmw_tensix<THCON_SEC0_REG0_TileDescriptor_ADDR32, 0, 0x0f>(sf);
    cfg_reg_rmw_tensix<THCON_SEC0_REG1_Unp_LF8_4b_exp_RMW>((sf & 0x1F) == to_underlying(DataFormat::Fp8_e4m3) ? 1 : 0);
    cfg_reg_rmw_tensix<THCON_SEC0_REG2_Out_data_format_RMW>(df);
    TT_SETDMAREG(0, LOWER_HALFWORD(mm_page(id)), 0, LO_16(p_gpr_unpack::TILE_SIZE_A));
    cfg_reg_rmw_tensix<UNP0_ADDR_CTRL_ZW_REG_1_Zstride_RMW>(canonical_unpA_z_stride(df));
    cfg_reg_rmw_tensix<
        UNP0_ADDR_CTRL_XY_REG_1_Ystride_ADDR32,
        UNP0_ADDR_CTRL_XY_REG_0_Ystride_SHAMT,
        UNP0_ADDR_CTRL_XY_REG_1_Ystride_MASK>(canonical_unpA_y_stride(df));
}

inline void mm_unp_copy_setup(const uint32_t cb) {
    const uint32_t id = get_operand_id(cb);
    cfg_reg_rmw_tensix<THCON_SEC0_REG2_Haloize_mode_RMW>(0);
    TTI_SETADCXX(p_setadc::UNP_A, FACE_R_DIM * FACE_C_DIM - 1, 0x0);
    if (mm_dest_direct(id)) {
        ckernel_template(4, 1, TT_OP_UNPACR(SrcA, 0b00010001, 0, 0, 0, 1, 0, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1))
            .program();
    } else {
        ckernel_template t(
            4,
            1,
            TT_OP_UNPACR_NOP(
                SrcB,
                0,
                0,
                p_unpacr_nop::SET_DVALID,
                0,
                p_unpacr_nop::WAIT_LIKE_UNPACR,
                0,
                0,
                p_unpacr_nop::UNP_ZEROSRC));
        t.set_start_op(TT_OP_UNPACR(SrcA, 0b1, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1));
        t.program();
    }
}

inline void mm_unp_copy_tiles(const uint32_t cb, const uint32_t first, const uint32_t n) {
    volatile uint32_t tt_reg_ptr* cfg = get_cfg_pointer();
    const uint32_t id = get_operand_id(cb);
    const uint32_t step = mm_page(id);
    const bool direct = mm_dest_direct(id);
    const uint32_t df = unpack_dst_format[id];
    uint32_t addr = mm_rd(id) + first * step;
    for (uint32_t t = 0; t < n; t++, addr += step) {
        TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111);
        wait_for_next_context(2);
        cfg[unp_cfg_context == 0 ? THCON_SEC0_REG3_Base_address_ADDR32 : THCON_SEC0_REG3_Base_cntx1_address_ADDR32] =
            addr;
        semaphore_post(semaphore::UNPACK_SYNC);
        if (direct) {
            const uint32_t dest_bytes = 16 * (4 + mailbox_read(ThreadId::MathThreadId));
            TTI_SETC16(SRCA_SET_Base_ADDR32, 0x0);
            TT_SETDMAREG(
                0,
                LOWER_HALFWORD(canonical_unpA_z_stride(df) << UNP0_ADDR_CTRL_ZW_REG_1_Zstride_SHAMT),
                0,
                LO_16(p_gpr_unpack::TMP_LO));
            TTI_WRCFG(p_gpr_unpack::TMP_LO, p_cfg::WRCFG_32b, UNP0_ADDR_CTRL_ZW_REG_1_Zstride_ADDR32);
            if (unp_cfg_context == 0) {
                cfg_reg_rmw_tensix<THCON_SEC0_REG2_Unpack_if_sel_cntx0_RMW>(1);
                cfg_reg_rmw_tensix<THCON_SEC0_REG5_Dest_cntx0_address_RMW>(dest_bytes);
            } else {
                cfg_reg_rmw_tensix<THCON_SEC0_REG2_Unpack_if_sel_cntx1_RMW>(1);
                cfg_reg_rmw_tensix<THCON_SEC0_REG5_Dest_cntx1_address_RMW>(dest_bytes);
            }
            t6_semaphore_wait_on_max<p_stall::STALL_UNPACK>(semaphore::UNPACK_TO_DEST);
        }
        TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG);
        ckernel_template::run();
        t6_semaphore_get(semaphore::UNPACK_SYNC);
        if (direct) {
            t6_semaphore_post<p_stall::UNPACK0>(semaphore::UNPACK_TO_DEST);
            TT_SETDMAREG(
                0,
                LOWER_HALFWORD(canonical_unpA_z_stride(df) << UNP0_ADDR_CTRL_ZW_REG_1_Zstride_SHAMT),
                0,
                LO_16(p_gpr_unpack::TMP_LO));
            TTI_WRCFG(p_gpr_unpack::TMP_LO, p_cfg::WRCFG_32b, UNP0_ADDR_CTRL_ZW_REG_1_Zstride_ADDR32);
            if (unp_cfg_context == 0) {
                cfg_reg_rmw_tensix<THCON_SEC0_REG2_Unpack_if_sel_cntx0_RMW>(0);
                cfg_reg_rmw_tensix<THCON_SEC0_REG5_Dest_cntx0_address_RMW>(4 * 16);
            } else {
                cfg_reg_rmw_tensix<THCON_SEC0_REG2_Unpack_if_sel_cntx1_RMW>(0);
                cfg_reg_rmw_tensix<THCON_SEC0_REG5_Dest_cntx1_address_RMW>(4 * 16);
            }
            TTI_SETC16(SRCA_SET_Base_ADDR32, 0x4);
        }
        switch_config_context(unp_cfg_context);
    }
}

}  // namespace
#endif

#ifdef TRISC_MATH
namespace {

constexpr bool MM_HI = MATH_FIDELITY != MathFidelity::LoFi;
constexpr uint32_t MM_PHASES = static_cast<uint32_t>(MATH_FIDELITY);
volatile tt_l1_ptr uint32_t* const mm_fw_flag = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(MEM_L1_ARC_FW_SCRATCH);
uint32_t mm_slow_mop = 0;

inline void mm_math_mop(const uint32_t ct, const uint32_t rt, const bool slow) {
    const bool along_w = ct >= rt;
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 8}, .dest = {.incr = 8}}.set(ADDR_MOD_0);
    addr_mod_t{.srca = {.incr = 16}, .srcb = {.incr = 0, .clr = 0, .cr = 1}, .dest = {.incr = 8}}.set(ADDR_MOD_1);
    addr_mod_t{.srca = {.incr = 0, .clr = 0, .cr = 1}, .srcb = {.incr = 32, .clr = 0, .cr = 1}, .dest = {.incr = 8}}
        .set(ADDR_MOD_2);
    addr_mod_t{
        .srca = {.incr = 32, .clr = 0, .cr = 1},
        .srcb = {.incr = 48, .clr = 0, .cr = 1},
        .dest = {.incr = 0, .clr = 0, .cr = 1}}
        .set(ADDR_MOD_4);
    addr_mod_t{
        .srca = {.incr = 0, .clr = 1, .cr = 1},
        .srcb = {.incr = 0, .clr = 1, .cr = 1},
        .dest = {.incr = 0, .clr = 1, .cr = 1},
        .fidelity = {.incr = MM_HI ? 1u : 0u, .clr = 0}}
        .set(ADDR_MOD_5);
    if (slow) {
        addr_mod_t{
            .srca = {.incr = 0, .clr = 1, .cr = 1},
            .srcb = {.incr = 0, .clr = 1, .cr = 1},
            .dest = {.incr = 0, .clr = 1, .cr = 1},
            .fidelity = {.incr = 0, .clr = 1}}
            .set(ADDR_MOD_6);
    }
    const uint32_t drop = along_w ? p_setrwc::CLR_A : p_setrwc::CLR_B;
    if (!slow) {
        load_replay_buf(0, 16, [along_w] {
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0);
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_2, 0);
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0);
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_4, 0);
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0);
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_2, 0);
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0);
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);
            if constexpr (MM_HI) {
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_5, 0);
            } else if (along_w) {
                TTI_MVMUL(p_setrwc::CLR_A, 0, ADDR_MOD_5, 0);
            } else {
                TTI_MVMUL(p_setrwc::CLR_B, 0, ADDR_MOD_5, 0);
            }
        });
        ckernel_template m(1, MM_HI ? MM_PHASES : 1, lltt::replay_insn(0, 16));
        if constexpr (MM_HI) {
            m.set_end_op(TT_OP_SETRWC(drop, 0, 0, 0, 0, p_setrwc::SET_ABD_F));
        }
        m.program();
    } else {
        load_replay_buf(0, 11, [] {
            TTI_NOP;
            TTI_NOP;
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);
            TTI_NOP;
            TTI_NOP;
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0);
            TTI_NOP;
            TTI_NOP;
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);
            TTI_NOP;
            TTI_NOP;
        });
        ckernel_template m(2, 2, lltt::replay_insn(0, 11), TT_OP_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_2, 0));
        m.set_last_inner_loop_instr(TT_OP_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_4, 0));
        m.set_last_outer_loop_instr(TT_OP_MVMUL(MM_HI ? p_setrwc::CLR_NONE : drop, 0, ADDR_MOD_5, 0));
        m.program();
    }
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_ABD_F);
    math::_configure_default_zero_flag_state_();
}

inline void mm_math_kstep(const uint32_t ct, const uint32_t rt) {
    const uint32_t slow = *mm_fw_flag % 2;
    if (slow != mm_slow_mop) {
        mm_math_mop(ct, rt, slow != 0);
        mm_slow_mop = slow;
    }
    const bool along_w = ct >= rt;
    const uint32_t outer = along_w ? rt : ct;
    const uint32_t inner = along_w ? ct : rt;
    for (uint32_t t = 0; t < outer; t++) {
        for (uint32_t r = 0; r < inner; r++) {
            math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(
                along_w ? ct * t + r : t + r * ct);
            if (MM_HI && slow) {
                for (uint32_t p = 0; p < MM_PHASES; p++) {
                    ckernel_template::run();
                }
                TTI_SETRWC(along_w ? p_setrwc::CLR_A : p_setrwc::CLR_B, 0, 0, 0, 0, p_setrwc::SET_ABD_F);
            } else {
                ckernel_template::run();
            }
            if (r == inner - 1) {
                TTI_SETRWC(along_w ? p_setrwc::CLR_B : p_setrwc::CLR_A, 0, 0, 0, 0, p_setrwc::SET_ABD_F);
            }
        }
    }
}

inline void mm_math_restart(const uint32_t ct, const uint32_t rt) {
    mm_math_mop(ct, rt, false);
    mm_slow_mop = 0;
}

inline void mm_math_srca_to(const uint32_t from_cb, const uint32_t to_cb) {
    const uint32_t df = unpack_dst_format[get_operand_id(to_cb)];
    if (unpack_dst_format[get_operand_id(from_cb)] == df) {
        return;
    }
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::MATH);
    cfg_reg_rmw_tensix<ALU_ACC_CTRL_INT8_math_enabled_RMW>(is_int8_or_int32_format(df) ? 1 : 0);
    math::src_zero_flag_srca_fmt = df;
    math::_configure_default_zero_flag_state_();
}

inline void mm_math_copy_setup(const uint32_t cb) {
    const uint32_t df = unpack_dst_format[get_operand_id(cb)];
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}}.set(ADDR_MOD_3);
    addr_mod_t{.srca = {.incr = 1}, .srcb = {.incr = 0}, .dest = {.incr = 1}}.set(ADDR_MOD_0);
    addr_mod_t{.srca = {.incr = 8}, .srcb = {.incr = 0}, .dest = {.incr = 8}}.set(ADDR_MOD_2);
    const bool add32 =
        (DST_ACCUM_MODE && df != to_underlying(DataFormat::UInt16)) || df == to_underlying(DataFormat::UInt8);
    const uint32_t mv = add32            ? TT_OP_ELWADD(0, 0, p_elwise::SRCB_NO_BCAST, ADDR_MOD_2, 0)
                        : DST_ACCUM_MODE ? TT_OP_MOVA2D(p_mov::DEST_32B_LOW, 0, ADDR_MOD_2, p_mova2d::MOV_8_ROWS, 0)
                                         : TT_OP_MOVA2D(p_mov::DEST_NORM, 0, ADDR_MOD_2, p_mova2d::MOV_8_ROWS, 0);
    ckernel_template m(4, 2, mv);
    m.set_end_op(TT_OP_SETRWC(p_setrwc::CLR_AB, 0, 0, 0, 0, p_setrwc::SET_AB));
    m.program();
    TTI_SETC16(CLR_DVALID_SrcA_Disable_ADDR32, 0);
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_ABD_F);
    math::_configure_copy_zero_flag_state_(df);
}

inline void mm_math_copy_tiles(const uint32_t cb, const uint32_t first_dst, const uint32_t n) {
    const uint32_t id = get_operand_id(cb);
    const uint32_t df = unpack_dst_format[id];
    const bool direct = is_32bit_input(unpack_src_format[id], df);
    const uint32_t m = masked_data_format(df);
    const int wide = static_cast<int>(
        m == to_underlying(DataFormat::Float32) || m == to_underlying(DataFormat::Int32) ||
        m == to_underlying(DataFormat::UInt32));
    for (uint32_t d = first_dst; d < first_dst + n; d++) {
        if (direct) {
            math_unpack_to_dest_math_ready();
            math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::DestReg>(d);
            math::math_unpack_to_dest_tile_ready();
            const uint32_t local = d & ((wide ? 4u : 8u) - 1);
#pragma GCC unroll 0
            for (uint32_t f = 0; f < 4; f++) {
                TT_ZEROACC(p_zeroacc::CLR_16, wide, 1, ADDR_MOD_3, (local << 2) + f);
            }
        } else {
            math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(d);
            ckernel_template::run();
            TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
        }
    }
}

}  // namespace
#endif

#ifdef TRISC_PACK
namespace {

inline void mm_pack_target(const uint32_t cb) {
    const uint32_t id = get_output_id(cb);
    const uint32_t sf = pack_src_format[id];
    const uint32_t df = pack_dst_format[id];
    const uint32_t s = masked_data_format(sf);
    const uint32_t d = masked_data_format(df);
    const bool e4m3 = (df & 0x1F) == to_underlying(DataFormat::Fp8_e4m3);
    const bool wide = sf == to_underlying(DataFormat::Int32) || sf == to_underlying(DataFormat::UInt32) ||
                      sf == to_underlying(DataFormat::Float32);
    const bool narrow = sf == to_underlying(DataFormat::Int8) || sf == to_underlying(DataFormat::UInt8);
    const uint32_t xs = datum_size_in_bytes(s);
    const uint32_t ys = FACE_C_DIM * xs;
    const uint32_t zs = FACE_R_DIM * ys;
    const uint32_t ws = TILE_NUM_FACES * FACE_C_DIM * FACE_R_DIM * xs;
    const bool thr = DST_ACCUM_MODE && IS_BFP_A_FORMAT(d);

    pack_config_u pc;
    pc.val[2] = 0;
    pc.f.uncompress = 1;
    pc.f.out_data_format = d;
    pc.f.in_data_format = e4m3 ? to_underlying(DataFormat::Float16) : s;
    TT_SETDMAREG(0, LOWER_HALFWORD(pc.val[2]), 0, LO_16(p_gpr_pack::TMP_LO));
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::PACK | p_stall::THCON);
    TTI_WRCFG(p_gpr_pack::TMP_LO, p_cfg::WRCFG_32b, THCON_SEC0_REG1_Row_start_section_size_ADDR32 + 2);

    dest_rd_ctrl_u rd;
    rd.val = 0;
    rd.f.PCK_DEST_RD_CTRL_Read_32b_data = wide || DST_ACCUM_MODE;
    rd.f.PCK_DEST_RD_CTRL_Read_int8 = !(DST_ACCUM_MODE || wide) && narrow;
    rd.f.PCK_DEST_RD_CTRL_Read_unsigned = df == to_underlying(DataFormat::UInt8) ? 1 : 0;
    rd.f.PCK_DEST_RD_CTRL_Round_10b_mant =
        ((DST_ACCUM_MODE && sf == to_underlying(DataFormat::Float16)) || (!IS_A_FORMAT(sf) && e4m3)) ? 1 : 0;
    cfg_reg_rmw_tensix<
        PCK_DEST_RD_CTRL_Read_32b_data_ADDR32,
        PCK_DEST_RD_CTRL_Read_32b_data_SHAMT,
        PCK_DEST_RD_CTRL_FIELDS_MASK>(rd.val);

    if (IS_BFP_FORMAT(d)) {
        cfg_reg_rmw_tensix<THCON_SEC0_REG1_Exp_section_size_RMW>(4);
    } else if (d == to_underlying(DataFormat::Lf8) || d == to_underlying(DataFormat::Int8)) {
        TTI_WRCFG(p_gpr::ZERO, p_cfg::WRCFG_32b, THCON_SEC0_REG1_Row_start_section_size_ADDR32);
    }
    cfg_reg_rmw_tensix<THCON_SEC0_REG1_Pac_LF8_4b_exp_RMW>(e4m3 ? 1 : 0);
    TT_SETDMAREG(0, LOWER_HALFWORD(get_local_cb_interface(id).fifo_page_size), 0, LO_16(p_gpr_pack::TILE_HEADER));
    cfg_reg_rmw_tensix<
        THCON_SEC0_REG1_Exp_threshold_ADDR32,
        0,
        THCON_SEC0_REG1_Exp_threshold_en_MASK | THCON_SEC0_REG1_Exp_threshold_MASK>(
        ((thr ? 113u : 0u) << THCON_SEC0_REG1_Exp_threshold_SHAMT) |
        ((thr ? 1u : 0u) << THCON_SEC0_REG1_Exp_threshold_en_SHAMT));
    cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG2_Dstacc_RMW>(e4m3 ? to_underlying(DataFormat::Float16) : s);

    TT_SETDMAREG(0, LOWER_HALFWORD((ys << PCK0_ADDR_CTRL_XY_REG_0_Ystride_SHAMT)), 0, LO_16(p_gpr_pack::TMP0));
    TT_SETDMAREG(0, UPPER_HALFWORD((ys << PCK0_ADDR_CTRL_XY_REG_0_Ystride_SHAMT)), 0, HI_16(p_gpr_pack::TMP0));
    TT_SETDMAREG(0, LOWER_HALFWORD((zs << PCK0_ADDR_CTRL_ZW_REG_0_Zstride_SHAMT)), 0, LO_16(p_gpr_pack::TMP1));
    TT_SETDMAREG(0, UPPER_HALFWORD((ws << PCK0_ADDR_CTRL_ZW_REG_0_Wstride_SHAMT)), 0, HI_16(p_gpr_pack::TMP1));
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::THCON);
    TTI_WRCFG(p_gpr_pack::TMP0, p_cfg::WRCFG_32b, PCK0_ADDR_CTRL_XY_REG_0_Xstride_ADDR32);
    TTI_WRCFG(p_gpr_pack::TMP1, p_cfg::WRCFG_32b, PCK0_ADDR_CTRL_ZW_REG_0_Zstride_ADDR32);
    TTI_WRCFG(p_gpr::ZERO, p_cfg::WRCFG_32b, PCK0_ADDR_CTRL_XY_REG_1_Xstride_ADDR32);
    TTI_WRCFG(p_gpr::ZERO, p_cfg::WRCFG_32b, PCK0_ADDR_CTRL_ZW_REG_1_Zstride_ADDR32);
    TTI_NOP;
    TTI_NOP;
    TTI_SETADCXX(p_setadc::PAC, FACE_C_DIM - 1, 0x0);
}

constexpr uint32_t MM_ACC_OFF = 0;
constexpr uint32_t MM_ACC_ON = 1;
constexpr uint32_t MM_ACC_KEEP = 2;

inline void mm_pack_tiles(const uint32_t cb, const uint32_t n, const uint32_t acc) {
    if (acc != MM_ACC_KEEP) {
        TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::PACK);
        cfg_reg_rmw_tensix<THCON_SEC0_REG1_Pack_L1_Acc_RMW>(acc);
#if !defined(TT_METAL_TTSIM)
        cfg_reg_rmw_tensix<PACK_GLOBAL_CFG_CTL_pack_disable_fast_tile_end_drain_RMW>(acc);
#endif
    }
    const uint32_t id = get_output_id(cb);
    for (uint32_t t = 0; t < n; t++) {
        auto& o = get_local_cb_interface(id);
        const uint32_t a = o.fifo_wr_ptr + o.fifo_wr_tile_ptr - 1;
        o.fifo_wr_tile_ptr += o.fifo_page_size;
        TT_SETADC(p_setadc::PAC, p_setadc::CH_0, p_setadc::SET_W, t);
        program_packer_destination(a);
        ckernel_template::run();
        TTI_SETADCZW(p_setadc::PAC, 0, 0, 0, 0, 0b0101);
    }
}

}  // namespace
#endif

#ifdef TRISC_MATH
namespace {

inline void mm_sfpu_entry() {
    TTI_SFPCONFIG(0, 15, 1);
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}}.set(ADDR_MOD_7);
}

inline __attribute__((always_inline)) void mm_silu_row_head(const bool last) {
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
    if (last) {
        TTI_SFPMAD(0, 1, 4, 1, 0);
        TTI_SFPMAD(0, 1, 2, 0, 0);
    } else {
        TTI_SFPMAD(0, 1, 4, 6, 0);
        TTI_SFPMAD(0, 6, 2, 0, 0);
    }
    TTI_SFPSETEXP(0, 0, 5, 0);
    TTI_SFPADD(10, 5, 10, 0, 0);
}

inline __attribute__((always_inline)) void mm_silu_row_tail(const bool last) {
    if (last) {
        TTI_SFPARECIP(0, 0, 1, 0);
        TTI_SFPMAD(0, 1, 12, 0, 2);
        TTI_SFPGT(0, 0, 9, 1);
        TTI_SFPLOADI(2, 0, 32768);
        TTI_SFPMAD(0, 1, 2, 1, 1);
        TTI_SFPENCC(3, 0, 0, 10);
        TTI_SFPMUL(3, 1, 9, 3, 0);
    } else {
        TTI_SFPARECIP(0, 0, 5, 0);
        TTI_SFPMAD(0, 5, 12, 0, 2);
        TTI_SFPGT(0, 0, 9, 1);
        TTI_SFPLOADI(6, 0, 32768);
        TTI_SFPMAD(0, 5, 6, 5, 1);
        TTI_SFPENCC(3, 0, 0, 10);
        TTI_SFPMUL(3, 5, 9, 3, 0);
    }
    TTI_SFP_STOCH_RND(0, 0, 9, 3, 3, 1);
    TTI_SFPSTORE(3, 0, 7, 0);
    TTI_INCRWC(0, 2, 0, 0);
}

inline void mm_math_swiglu_pair() {
    mm_sfpu_entry();
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_ABD_F);
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_ABD_F);
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_ABD_F);
    TTI_SFPLOADI(0, 0, 16384);
    TTI_SFPCONFIG(0, 12, 0);
    TTI_SFPLOADI(0, 2, 43579);
    TTI_SFPLOADI(0, 8, 16312);
    TTI_SFPCONFIG(0, 13, 0);
    math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(0);
    TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::MATH);
#pragma GCC unroll 0
    for (uint32_t f = 0; f < 4; ++f) {
        TTI_SFPLOADI(2, 2, 14469);
        TTI_SFPLOADI(2, 8, 16256);
        TTI_SFPLOADI(4, 2, 23258);
        TTI_SFPLOADI(4, 8, 13224);
        TTI_SFPLOADI(1, 2, 42008);
        TTI_SFPLOADI(1, 8, 10156);
        TTI_REPLAY(0, 16, 1, 1);
        mm_silu_row_head(false);
        mm_silu_row_tail(false);
#pragma GCC unroll 6
        for (uint32_t r = 0; r < 6; ++r) {
            TTI_REPLAY(0, 16, 0, 0);
            mm_silu_row_tail(false);
        }
        mm_silu_row_head(true);
        mm_silu_row_tail(true);
        TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
        TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
    }
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);

    mm_sfpu_entry();
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_ABD_F);
    math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(0);
    TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::MATH);
#pragma GCC unroll 0
    for (uint32_t f = 0; f < 4; ++f) {
#pragma GCC unroll 0
        for (uint32_t r = 0; r < 8; ++r) {
            TTI_SFPLOAD(3, 0, 7, 0);
            TTI_SFPLOAD(2, 0, 7, 64);
            TTI_SFPMUL(3, 2, 9, 0, 0);
            TTI_SFPSHFT(4080, 0, 1, 5);
            TTI_SFPLOADI(4, 2, 1);
            TTI_SFPAND(1, 4, 1, 1);
            TTI_SFPLOADI(4, 2, 32767);
            TTI_SFPIADD(0, 4, 0, 4);
            TTI_SFPIADD(0, 1, 0, 4);
            TTI_SFPLOADI(1, 0, 65535);
            TTI_SFPAND(0, 1, 0, 1);
            TTI_SFPSETCC(0, 3, 0, 2);
            TTI_SFPSETCC(0, 2, 0, 2);
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
}

}  // namespace
#endif

static_assert(DST_SYNC_MODE == DstSync::SyncHalf);

ALWI void mm_acquire() { MATH((mm_math_dest_take())); }
ALWI void mm_commit() { MATH((mm_math_dest_give())); }
ALWI void mm_wait() { PACK((mm_pack_dest_take())); }
ALWI void mm_release() { PACK((mm_pack_dest_give())); }

ALWI void mm_restart(const uint32_t in1_cb, const uint32_t ct, const uint32_t rt, const uint32_t kt) {
    UNPACK((mm_unp_stream(in1_cb, ct, rt, kt)));
    MATH((mm_math_restart(ct, rt)));
}

ALWI void mm_srca_to(const uint32_t from_cb, const uint32_t to_cb) {
    UNPACK((mm_unp_srca_to(from_cb, to_cb)));
    MATH((mm_math_srca_to(from_cb, to_cb)));
}

ALWI void mm_copy_setup(const uint32_t cb) {
    UNPACK((mm_unp_copy_setup(cb)));
    MATH((mm_math_copy_setup(cb)));
}

ALWI void mm_copy_tiles(const uint32_t cb, const uint32_t first_in, const uint32_t first_dst, const uint32_t n) {
    UNPACK((mm_unp_copy_tiles(cb, first_in, n)));
    MATH((mm_math_copy_tiles(cb, first_dst, n)));
}

ALWI void mm_kstep(
    const uint32_t in0_cb,
    const uint32_t in1_cb,
    const uint32_t i0,
    const uint32_t i1,
    const uint32_t ct,
    const uint32_t rt,
    const uint32_t kt) {
    UNPACK((mm_unp_kstep(in0_cb, in1_cb, i0, i1, ct, rt, kt)));
    MATH((mm_math_kstep(ct, rt)));
}

template <typename PartialsDFB>
FORCE_INLINE void mm_reload(
    PartialsDFB& partials,
    const uint32_t partials_cb,
    const uint32_t in1_cb,
    const uint32_t n,
    const uint32_t ct,
    const uint32_t rt,
    const uint32_t kt) {
    mm_srca_to(in1_cb, partials_cb);
    mm_copy_setup(partials_cb);
    partials.wait_front(static_cast<uint16_t>(n));
    mm_copy_tiles(partials_cb, 0, 0, n);
    partials.pop_front(static_cast<uint16_t>(n));
    mm_srca_to(partials_cb, in1_cb);
    mm_restart(in1_cb, ct, rt, kt);
}

void kernel_main() {
#ifdef MATMUL_DRAM_SHARDED
    const bool is_worker_core = get_arg(args::is_worker_core) == 1;
    if (not is_worker_core) {
        return;
    }
#endif

    constexpr auto in0_block_w = get_arg(args::in0_block_w);
    constexpr auto in0_num_subblocks = get_arg(args::in0_num_subblocks);
    constexpr auto in0_block_num_tiles = get_arg(args::in0_block_num_tiles);
    constexpr auto in0_subblock_num_tiles = get_arg(args::in0_subblock_num_tiles);
    constexpr auto in1_num_subblocks = get_arg(args::in1_num_subblocks);
    constexpr auto in1_block_num_tiles = get_arg(args::in1_block_num_tiles);
    constexpr auto in1_block_w = get_arg(args::in1_block_w);
    constexpr auto num_blocks_inner_dim = get_arg(args::num_blocks_inner_dim);
    constexpr auto num_blocks_w_dim = get_arg(args::num_blocks_w_dim);
    constexpr auto num_blocks_h_dim = get_arg(args::num_blocks_h_dim);
    constexpr auto out_subblock_h = get_arg(args::out_subblock_h);
    constexpr auto out_subblock_w = get_arg(args::out_subblock_w);
    constexpr auto out_subblock_num_tiles = get_arg(args::out_subblock_num_tiles);
    constexpr auto batch = get_arg(args::batch);
    constexpr bool untilize_out = get_arg(args::untilize_out) != 0;
    constexpr bool get_batch_from_reader = static_cast<bool>(get_arg(args::get_batch_from_reader));
    static_assert(!untilize_out, "untilize_out is not used by this kernel");

    constexpr uint32_t in0_dfb_id = dfb::in0;
    constexpr uint32_t in1_dfb_id = dfb::in1;
    constexpr uint32_t out_dfb_id = dfb::out;
    constexpr uint32_t mm_partials_dfb_id = dfb::intermed0;

    DataflowBuffer in0_dfb(in0_dfb_id);
    DataflowBuffer in1_dfb(in1_dfb_id);
    DataflowBuffer mm_partials_dfb(mm_partials_dfb_id);
    DataflowBuffer out_dfb(out_dfb_id);

#ifdef MATMUL_DRAM_SHARDED
    constexpr auto last_subblock_w_valid = get_arg(args::last_subblock_w_valid);
#else
    constexpr uint32_t last_subblock_w_valid = out_subblock_w;
#endif
    constexpr bool last_subblock_padded = last_subblock_w_valid < out_subblock_w;

#ifdef OUT_PINGPONG
    DataflowBuffer partials_b_dfb(dfb::intermed0_b);
    DataflowBuffer out_b_dfb(dfb::out_b);
#ifdef OUT_SWIGLU
    static_assert(out_subblock_h == 1 && out_subblock_w % 2 == 0, "OUT_SWIGLU needs a 1 x even subblock");
    DataflowBuffer swiglu_out_dfb(dfb::swiglu_out);
#endif
#endif

    compute_kernel_hw_startup<SrcOrder::Reverse>(in0_dfb_id, in1_dfb_id, mm_partials_dfb_id);
    mm_restart(in1_dfb_id, out_subblock_w, out_subblock_h, in0_block_w);
    for (uint32_t b = 0; b < batch; b++) {
        if constexpr (get_batch_from_reader) {
            bool is_batch_valid = false;
            UNPACK(is_batch_valid = static_cast<bool>(mailbox_read(ckernel::ThreadId::BriscThreadId));)
            MATH(is_batch_valid = static_cast<bool>(mailbox_read(ckernel::ThreadId::BriscThreadId));)
            PACK(is_batch_valid = static_cast<bool>(mailbox_read(ckernel::ThreadId::BriscThreadId));)
            if (!is_batch_valid) {
                continue;
            }
        }

        for (uint32_t bh = 0; bh < num_blocks_h_dim; ++bh) {
            for (uint32_t bw = 0; bw < num_blocks_w_dim; ++bw) {
                bool enable_reload = false;
#ifdef OUT_PINGPONG
                const bool odd_block = (bw & 1u) != 0;
                auto& partials_sel = odd_block ? partials_b_dfb : mm_partials_dfb;
                const uint32_t partials_sel_id =
                    odd_block ? static_cast<uint32_t>(dfb::intermed0_b) : mm_partials_dfb_id;
                auto& out_sel = odd_block ? out_b_dfb : out_dfb;
                const uint32_t out_sel_id = odd_block ? static_cast<uint32_t>(dfb::out_b) : out_dfb_id;
#else
                auto& partials_sel = mm_partials_dfb;
                constexpr uint32_t partials_sel_id = mm_partials_dfb_id;
                auto& out_sel = out_dfb;
                constexpr uint32_t out_sel_id = out_dfb_id;
#endif

                if constexpr (batch > 1 || num_blocks_h_dim > 1 || num_blocks_w_dim > 1) {
                    PACK((mm_pack_target(partials_sel_id)));
                }

                for (uint32_t block = 0; block < num_blocks_inner_dim; block++) {
                    const bool last_out = block == (num_blocks_inner_dim - 1);

#ifdef IN0_RESIDENT
                    if (bw == 0) {
                        in0_dfb.wait_front((block + 1) * in0_block_num_tiles);
                    }
                    in1_dfb.wait_front(in1_block_num_tiles);
                    uint32_t in0_index_subblock_offset = block * in0_block_num_tiles;
#else
                    in0_dfb.wait_front(in0_block_num_tiles);
                    in1_dfb.wait_front(in1_block_num_tiles);
                    uint32_t in0_index_subblock_offset = 0;
#endif
                    for (uint32_t in0_subblock = 0; in0_subblock < in0_num_subblocks; in0_subblock++) {
                        uint32_t in1_index_subblock_offset = 0;
                        for (uint32_t in1_subblock = 0; in1_subblock < in1_num_subblocks; in1_subblock++) {
                            const bool is_last_in1_subblock_padded =
                                last_subblock_padded && (in1_subblock == in1_num_subblocks - 1);
                            const uint32_t effective_subblock_w =
                                is_last_in1_subblock_padded ? last_subblock_w_valid : out_subblock_w;

                            mm_acquire();
                            if (enable_reload) {
                                mm_reload(
                                    partials_sel,
                                    partials_sel_id,
                                    in1_dfb_id,
                                    out_subblock_num_tiles,
                                    out_subblock_w,
                                    out_subblock_h,
                                    in0_block_w);
                            }

                            uint32_t in0_index = in0_index_subblock_offset;
                            uint32_t in1_index = in1_index_subblock_offset;
                            for (uint32_t inner_dim_idx = 0; inner_dim_idx < in0_block_w; ++inner_dim_idx) {
                                mm_kstep(
                                    in0_dfb_id,
                                    in1_dfb_id,
                                    in0_index,
                                    in1_index,
                                    effective_subblock_w,
                                    out_subblock_h,
                                    in0_block_w);
                                in0_index++;
                                in1_index += in1_block_w;
                            }

                            if (last_out) {
                                mm_commit();
                                out_sel.reserve_back(out_subblock_num_tiles);
                                mm_wait();
                                PACK((mm_pack_target(out_sel_id)));
                                PACK((mm_pack_tiles(out_sel_id, out_subblock_num_tiles, MM_ACC_OFF)));
                                mm_release();
                                out_sel.push_back(out_subblock_num_tiles);
#ifdef OUT_SWIGLU
                                out_sel.wait_front(out_subblock_num_tiles);
                                swiglu_out_dfb.reserve_back(out_subblock_num_tiles / 2);
                                mm_srca_to(in1_dfb_id, out_sel_id);
                                PACK((mm_pack_target(static_cast<uint32_t>(dfb::swiglu_out))));
                                mm_copy_setup(out_sel_id);
                                for (uint32_t p = 0; p < out_subblock_num_tiles / 2; ++p) {
                                    mm_acquire();
                                    mm_copy_tiles(out_sel_id, 2 * p, 0, 2);
                                    MATH((mm_math_swiglu_pair()));
                                    mm_commit();
                                    mm_wait();
                                    PACK((mm_pack_tiles(static_cast<uint32_t>(dfb::swiglu_out), 1, MM_ACC_KEEP)));
                                    mm_release();
                                }
                                swiglu_out_dfb.push_back(out_subblock_num_tiles / 2);
                                out_sel.pop_front(out_subblock_num_tiles);
                                mm_srca_to(out_sel_id, in1_dfb_id);
                                mm_restart(in1_dfb_id, out_subblock_w, out_subblock_h, in0_block_w);
                                PACK((mm_pack_target(out_sel_id)));
#endif
                            } else {
                                mm_commit();
                                partials_sel.reserve_back(out_subblock_num_tiles);
                                mm_wait();
                                PACK((mm_pack_tiles(
                                    partials_sel_id,
                                    out_subblock_num_tiles,
                                    block == 0   ? MM_ACC_OFF
                                    : block == 1 ? MM_ACC_ON
                                                 : MM_ACC_KEEP)));
                                mm_release();
                                partials_sel.push_back(out_subblock_num_tiles);
                            }

                            in1_index_subblock_offset += out_subblock_w;
                        }
                        in0_index_subblock_offset += in0_subblock_num_tiles;
                    }

                    if (block < num_blocks_inner_dim - 2) {
                        for (uint32_t s = 0; s < get_arg(args::out_block_num_tiles); s += out_subblock_num_tiles) {
                            partials_sel.wait_front(out_subblock_num_tiles);
                            partials_sel.pop_front(out_subblock_num_tiles);
                        }
                    }
                    if (block == num_blocks_inner_dim - 2) {
                        enable_reload = true;
                    }

#ifdef IN0_RESIDENT
                    if (bw == num_blocks_w_dim - 1 && block == num_blocks_inner_dim - 1) {
                        in0_dfb.pop_front(num_blocks_inner_dim * in0_block_num_tiles);
                    }
#else
                    in0_dfb.pop_front(in0_block_num_tiles);
#endif
                    in1_dfb.pop_front(in1_block_num_tiles);
                }

                if constexpr (batch > 1 || num_blocks_w_dim > 1 || num_blocks_h_dim > 1) {
                    mm_srca_to(partials_sel_id, in1_dfb_id);
                    mm_restart(in1_dfb_id, out_subblock_w, out_subblock_h, in0_block_w);
                }
            }
        }
    }
}
