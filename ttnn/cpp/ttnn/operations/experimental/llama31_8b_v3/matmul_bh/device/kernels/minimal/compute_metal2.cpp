// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/dfb_binding_token.h"
#include "experimental/kernel_args.h"
#include "ttnn/operations/common/shared_with_host/runtime_length.hpp"

#if defined(FUSE_SWIGLU) || defined(FUSE_BIAS) || defined(FUSE_TERNARY) || defined(SFPU_OP_INIT_ACTIVATION) || \
    defined(TERNARY_B_IS_FLOAT32) || defined(ARCH_QUASAR)
#error "configuration not supported by this kernel"
#endif

constexpr uint32_t MM_ACC_OFF = 0;
constexpr uint32_t MM_ACC_ON = 1;
constexpr uint32_t MM_ACC_KEEP = 2;

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

inline void mm_unp_formats(const uint32_t a_cb, const bool with_b, const uint32_t b_cb) {
    const uint32_t ia = get_operand_id(a_cb);
    const uint32_t sa = unpack_src_format[ia];
    const uint32_t da = unpack_dst_format[ia];
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::UNPACK0);
    cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG0_SrcAUnsigned_RMW>(sa == to_underlying(DataFormat::UInt8) ? 1 : 0);
    cfg_reg_rmw_tensix<THCON_SEC0_REG0_TileDescriptor_ADDR32, 0, 0x0f>(sa);
    cfg_reg_rmw_tensix<THCON_SEC0_REG1_Unp_LF8_4b_exp_RMW>((sa & 0x1F) == to_underlying(DataFormat::Fp8_e4m3) ? 1 : 0);
    cfg_reg_rmw_tensix<THCON_SEC0_REG2_Out_data_format_RMW>(da);
    TT_SETDMAREG(0, LOWER_HALFWORD(mm_page(ia)), 0, LO_16(p_gpr_unpack::TILE_SIZE_A));
    cfg_reg_rmw_tensix<UNP0_ADDR_CTRL_ZW_REG_1_Zstride_RMW>(canonical_unpA_z_stride(da));
    cfg_reg_rmw_tensix<
        UNP0_ADDR_CTRL_XY_REG_1_Ystride_ADDR32,
        UNP0_ADDR_CTRL_XY_REG_0_Ystride_SHAMT,
        UNP0_ADDR_CTRL_XY_REG_1_Ystride_MASK>(canonical_unpA_y_stride(da));
    if (!with_b) {
        return;
    }
    const uint32_t ib = get_operand_id(b_cb);
    const uint32_t sb = unpack_src_format[ib];
    const uint32_t db = unpack_dst_format[ib];
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::UNPACK1);
    cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG0_SrcBUnsigned_RMW>(sb == to_underlying(DataFormat::UInt8) ? 1 : 0);
    cfg_reg_rmw_tensix<THCON_SEC1_REG0_TileDescriptor_ADDR32, 0, 0x0f>(sb);
    cfg_reg_rmw_tensix<THCON_SEC1_REG1_Unp_LF8_4b_exp_RMW>((sb & 0x1F) == to_underlying(DataFormat::Fp8_e4m3) ? 1 : 0);
    cfg_reg_rmw_tensix<THCON_SEC1_REG2_Out_data_format_RMW>(db);
    TT_SETDMAREG(0, LOWER_HALFWORD(mm_page(ib)), 0, LO_16(p_gpr_unpack::TILE_SIZE_B));
    cfg_reg_rmw_tensix<UNP1_ADDR_CTRL_ZW_REG_1_Zstride_RMW>(datum_size_in_bytes(db) * FACE_C_DIM * FACE_R_DIM);
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

inline void mm_math_formats(const uint32_t a_cb, const bool with_b, const uint32_t b_cb) {
    const uint32_t fa = unpack_dst_format[get_operand_id(a_cb)];
    const uint32_t fb = with_b ? unpack_dst_format[get_operand_id(b_cb)] : 0;
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::MATH);
    cfg_reg_rmw_tensix<ALU_ACC_CTRL_INT8_math_enabled_RMW>(
        (is_int8_or_int32_format(fa) || (with_b && is_int8_or_int32_format(fb))) ? 1 : 0);
    math::src_zero_flag_srca_fmt = fa;
    if (with_b) {
        math::src_zero_flag_srcb_fmt = fb;
    }
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

inline void mm_pack_target(const uint32_t cb, const uint32_t acc) {
    if (acc != MM_ACC_KEEP) {
        TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::PACK);
        cfg_reg_rmw_tensix<THCON_SEC0_REG1_Pack_L1_Acc_RMW>(acc);
#if !defined(TT_METAL_TTSIM)
        cfg_reg_rmw_tensix<PACK_GLOBAL_CFG_CTL_pack_disable_fast_tile_end_drain_RMW>(acc);
#endif
    }
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

inline void mm_pack_tile(
    const uint32_t cb, const uint32_t dst_tile, const uint32_t slot, const bool at_slot, const uint32_t acc) {
    if (acc != MM_ACC_KEEP) {
        TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::PACK);
        cfg_reg_rmw_tensix<THCON_SEC0_REG1_Pack_L1_Acc_RMW>(acc);
#if !defined(TT_METAL_TTSIM)
        cfg_reg_rmw_tensix<PACK_GLOBAL_CFG_CTL_pack_disable_fast_tile_end_drain_RMW>(acc);
#endif
    }
    auto& o = get_local_cb_interface(get_output_id(cb));
    uint32_t a;
    if (at_slot) {
        a = o.fifo_wr_ptr + o.fifo_page_size * slot - 1;
    } else {
        a = o.fifo_wr_ptr + o.fifo_wr_tile_ptr - 1;
        o.fifo_wr_tile_ptr += o.fifo_page_size;
    }
    TT_SETADC(p_setadc::PAC, p_setadc::CH_0, p_setadc::SET_W, dst_tile);
    program_packer_destination(a);
    ckernel_template::run();
    TTI_SETADCZW(p_setadc::PAC, 0, 0, 0, 0, 0b0101);
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

ALWI void mm_formats(const uint32_t a_cb, const bool with_b, const uint32_t b_cb) {
    UNPACK((mm_unp_formats(a_cb, with_b, b_cb)));
    MATH((mm_math_formats(a_cb, with_b, b_cb)));
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

inline void mm_blocks(
    const uint32_t in0_cb,
    const uint32_t in1_cb,
    const uint32_t out_cb,
    const uint32_t M_block_tiles,
    const uint32_t N_block_tiles,
    const uint32_t full_N_block_tiles,
    const uint32_t K_block_tiles,
    const uint32_t subblock_h,
    const uint32_t subblock_w,
    const uint32_t in0_base,
    uint32_t acc) {
    uint32_t in0_index_offset = in0_base;
    for (uint32_t M_start = 0; M_start < M_block_tiles; M_start += subblock_h) {
        uint32_t in1_index_offset = 0;
        for (uint32_t N_start = 0; N_start < N_block_tiles; N_start += subblock_w) {
            mm_acquire();
            uint32_t in0_index = in0_index_offset;
            uint32_t in1_index = in1_index_offset;
            for (uint32_t inner_dim = 0; inner_dim < K_block_tiles; inner_dim++) {
                mm_kstep(in0_cb, in1_cb, in0_index, in1_index, subblock_w, subblock_h, K_block_tiles);
                in0_index++;
                in1_index += full_N_block_tiles;
            }
            mm_commit();
            mm_wait();
            uint32_t write_dst_index = 0;
            for (uint32_t h = 0; h < subblock_h; h++) {
                for (uint32_t w = 0; w < subblock_w; w++) {
                    PACK((mm_pack_tile(
                        out_cb, write_dst_index, (M_start + h) * full_N_block_tiles + N_start + w, true, acc)));
                    acc = MM_ACC_KEEP;
                    write_dst_index++;
                }
            }
            mm_release();
            in1_index_offset += subblock_w;
        }
        in0_index_offset += subblock_h * K_block_tiles;
    }
}

inline void mm_copy_out(
    const uint32_t in_cb, const uint32_t out_cb, const uint32_t M_block_tiles, const uint32_t N_block_tiles) {
    DataflowBuffer out(out_cb);
    mm_formats(in_cb, false, 0);
    PACK((mm_pack_target(out_cb, MM_ACC_OFF)));
    UNPACK((mm_unp_copy_setup(in_cb)));
    MATH((mm_math_copy_setup(in_cb)));
    uint32_t tile_id = 0;
    for (uint32_t m = 0; m < M_block_tiles; m++) {
        for (uint32_t n = 0; n < N_block_tiles; n++) {
            mm_acquire();
            mm_wait();
            UNPACK((mm_unp_copy_tiles(in_cb, tile_id, 1)));
            MATH((mm_math_copy_tiles(in_cb, 0, 1)));
            PACK((mm_pack_tile(out_cb, 0, 0, false, MM_ACC_KEEP)));
            mm_commit();
            mm_release();
            tile_id++;
        }
        out.push_back(N_block_tiles);
    }
}

#if defined(DIR_SPLIT)
#define K_FORWARD_AT(n) ((n) < DIR_SPLIT)
#elif defined(DIR_PERIOD)
#define K_FORWARD_AT(n) (((((n) % DIR_PERIOD) / DIR_BLOCK) % 2) == 0)
#endif

#if defined(N_ROTATE) && defined(DIR_PERIOD)
#define N_NEXT_TILE n_next_tile
#else
#define N_NEXT_TILE (n_tile + N_block_tiles)
#endif
void kernel_main() {
    constexpr auto K_num_blocks = get_arg(args::K_num_blocks);
    constexpr auto M_block_tiles = get_arg(args::M_block_tiles);
    constexpr auto K_block_tiles = get_arg(args::K_block_tiles);
    constexpr auto N_block_tiles = get_arg(args::N_block_tiles);
    constexpr auto M_cores = get_arg(args::M_cores);
    constexpr auto N_blocks_per_core = get_arg(args::N_blocks_per_core);
    constexpr auto subblock_h = get_arg(args::subblock_h);
    constexpr auto subblock_w = get_arg(args::subblock_w);
    static_assert(get_arg(args::swiglu_in_k_loop) != 1, "swiglu_in_k_loop is not used by this kernel");

    const auto in0_idx = get_arg(args::in0_idx);
    const auto N_start_tile = get_arg(args::N_start_tile);
    const auto N_end_tile = get_arg(args::N_end_tile);

    const uint32_t M_tiles = get_arg(args::M_tiles);
    [[maybe_unused]] const auto [M_tiles_per_core, M_blocks_per_core, padded_M_tiles, M_start_tile, M_end_tile] =
        ttnn::runtime_length::uniform_split(M_tiles, M_cores, in0_idx, M_block_tiles);
#if defined(N_ROTATE) && defined(DIR_PERIOD)
    uint32_t n_rot_start = 0;
    while (n_rot_start + 1 < N_blocks_per_core && !K_FORWARD_AT(N_start_tile + n_rot_start * N_block_tiles)) {
        n_rot_start++;
    }
#endif

    DataflowBuffer dfb_in0(dfb::in0);
    DataflowBuffer dfb_in1(dfb::in1);
    DataflowBuffer dfb_out(dfb::out);
    DataflowBuffer dfb_intermediate(dfb::intermediate);

    compute_kernel_hw_startup<SrcOrder::Reverse>(dfb::in0, dfb::in1, dfb::intermediate);
    mm_restart(dfb::in1, 1, 1, 1);

    constexpr uint32_t in0_block_num_tiles = M_block_tiles * K_block_tiles;
    constexpr uint32_t in1_block_num_tiles = K_block_tiles * N_block_tiles;
    constexpr uint32_t out_block_num_tiles = M_block_tiles * N_block_tiles;

    bool reuse_in0_block = false;

    for (uint32_t m_block_iter = 0; m_block_iter < M_blocks_per_core; m_block_iter++) {
        uint32_t m_tile = M_start_tile + m_block_iter * M_block_tiles;
        uint32_t m_tile_end = std::min(m_tile + M_block_tiles, M_end_tile);
        const uint32_t current_M_block_tiles = m_tile_end - m_tile;
        const uint32_t current_subblock_h = std::min(current_M_block_tiles, subblock_h);

        for (uint32_t n_block_iter = 0; n_block_iter < N_blocks_per_core; n_block_iter++) {
#if defined(N_ROTATE) && defined(DIR_PERIOD)
            const uint32_t n_block_rot = (n_block_iter + n_rot_start) % N_blocks_per_core;
            uint32_t n_tile = N_start_tile + n_block_rot * N_block_tiles;
            [[maybe_unused]] const uint32_t n_next_tile =
                N_start_tile + ((n_block_iter + 1 + n_rot_start) % N_blocks_per_core) * N_block_tiles;
#else
            uint32_t n_tile = N_start_tile + n_block_iter * N_block_tiles;
#endif
            uint32_t n_tile_end = std::min(n_tile + N_block_tiles, N_end_tile);
            const uint32_t current_N_block_tiles = n_tile_end - n_tile;
            const uint32_t current_subblock_w = std::min(current_N_block_tiles, subblock_w);

            mm_formats(dfb::in1, true, dfb::in0);
            PACK((mm_pack_target(dfb::intermediate, MM_ACC_KEEP)));
            mm_restart(dfb::in1, current_subblock_w, current_subblock_h, K_block_tiles);
            dfb_intermediate.reserve_back(out_block_num_tiles);
            for (uint32_t k_block = 0; k_block < K_num_blocks; k_block++) {
#ifdef IN0_RESIDENT
#ifdef DIR_PERIOD
                const uint32_t in0_k_block = K_FORWARD_AT(n_tile) ? k_block : (K_num_blocks - 1) - k_block;
#else
                const uint32_t in0_k_block = (n_block_iter % 2 == 0) ? k_block : (K_num_blocks - 1) - k_block;
#endif
                dfb_in0.wait_front((in0_k_block + 1) * in0_block_num_tiles);
                const uint32_t in0_base = in0_k_block * in0_block_num_tiles;
#else
                dfb_in0.wait_front(in0_block_num_tiles);
                const uint32_t in0_base = 0;
#endif
                dfb_in1.wait_front(in1_block_num_tiles);

                mm_blocks(
                    dfb::in0,
                    dfb::in1,
                    dfb::intermediate,
                    current_M_block_tiles,
                    current_N_block_tiles,
                    N_block_tiles,
                    K_block_tiles,
                    current_subblock_h,
                    current_subblock_w,
                    in0_base,
                    k_block == 1 ? MM_ACC_ON : MM_ACC_KEEP);

                if (k_block == K_num_blocks - 1) {
#ifdef DIR_PERIOD
                    if (n_block_iter < N_blocks_per_core - 1 && K_FORWARD_AT(n_tile) != K_FORWARD_AT(N_NEXT_TILE)) {
#else
                    if (n_block_iter < N_blocks_per_core - 1) {
#endif
                        reuse_in0_block = true;
                    }
                }
#ifndef IN0_RESIDENT
                if (!reuse_in0_block) {
                    dfb_in0.pop_front(in0_block_num_tiles);
                }
#endif
                dfb_in1.pop_front(in1_block_num_tiles);
                reuse_in0_block = false;
            }

            dfb_intermediate.push_back(out_block_num_tiles);
            dfb_out.reserve_back(out_block_num_tiles);
            dfb_intermediate.wait_front(out_block_num_tiles);
            mm_copy_out(dfb::intermediate, dfb::out, M_block_tiles, N_block_tiles);
            dfb_intermediate.pop_front(out_block_num_tiles);
        }
#ifdef IN0_RESIDENT
        dfb_in0.pop_front(K_num_blocks * in0_block_num_tiles);
#endif
    }
}
