// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/dataflow/dataflow_buffer.h"
#include "ckernel_ops.h"
#include "ckernel_template.h"
#include "experimental/kernel_args.h"

#if !defined(ARCH_BLACKHOLE)
#error "rms_norm_residual compute kernel: unsupported configuration"
#endif

namespace {

using ckernel::BroadcastType;
using ckernel::EltwiseBinaryType;

#ifdef TRISC_UNPACK
template <BroadcastType bcast>
inline void unpack_pair_setup() {
    using namespace ckernel;
    using namespace ckernel::unpacker;
    cfg_reg_rmw_tensix<THCON_SEC0_REG2_Haloize_mode_RMW>(0);
    config_unpacker_x_end<p_setadc::UNP_AB>(FACE_R_DIM);
    constexpr std::uint32_t srca = TT_OP_UNPACR(SrcA, 0b1, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    constexpr std::uint32_t srcb = TT_OP_UNPACR(SrcB, 0b1, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    if constexpr (bcast == BroadcastType::COL) {
        ckernel_template tmp(2, 2, srca);
        tmp.set_start_op(srcb);
        tmp.set_end_ops(TT_OP_SETADCZW(p_setadc::UNP_B, 0, 0, 0, 2, 0b0001), TT_OP_NOP);
        tmp.program();
    } else if constexpr (bcast == BroadcastType::ROW) {
        ckernel_template tmp(2, 2, srcb, srca);
        tmp.set_end_ops(TT_OP_SETADCZW(p_setadc::UNP_B, 0, 0, 0, 0, 0b0001), TT_OP_NOP);
        tmp.program();
    } else {
        ckernel_template tmp(2, 2, srca, srcb);
        tmp.set_end_op(TT_OP_NOP);
        tmp.program();
    }
}

inline void unpack_format_a(std::uint32_t in) {
    using namespace ckernel;
    using namespace ckernel::unpacker;
    const std::uint32_t src = unpack_src_format[in];
    const std::uint32_t dst = unpack_dst_format[in];
    const std::uint32_t tile_size = get_local_cb_interface(in).fifo_page_size;
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::UNPACK0);
    cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG0_SrcAUnsigned_RMW>((src == to_underlying(DataFormat::UInt8)) ? 1 : 0);
    cfg_reg_rmw_tensix<THCON_SEC0_REG0_TileDescriptor_ADDR32, 0, 0x0f>(src);
    cfg_reg_rmw_tensix<THCON_SEC0_REG1_Unp_LF8_4b_exp_RMW>(
        ((src & 0x1F) == static_cast<std::uint32_t>(DataFormat::Fp8_e4m3)) ? 1 : 0);
    cfg_reg_rmw_tensix<THCON_SEC0_REG2_Out_data_format_RMW>(dst);
    TT_SETDMAREG(0, LOWER_HALFWORD(tile_size), 0, LO_16(p_gpr_unpack::TILE_SIZE_A));
    cfg_reg_rmw_tensix<UNP0_ADDR_CTRL_ZW_REG_1_Zstride_RMW>(canonical_unpA_z_stride(dst));
    cfg_reg_rmw_tensix<
        UNP0_ADDR_CTRL_XY_REG_1_Ystride_ADDR32,
        UNP0_ADDR_CTRL_XY_REG_0_Ystride_SHAMT,
        UNP0_ADDR_CTRL_XY_REG_1_Ystride_MASK>(canonical_unpA_y_stride(dst));
}

inline void unpack_format_b(std::uint32_t in) {
    using namespace ckernel;
    using namespace ckernel::unpacker;
    const std::uint32_t src = unpack_src_format[in];
    const std::uint32_t dst = unpack_dst_format[in];
    const std::uint32_t tile_size = get_local_cb_interface(in).fifo_page_size;
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::UNPACK1);
    cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG0_SrcBUnsigned_RMW>((src == to_underlying(DataFormat::UInt8)) ? 1 : 0);
    cfg_reg_rmw_tensix<THCON_SEC1_REG0_TileDescriptor_ADDR32, 0, 0x0f>(src);
    cfg_reg_rmw_tensix<THCON_SEC1_REG1_Unp_LF8_4b_exp_RMW>(
        ((src & 0x1F) == static_cast<std::uint32_t>(DataFormat::Fp8_e4m3)) ? 1 : 0);
    cfg_reg_rmw_tensix<THCON_SEC1_REG2_Out_data_format_RMW>(dst);
    TT_SETDMAREG(0, LOWER_HALFWORD(tile_size), 0, LO_16(p_gpr_unpack::TILE_SIZE_B));
    cfg_reg_rmw_tensix<UNP1_ADDR_CTRL_ZW_REG_1_Zstride_RMW>(datum_size_in_bytes(dst) * FACE_C_DIM * FACE_R_DIM);
}

inline bool unpack_format_differs(std::uint32_t old_in, std::uint32_t new_in) {
    return (unpack_src_format[old_in] != unpack_src_format[new_in]) ||
           (unpack_dst_format[old_in] != unpack_dst_format[new_in]);
}

inline void unpack_rowsum_setup() {
    using namespace ckernel;
    using namespace ckernel::unpacker;
    cfg_reg_rmw_tensix<THCON_SEC0_REG2_Haloize_mode_RMW>(1);
    TT_SETADCXX(p_setadc::UNP_B, MAX_NUM_FACES * FACE_R_DIM * FACE_C_DIM - 1, 0x0);
    TT_SETADCXX(p_setadc::UNP_A, MAX_NUM_FACES * FACE_R_DIM * FACE_C_DIM - 1, 0x0);
    load_replay_buf(0, 2, [] {
        TTI_UNPACR(SrcA, 0b0, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
        TTI_UNPACR(SrcB, 0b0, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    });
    ckernel_template tmp(1, 1, lltt::replay_insn(0, 2));
    tmp.program();
}

inline void unpack_pair(std::uint32_t in_a, std::uint32_t in_b, std::uint32_t tile_a, std::uint32_t tile_b) {
    using namespace ckernel;
    using namespace ckernel::unpacker;
    const std::uint32_t address_a =
        get_local_cb_interface(in_a).fifo_rd_ptr - 1 + get_local_cb_interface(in_a).fifo_page_size * tile_a;
    const std::uint32_t address_b =
        get_local_cb_interface(in_b).fifo_rd_ptr - 1 + get_local_cb_interface(in_b).fifo_page_size * tile_b;
    TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111);
    volatile std::uint32_t tt_reg_ptr* cfg = get_cfg_pointer();
    wait_for_next_context(2);
    if (0 == unp_cfg_context) {
        cfg[THCON_SEC0_REG3_Base_address_ADDR32] = address_a;
        cfg[THCON_SEC1_REG3_Base_address_ADDR32] = address_b;
    } else {
        cfg[THCON_SEC0_REG3_Base_cntx1_address_ADDR32] = address_a;
        cfg[THCON_SEC1_REG3_Base_cntx1_address_ADDR32] = address_b;
    }
    semaphore_post(semaphore::UNPACK_SYNC);
    TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG);
    ckernel_template::run();
    t6_semaphore_get(semaphore::UNPACK_SYNC);
    switch_config_context(unp_cfg_context);
}
#endif

#ifdef TRISC_MATH
template <EltwiseBinaryType op, BroadcastType bcast>
inline void math_pair_setup() {
    using namespace ckernel;
    using namespace ckernel::math;
    constexpr bool is_mul = op == EltwiseBinaryType::ELWMUL;
    constexpr std::uint8_t srcb_incr = (bcast == BroadcastType::ROW) ? 0 : MAX_FPU_ROWS;
    addr_mod_t{
        .srca = {.incr = MAX_FPU_ROWS},
        .srcb = {.incr = srcb_incr},
        .dest = {.incr = MAX_FPU_ROWS},
    }
        .set(ADDR_MOD_0);
    addr_mod_t{
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 0},
    }
        .set(ADDR_MOD_1);
    addr_mod_t{
        .srca = {.incr = 0, .clr = 1},
        .srcb = {.incr = 0, .clr = 1},
        .dest = {.incr = 0, .clr = 0, .cr = 1},
        .fidelity = {.incr = is_mul ? 1 : 0}}
        .set(ADDR_MOD_2);
    addr_mod_t{
        .srca = {.incr = 0, .clr = 1},
        .srcb = {.incr = 0, .clr = 1},
        .dest = {.incr = MAX_FPU_ROWS, .clr = 0, .cr = 0, .c_to_cr = 1},
        .fidelity = {.incr = 0, .clr = 1}}
        .set(ADDR_MOD_3);
    constexpr auto srcb_mode = (bcast == BroadcastType::COL)   ? p_elwise::SRCB_BCAST_COL
                               : (bcast == BroadcastType::ROW) ? p_elwise::SRCB_BCAST_ROW
                                                               : p_elwise::SRCB_NO_BCAST;
    constexpr auto clr_src = (bcast == BroadcastType::COL) ? p_setrwc::CLR_A : p_setrwc::CLR_AB;
    if constexpr (is_mul) {
        ckernel_template tmp(2, 2, TT_OP_ELWMUL(0, 0, srcb_mode, ADDR_MOD_0, 0));
        tmp.set_last_inner_loop_instr(TT_OP_ELWMUL(0, 0, srcb_mode, ADDR_MOD_2, 0));
        tmp.set_last_outer_loop_instr(TT_OP_ELWMUL(clr_src, 0, srcb_mode, ADDR_MOD_3, 0));
        tmp.program();
    } else {
        ckernel_template tmp(4, 2, TT_OP_ELWADD(0, 0, srcb_mode, ADDR_MOD_0, 0));
        tmp.set_end_op(TT_OP_SETRWC(clr_src, p_setrwc::CR_AB, 0, 0, 0, p_setrwc::SET_AB));
        tmp.program();
    }
    TTI_SETC16(CLR_DVALID_SrcA_Disable_ADDR32, 0);
    reset_counters(p_setrwc::SET_ABD_F);
    _configure_default_zero_flag_state_();
}

template <EltwiseBinaryType op, BroadcastType bcast>
inline void math_pair(std::uint32_t dst_index) {
    using namespace ckernel;
    using namespace ckernel::math;
    set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_index);
    if constexpr (op == EltwiseBinaryType::ELWADD) {
        ckernel_template::run();
    } else if constexpr (bcast == BroadcastType::COL) {
#pragma GCC unroll 0
        for (std::uint32_t face_row = 0; face_row < 2; face_row++) {
#pragma GCC unroll 0
            for (std::uint32_t i = 0; i < 2; i++) {
                ckernel_template::run();
            }
            TTI_SETRWC(p_setrwc::CLR_B, 0, 0, 0, 0, 0);
        }
    } else {
#pragma GCC unroll 0
        for (std::uint32_t i = 0; i < 4; i++) {
            ckernel_template::run();
        }
    }
    clear_dst_reg_addr();
}

inline void math_rowsum_setup() {
    using namespace ckernel;
    using namespace ckernel::math;
    addr_mod_t{
        .srcb = {.incr = 8},
        .dest = {.incr = 8},
    }
        .set(ADDR_MOD_0);
    addr_mod_t{
        .srcb = {.incr = 24},
        .dest = {.incr = 24},
    }
        .set(ADDR_MOD_1);
    addr_mod_t{
        .srca = {.incr = 16},
        .srcb = {.incr = 16, .cr = 1},
        .dest = {.cr = 1},
    }
        .set(ADDR_MOD_2);
    addr_mod_t{
        .srca = {.clr = 1, .cr = 1},
        .srcb = {.clr = 1, .cr = 1},
        .dest = {.clr = 1, .cr = 1},
        .fidelity = {.incr = 1u},
    }
        .set(ADDR_MOD_3);
    constexpr std::uint32_t replay_start = 0;
    load_replay_buf(replay_start, 8, [] {
        TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);
        TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0);
        TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);
        TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_2, 0);
        TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);
        TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0);
        TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);
        TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_3, 0);
    });
    ckernel_template tmp(1, 2, lltt::replay_insn(replay_start, 8));
    tmp.set_end_op(TT_OP_SETRWC(p_setrwc::CLR_AB, 0, 0, 0, 0, p_setrwc::SET_ABD_F));
    tmp.program();
    TTI_SETC16(CLR_DVALID_SrcA_Disable_ADDR32, 0);
    reset_counters(p_setrwc::SET_ABD_F);
    _configure_default_zero_flag_state_();
}

inline void math_rowsum(std::uint32_t dst_index) {
    using namespace ckernel;
    using namespace ckernel::math;
    set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_index);
    ckernel_template::run();
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_ABD);
}

inline void math_rsqrt_setup() {
    using namespace ckernel;
    using namespace ckernel::math;
    TTI_SFPCONFIG(0, 15, 1);
    addr_mod_t{
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 0},
    }
        .set(ADDR_MOD_7);
    reset_counters(p_setrwc::SET_ABD_F);
    reset_counters(p_setrwc::SET_ABD_F);
    reset_counters(p_setrwc::SET_ABD_F);
    TTI_SFPLOADI(0, 2, 4256);
    TTI_SFPLOADI(0, 8, 24337);
    TTI_SFPCONFIG(0, 12, 0);
    TTI_SFPLOADI(0, 2, 5321);
    TTI_SFPLOADI(0, 8, 16402);
    TTI_SFPCONFIG(0, 13, 0);
    TTI_SFPLOADI(0, 2, 13862);
    TTI_SFPLOADI(0, 8, 16400);
    TTI_SFPCONFIG(0, 14, 0);
}

inline void math_rsqrt_tail() {
    TTI_SFPSETCC(0, 3, 0, 2);
    TTI_SFPMAD(2, 5, 0, 0, 0);
    TTI_SFPPUSHC(0, 0, 0, 0);
    TTI_SFPGT(0, 1, 9, 1);
    TTI_SFPLOADI(0, 0, 32704);
    TTI_SFPPOPC(0, 0, 0, 0);
    TTI_SFPCOMPC(0, 0, 0, 0);
    TTI_SFPMOV(0, 4, 0, 0);
}

inline void math_rsqrt_row_end() {
    TTI_SFPENCC(3, 0, 0, 10);
    TTI_SFPSTORE(0, 0, 7, 0);
    TTI_INCRWC(0, 2, 0, 0);
}

inline void math_rsqrt_row_again() {
    TTI_REPLAY(0, 16, 0, 0);
    math_rsqrt_tail();
    math_rsqrt_row_end();
}

inline void math_rsqrt(std::uint32_t dst_index) {
    using namespace ckernel;
    using namespace ckernel::math;
    set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_index);
    TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::MATH);
#pragma GCC unroll 0
    for (int face = 0; face < 4; face++) {
        TTI_REPLAY(0, 16, 1, 1);
        TTI_SFPLOAD(1, 0, 7, 0);
        TTI_SFPSHFT(4095, 1, 0, 5);
        TTI_SFPIADD(0, 12, 0, 6);
        TTI_SFPMUL(1, 0, 9, 2, 0);
        TTI_SFPMUL(0, 2, 9, 2, 1);
        TTI_SFPLOADI(3, 0, 32640);
        TTI_SFPADD(10, 14, 2, 4, 0);
        TTI_SFPMAD(2, 4, 13, 2, 0);
        TTI_SFPMUL(0, 2, 9, 0, 0);
        TTI_SFPMUL(1, 0, 9, 2, 0);
        TTI_SFPMAD(0, 2, 10, 2, 1);
        TTI_SFPDIVP2(4095, 0, 5, 1);
        TTI_SFPMOV(0, 1, 4, 2);
        TTI_SFPIADD(0, 3, 4, 6);
        TTI_SFPSETCC(0, 4, 0, 2);
        TTI_SFPSHFT(1, 1, 3, 7);
        math_rsqrt_tail();
        math_rsqrt_row_end();
        math_rsqrt_row_again();
        math_rsqrt_row_again();
        math_rsqrt_row_again();
        math_rsqrt_row_again();
        math_rsqrt_row_again();
        math_rsqrt_row_again();
        math_rsqrt_row_again();
        inc_dst_addr<8>();
        inc_dst_addr<8>();
    }
    clear_dst_reg_addr();
}

inline void math_format_a(std::uint32_t format) {
    using namespace ckernel;
    using namespace ckernel::math;
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::MATH);
    cfg_reg_rmw_tensix<ALU_ACC_CTRL_INT8_math_enabled_RMW>(is_int8_or_int32_format(format));
    src_zero_flag_srca_fmt = format;
    _configure_default_zero_flag_state_();
}

inline void math_format_b(std::uint32_t format) {
    using namespace ckernel;
    using namespace ckernel::math;
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::MATH);
    cfg_reg_rmw_tensix<ALU_ACC_CTRL_INT8_math_enabled_RMW>(is_int8_or_int32_format(format));
    src_zero_flag_srcb_fmt = format;
    _configure_default_zero_flag_state_();
}

inline void math_format_ab(std::uint32_t format_a, std::uint32_t format_b) {
    using namespace ckernel;
    using namespace ckernel::math;
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::MATH);
    cfg_reg_rmw_tensix<ALU_ACC_CTRL_INT8_math_enabled_RMW>(
        is_int8_or_int32_format(format_a) || is_int8_or_int32_format(format_b));
    src_zero_flag_srca_fmt = format_a;
    src_zero_flag_srcb_fmt = format_b;
    _configure_default_zero_flag_state_();
}

inline void math_formats(std::uint32_t old_a, std::uint32_t new_a, std::uint32_t old_b, std::uint32_t new_b) {
    const bool a_changes = unpack_dst_format[old_a] != unpack_dst_format[new_a];
    const bool b_changes = unpack_dst_format[old_b] != unpack_dst_format[new_b];
    if (a_changes && b_changes) {
        math_format_ab(unpack_dst_format[new_a], unpack_dst_format[new_b]);
    } else if (a_changes) {
        math_format_a(unpack_dst_format[new_a]);
    } else if (b_changes) {
        math_format_b(unpack_dst_format[new_b]);
    }
}

inline void math_dest_done() {
    using namespace ckernel;
    using namespace ckernel::math;
    set_math_semaphores();
    math_sync_tile_dst_index = 0;
    dest_section_flip();
}
#endif

#ifdef TRISC_PACK
inline void pack_one(std::uint32_t dst_index, std::uint32_t out) {
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

inline void pack_row_mask_on(std::uint32_t out) {
    using namespace ckernel;
    using namespace ckernel::packer;
    const ckernel::TensorShape tensor_shape = get_output_tensor_shape(out);
    pck_edge_offset_u pack_edge_offset = {.val = 0};
    pack_edge_offset.f.mask = 0x0;
    pack_edge_offset.f.tile_row_set_select_pack0 = 1;
    const std::uint32_t edge_offset_sec1_mask = 0x0001;
    const std::uint32_t row_set_mapping_1 = 0x55555555;
    TTI_SETDMAREG(0, LOWER_HALFWORD(pack_edge_offset.val), 0, LO_16(p_gpr_pack::TMP0));
    TTI_SETDMAREG(0, UPPER_HALFWORD(pack_edge_offset.val), 0, HI_16(p_gpr_pack::TMP0));
    TTI_SETDMAREG(0, LOWER_HALFWORD(edge_offset_sec1_mask), 0, LO_16(p_gpr_pack::TMP_LO));
    TTI_SETDMAREG(0, LOWER_HALFWORD(row_set_mapping_1), 0, LO_16(p_gpr_pack::TMP1));
    TTI_SETDMAREG(0, UPPER_HALFWORD(row_set_mapping_1), 0, HI_16(p_gpr_pack::TMP1));
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::PACK);
    cfg_reg_rmw_tensix<PACK_COUNTERS_SEC0_pack_reads_per_xy_plane_RMW>(tensor_shape.face_r_dim);
    TTI_WRCFG(p_gpr_pack::TMP0, p_cfg::WRCFG_32b, PCK_EDGE_OFFSET_SEC0_mask_ADDR32);
    TTI_WRCFG(p_gpr_pack::TMP_LO, p_cfg::WRCFG_32b, PCK_EDGE_OFFSET_SEC1_mask_ADDR32);
    TTI_WRCFG(p_gpr_pack::TMP1, p_cfg::WRCFG_32b, TILE_ROW_SET_MAPPING_1_row_set_mapping_0_ADDR32);
    const std::uint32_t face_set_mapping = [&tensor_shape] {
        if (tensor_shape.num_faces_r_dim == 1 && tensor_shape.num_faces_c_dim == 1) {
            return 0x55555555;
        } else if (tensor_shape.num_faces_r_dim == 1 && tensor_shape.num_faces_c_dim == 2) {
            return 0x11111111;
        } else if (tensor_shape.num_faces_r_dim == 2 && tensor_shape.num_faces_c_dim == 1) {
            return 0x55555555;
        } else {
            return 0x11111111;
        }
    }();
    TTI_WRCFG(p_gpr::ZERO, p_cfg::WRCFG_32b, TILE_ROW_SET_MAPPING_0_row_set_mapping_0_ADDR32);
    cfg_reg_rmw_tensix<TILE_FACE_SET_MAPPING_0_face_set_mapping_0_ADDR32, 0, 0xffffffff>(face_set_mapping);
    cfg_reg_rmw_tensix<PCK_EDGE_TILE_FACE_SET_SELECT_select_ADDR32, 0, 0x1ff>(0x100);
    TTI_NOP;
    TTI_NOP;
}

inline void pack_row_mask_off() {
    using namespace ckernel;
    using namespace ckernel::packer;
    constexpr std::uint32_t all_rows = 0xffff;
    TTI_SETDMAREG(0, all_rows, 0, LO_16(p_gpr_pack::TMP0));
    TTI_SETDMAREG(0, 0, 0, HI_16(p_gpr_pack::TMP0));
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::PACK);
    TTI_WRCFG(p_gpr_pack::TMP0, p_cfg::WRCFG_32b, PCK_EDGE_OFFSET_SEC0_mask_ADDR32);
    TTI_WRCFG(p_gpr_pack::TMP0, p_cfg::WRCFG_32b, PCK_EDGE_OFFSET_SEC1_mask_ADDR32);
    TTI_WRCFG(p_gpr::ZERO, p_cfg::WRCFG_32b, TILE_ROW_SET_MAPPING_1_row_set_mapping_0_ADDR32);
    cfg_reg_rmw_tensix<PCK_EDGE_TILE_FACE_SET_SELECT_enable_RMW>(0);
    cfg_reg_rmw_tensix<PACK_COUNTERS_SEC0_pack_reads_per_xy_plane_RMW>(1);
    TTI_NOP;
    TTI_NOP;
}

inline void pack_dest_done() {
    using namespace ckernel;
    using namespace ckernel::packer;
    TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::PACK);
    TT_ZEROACC(p_zeroacc::CLR_HALF, 1, 0, ADDR_MOD_1, dest_offset_id % 2);
    t6_semaphore_get<p_stall::NONE>(semaphore::MATH_PACK);
    flip_packer_dest_offset_id();
    select_packer_dest_registers<DstSync::SyncHalf>();
}
#endif

inline void set_formats([[maybe_unused]] std::uint32_t in_a, [[maybe_unused]] std::uint32_t in_b) {
    UNPACK((unpack_format_a(in_a)));
    UNPACK((unpack_format_b(in_b)));
    MATH((math_format_ab(unpack_dst_format[in_a], unpack_dst_format[in_b])));
}

inline void set_formats(
    [[maybe_unused]] std::uint32_t old_a,
    [[maybe_unused]] std::uint32_t new_a,
    [[maybe_unused]] std::uint32_t old_b,
    [[maybe_unused]] std::uint32_t new_b) {
#ifdef TRISC_UNPACK
    if (unpack_format_differs(old_a, new_a)) {
        unpack_format_a(new_a);
    }
    if (unpack_format_differs(old_b, new_b)) {
        unpack_format_b(new_b);
    }
#endif
    MATH((math_formats(old_a, new_a, old_b, new_b)));
}

inline void set_format_b([[maybe_unused]] std::uint32_t old_b, [[maybe_unused]] std::uint32_t new_b) {
#ifdef TRISC_UNPACK
    if (unpack_format_differs(old_b, new_b)) {
        unpack_format_b(new_b);
    }
#endif
#ifdef TRISC_MATH
    if (unpack_dst_format[old_b] != unpack_dst_format[new_b]) {
        math_format_b(unpack_dst_format[new_b]);
    }
#endif
}

inline void set_pack_format([[maybe_unused]] std::uint32_t out) {
    PACK((ckernel::packer::reconfig_packer_data_format<true>(
        pack_src_format[out],
        pack_dst_format[out],
        get_local_cb_interface(out).fifo_page_size,
        pack_tile_c_dim[out],
        pack_tile_num_faces[out],
        false)));
}

inline void add_setup([[maybe_unused]] std::uint32_t in_a, [[maybe_unused]] std::uint32_t in_b) {
    MATH((math_pair_setup<EltwiseBinaryType::ELWADD, BroadcastType::NONE>()));
    UNPACK((unpack_pair_setup<BroadcastType::NONE>()));
}

inline void add_pair(
    [[maybe_unused]] std::uint32_t in_a,
    [[maybe_unused]] std::uint32_t in_b,
    [[maybe_unused]] std::uint32_t tile_a,
    [[maybe_unused]] std::uint32_t tile_b,
    [[maybe_unused]] std::uint32_t dst_index) {
    UNPACK((unpack_pair(in_a, in_b, tile_a, tile_b)));
    MATH((math_pair<EltwiseBinaryType::ELWADD, BroadcastType::NONE>(dst_index)));
}

inline void mul_setup([[maybe_unused]] std::uint32_t in_a, [[maybe_unused]] std::uint32_t in_b) {
    MATH((math_pair_setup<EltwiseBinaryType::ELWMUL, BroadcastType::NONE>()));
    UNPACK((unpack_pair_setup<BroadcastType::NONE>()));
}

inline void mul_pair(
    [[maybe_unused]] std::uint32_t in_a,
    [[maybe_unused]] std::uint32_t in_b,
    [[maybe_unused]] std::uint32_t tile_a,
    [[maybe_unused]] std::uint32_t tile_b,
    [[maybe_unused]] std::uint32_t dst_index) {
    UNPACK((unpack_pair(in_a, in_b, tile_a, tile_b)));
    MATH((math_pair<EltwiseBinaryType::ELWMUL, BroadcastType::NONE>(dst_index)));
}

inline void mul_cols_setup([[maybe_unused]] std::uint32_t in_a, [[maybe_unused]] std::uint32_t in_b) {
    MATH((math_pair_setup<EltwiseBinaryType::ELWMUL, BroadcastType::COL>()));
    UNPACK((unpack_pair_setup<BroadcastType::COL>()));
}

inline void mul_cols_pair(
    [[maybe_unused]] std::uint32_t in_a,
    [[maybe_unused]] std::uint32_t in_b,
    [[maybe_unused]] std::uint32_t tile_a,
    [[maybe_unused]] std::uint32_t tile_b,
    [[maybe_unused]] std::uint32_t dst_index) {
    MATH((math_pair<EltwiseBinaryType::ELWMUL, BroadcastType::COL>(dst_index)));
    UNPACK((unpack_pair(in_a, in_b, tile_a, tile_b)));
}

inline void mul_rows_setup([[maybe_unused]] std::uint32_t in_a, [[maybe_unused]] std::uint32_t in_b) {
    MATH((math_pair_setup<EltwiseBinaryType::ELWMUL, BroadcastType::ROW>()));
    UNPACK((unpack_pair_setup<BroadcastType::ROW>()));
}

inline void mul_rows_pair(
    [[maybe_unused]] std::uint32_t in_a,
    [[maybe_unused]] std::uint32_t in_b,
    [[maybe_unused]] std::uint32_t tile_a,
    [[maybe_unused]] std::uint32_t tile_b,
    [[maybe_unused]] std::uint32_t dst_index) {
    MATH((math_pair<EltwiseBinaryType::ELWMUL, BroadcastType::ROW>(dst_index)));
    UNPACK((unpack_pair(in_a, in_b, tile_a, tile_b)));
}

inline void dest_acquire() { MATH((ckernel::math::math_dest_wait())); }

inline void dest_commit() { MATH((math_dest_done())); }

inline void dest_wait() {
    PACK((TTI_SEMWAIT(
        ckernel::p_stall::STALL_TDMA | ckernel::p_stall::STALL_SYNC,
        ckernel::semaphore::t6_sem(ckernel::semaphore::MATH_PACK),
        ckernel::p_stall::STALL_ON_ZERO)));
}

inline void dest_release() { PACK((pack_dest_done())); }

inline void pack_dest([[maybe_unused]] std::uint32_t dst_index, [[maybe_unused]] std::uint32_t out) {
    PACK((pack_one(dst_index, out)));
}

inline void rowsum_setup(
    [[maybe_unused]] std::uint32_t in, [[maybe_unused]] std::uint32_t scaler, [[maybe_unused]] std::uint32_t out) {
    UNPACK((unpack_rowsum_setup()));
    MATH((math_rowsum_setup()));
    PACK((pack_row_mask_on(out)));
}

inline void rowsum_pair(
    [[maybe_unused]] std::uint32_t in,
    [[maybe_unused]] std::uint32_t scaler,
    [[maybe_unused]] std::uint32_t tile,
    [[maybe_unused]] std::uint32_t scaler_tile,
    [[maybe_unused]] std::uint32_t dst_index) {
    MATH((math_rowsum(dst_index)));
    UNPACK((unpack_pair(scaler, in, scaler_tile, tile)));
}

inline void rowsum_done() { PACK((pack_row_mask_off())); }

inline void rsqrt_setup() { MATH((math_rsqrt_setup())); }

inline void rsqrt_dest([[maybe_unused]] std::uint32_t dst_index) { MATH((math_rsqrt(dst_index))); }

inline void skip_front(DataflowBuffer& buffer, std::uint16_t tiles) {
    buffer.wait_front(tiles);
    buffer.pop_front(tiles);
}

inline void set_format_a([[maybe_unused]] std::uint32_t old_a, [[maybe_unused]] std::uint32_t new_a) {
#ifdef TRISC_UNPACK
    if (unpack_format_differs(old_a, new_a)) {
        unpack_format_a(new_a);
    }
#endif
#ifdef TRISC_MATH
    if (unpack_dst_format[old_a] != unpack_dst_format[new_a]) {
        math_format_a(unpack_dst_format[new_a]);
    }
#endif
}

#ifdef TRISC_MATH
inline void math_scale_setup() {
    using namespace ckernel;
    using namespace ckernel::math;
    TTI_SFPCONFIG(0, 15, 1);
    addr_mod_t{
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 0},
    }
        .set(ADDR_MOD_7);
    reset_counters(p_setrwc::SET_ABD_F);
}

template <std::uint32_t factor>
inline void math_scale(std::uint32_t dst_index) {
    using namespace ckernel;
    using namespace ckernel::math;
    set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_index);
    TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::MATH);
#pragma GCC unroll 0
    for (int face = 0; face < 4; face++) {
        TTI_REPLAY(8, 8, 1, 1);
        TTI_SFPLOAD(0, 0, 7, 0);
        TTI_SFPMULI(factor, 0, 0);
        TTI_SFPSTORE(0, 0, 7, 0);
        TTI_INCRWC(0, 2, 0, 0);
        TTI_SFPLOAD(0, 0, 7, 0);
        TTI_SFPMULI(factor, 0, 0);
        TTI_SFPSTORE(0, 0, 7, 0);
        TTI_INCRWC(0, 2, 0, 0);
        TTI_REPLAY(8, 8, 0, 0);
        TTI_REPLAY(8, 8, 0, 0);
        TTI_REPLAY(8, 8, 0, 0);
        inc_dst_addr<8>();
        inc_dst_addr<8>();
    }
    clear_dst_reg_addr();
}
#endif

inline void scale_setup() { MATH((math_scale_setup())); }

template <std::uint32_t factor>
inline void scale_dest([[maybe_unused]] std::uint32_t dst_index) {
    MATH((math_scale<factor>(dst_index)));
}

}  // namespace

void kernel_main() {
    const uint32_t NCHt = get_arg(args::NCHt);
    constexpr uint32_t Wt = get_arg(args::Wt);
    constexpr uint32_t block_size = get_arg(args::block_size);
    constexpr uint32_t W = get_arg(args::W);
    constexpr uint32_t tile_width = get_arg(args::tile_width);
    static_assert(get_arg(args::do_gamma) == 1 && get_arg(args::do_beta) == 0);
    static_assert(get_arg(args::fp32_dest_acc_en) == 1);
    static_assert(tile_width == 32 && W % tile_width == 0 && W == Wt * tile_width);
    static_assert(block_size >= 1 && block_size <= 4 && Wt % block_size == 0);

    constexpr std::uint32_t inv_w_bits = __builtin_bit_cast(std::uint32_t, 1.0f / static_cast<float>(W));
    static_assert((inv_w_bits & 0xffff) == 0);
    constexpr std::uint32_t inv_w = inv_w_bits >> 16;

    constexpr uint32_t cb_scaler = static_cast<uint32_t>(dfb::scaler);
    constexpr uint32_t cb_eps = static_cast<uint32_t>(dfb::eps);
    constexpr uint32_t cb_in = static_cast<uint32_t>(dfb::in);
    constexpr uint32_t cb_inb = static_cast<uint32_t>(dfb::inb);
    constexpr uint32_t cb_out = static_cast<uint32_t>(dfb::out);
    constexpr uint32_t cb_gamma = static_cast<uint32_t>(dfb::gamma);
    constexpr uint32_t cb_xmm = static_cast<uint32_t>(dfb::xmm);
    constexpr uint32_t cb_ex2 = static_cast<uint32_t>(dfb::ex2);
    constexpr uint32_t cb_xmm2 = static_cast<uint32_t>(dfb::xmm2);
    constexpr uint32_t cb_ex2pe = static_cast<uint32_t>(dfb::ex2pe);
    constexpr uint32_t cb_fusion = static_cast<uint32_t>(dfb::fusion);
    constexpr uint32_t cb_hout = static_cast<uint32_t>(dfb::hout);

    DataflowBuffer dfb_eps(dfb::eps);
    DataflowBuffer dfb_in(dfb::in);
    DataflowBuffer dfb_xmm(dfb::xmm);
    DataflowBuffer dfb_out(dfb::out);
    DataflowBuffer dfb_ex2pe(dfb::ex2pe);
    DataflowBuffer dfb_scaler(dfb::scaler);
    DataflowBuffer dfb_xmm2(dfb::xmm2);
    DataflowBuffer dfb_ex2(dfb::ex2);
    DataflowBuffer dfb_inb(dfb::inb);
    DataflowBuffer dfb_gamma(dfb::gamma);
    DataflowBuffer dfb_fusion(dfb::fusion);
    DataflowBuffer dfb_hout(dfb::hout);

    compute_kernel_hw_startup(dfb::in, dfb::inb, dfb::xmm);

    dfb_eps.wait_front(1);

    for (uint32_t ncht = 0; ncht < NCHt; ncht++) {
        set_formats(cb_in, cb_inb);
        set_pack_format(cb_xmm);
        add_setup(cb_in, cb_inb);
        for (uint32_t b = 0; b < Wt; b += block_size) {
            dfb_in.wait_front(block_size);
            dfb_inb.wait_front(block_size);
            dfb_xmm.reserve_back(block_size);
            dfb_hout.reserve_back(block_size);
            dest_acquire();
            for (uint32_t i = 0; i < block_size; ++i) {
                add_pair(cb_in, cb_inb, i, i, i);
            }
            dest_commit();
            dest_wait();
            for (uint32_t i = 0; i < block_size; ++i) {
                pack_dest(i, cb_xmm);
            }
            for (uint32_t i = 0; i < block_size; ++i) {
                pack_dest(i, cb_hout);
            }
            dest_release();
            dfb_xmm.push_back(block_size);
            dfb_hout.push_back(block_size);
            dfb_in.pop_front(block_size);
            dfb_inb.pop_front(block_size);
        }
        set_pack_format(cb_xmm2);
        set_formats(cb_in, cb_xmm, cb_inb, cb_xmm);

        mul_setup(cb_xmm, cb_xmm);
        for (uint32_t b = 0; b < Wt; b += block_size) {
            dest_acquire();
            dfb_xmm.wait_front(b + block_size);
#pragma GCC unroll 0
            for (uint32_t j = 0; j < block_size; ++j) {
                mul_pair(cb_xmm, cb_xmm, b + j, b + j, j);
            }
            dest_commit();
            dest_wait();
            dfb_xmm2.reserve_back(block_size);
#pragma GCC unroll 0
            for (uint32_t j = 0; j < block_size; ++j) {
                pack_dest(j, cb_xmm2);
            }
            dfb_xmm2.push_back(block_size);
            dest_release();
        }
        set_formats(cb_xmm, cb_xmm2, cb_xmm, cb_scaler);

        dfb_scaler.wait_front(1);
        set_formats(cb_xmm2, cb_scaler);
        dest_acquire();
        set_formats(cb_scaler, cb_xmm2);
        rowsum_setup(cb_xmm2, cb_scaler, cb_ex2);
        for (uint32_t b = 0; b < Wt; b += block_size) {
            dfb_xmm2.wait_front(block_size);
            for (uint32_t j = 0; j < block_size; ++j) {
                rowsum_pair(cb_xmm2, cb_scaler, j, 0, 0);
            }
            dfb_xmm2.pop_front(block_size);
        }
        rowsum_done();
        set_formats(cb_xmm2, cb_scaler);
        scale_setup();
        scale_dest<inv_w>(0);
        dest_commit();
        dest_wait();
        dfb_ex2.reserve_back(1);
        set_pack_format(cb_ex2);
        pack_dest(0, cb_ex2);
        dest_release();
        dfb_ex2.push_back(1);
        dfb_ex2.wait_front(1);

        set_pack_format(cb_ex2pe);
        set_formats(cb_ex2, cb_eps);
        add_setup(cb_ex2, cb_eps);
        rsqrt_setup();
        dest_acquire();
        dfb_ex2.wait_front(1);
        add_pair(cb_ex2, cb_eps, 0, 0, 0);
        dfb_ex2.pop_front(1);
        rsqrt_dest(0);
        dest_commit();
        dest_wait();
        dfb_ex2pe.reserve_back(1);
        pack_dest(0, cb_ex2pe);
        dfb_ex2pe.push_back(1);
        dest_release();

        constexpr uint32_t norm_tiles = block_size * 2;
        static_assert(Wt % norm_tiles == 0);
        for (uint32_t start = 0; start < Wt; start += norm_tiles) {
            set_pack_format(cb_fusion);
            set_formats(cb_xmm, cb_ex2pe);
            mul_cols_setup(cb_xmm, cb_ex2pe);
            dfb_xmm.wait_front(norm_tiles + start);
            dfb_ex2pe.wait_front(1);
            for (uint32_t b = 0; b < norm_tiles; b += block_size) {
                dest_acquire();
#pragma GCC unroll 0
                for (uint32_t j = 0; j < block_size; ++j) {
                    mul_cols_pair(cb_xmm, cb_ex2pe, start + b + j, 0, j);
                }
                dest_commit();
                dest_wait();
                dfb_fusion.reserve_back(block_size);
#pragma GCC unroll 0
                for (uint32_t j = 0; j < block_size; ++j) {
                    pack_dest(j, cb_fusion);
                }
                dfb_fusion.push_back(block_size);
                dest_release();
            }

            set_format_a(cb_xmm, cb_fusion);
            set_pack_format(cb_out);
            set_format_b(cb_ex2pe, cb_gamma);
            mul_rows_setup(cb_fusion, cb_gamma);
            dfb_gamma.wait_front(norm_tiles + start);
            for (uint32_t b = 0; b < norm_tiles; b += block_size) {
                dest_acquire();
                dfb_fusion.wait_front(block_size);
#pragma GCC unroll 0
                for (uint32_t j = 0; j < block_size; ++j) {
                    mul_rows_pair(cb_fusion, cb_gamma, j, start + b + j, j);
                }
                dfb_fusion.pop_front(block_size);
                dest_commit();
                dest_wait();
                dfb_out.reserve_back(block_size);
#pragma GCC unroll 0
                for (uint32_t j = 0; j < block_size; ++j) {
                    pack_dest(j, cb_out);
                }
                dfb_out.push_back(block_size);
                dest_release();
            }
        }
        dfb_ex2pe.pop_front(1);
        dfb_xmm.pop_front(static_cast<uint16_t>(Wt));
    }
    dfb_scaler.pop_front(1);
    dfb_eps.pop_front(1);
    if (NCHt > 0) {
        dfb_gamma.pop_front(Wt);
    }
}
