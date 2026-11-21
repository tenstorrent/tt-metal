// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

static_assert(DST_SYNC_MODE == DstSync::SyncHalf);

#ifdef TRISC_UNPACK
ALWI void sdo_hw_unpack(const std::uint32_t a_cb, const std::uint32_t b_cb) {
    const std::uint32_t ia = get_operand_id(a_cb);
    const std::uint32_t ib = get_operand_id(b_cb);
    const std::uint32_t sa = unpack_src_format[ia];
    const std::uint32_t sb = unpack_src_format[ib];
    const std::uint32_t da = masked_data_format(unpack_dst_format[ia]);
    const std::uint32_t db = masked_data_format(unpack_dst_format[ib]);
    const std::uint32_t ra = get_operand_face_r_dim(ia);
    const std::uint32_t rb = get_operand_face_r_dim(ib);
    volatile std::uint32_t tt_reg_ptr* cfg = get_cfg_pointer();

    while (semaphore_read(semaphore::UNPACK_SYNC) > 0) {
    }
    TTI_SETADCXY(p_setadc::UNP_A | p_setadc::UNP_B, 0, 0, 0, 0, 0b1011);
    TTI_SETADCZW(p_setadc::UNP_A | p_setadc::UNP_B, 0, 0, 0, 0, 0b1111);
    cfg[UNP0_ADDR_CTRL_ZW_REG_1_Zstride_ADDR32] = (FACE_C_DIM * FACE_R_DIM * datum_size_in_bytes(da))
                                                  << UNP0_ADDR_CTRL_ZW_REG_1_Zstride_SHAMT;
    cfg[UNP1_ADDR_CTRL_ZW_REG_1_Zstride_ADDR32] = (FACE_C_DIM * FACE_R_DIM * datum_size_in_bytes(db))
                                                  << UNP1_ADDR_CTRL_ZW_REG_1_Zstride_SHAMT;
    cfg_reg_rmw_tensix<
        UNP0_ADDR_CTRL_XY_REG_1_Ystride_ADDR32,
        UNP0_ADDR_CTRL_XY_REG_0_Ystride_SHAMT,
        UNP0_ADDR_CTRL_XY_REG_1_Ystride_MASK>(canonical_unpA_y_stride(da));

    t6_mutex_acquire(mutex::REG_RMW);
    alu_config_u alu = {.val = 0};
    alu.f.ALU_FORMAT_SPEC_REG0_SrcAUnsigned = sa == to_underlying(DataFormat::UInt8) ? 1 : 0;
    alu.f.ALU_FORMAT_SPEC_REG0_SrcBUnsigned = sb == to_underlying(DataFormat::UInt8) ? 1 : 0;
    cfg_reg_rmw_tensix<
        ALU_FORMAT_SPEC_REG0_SrcA_ADDR32,
        0,
        ALU_FORMAT_SPEC_REG0_SrcAUnsigned_MASK | ALU_FORMAT_SPEC_REG0_SrcBUnsigned_MASK |
            ALU_ROUNDING_MODE_Fpu_srnd_en_MASK | ALU_ROUNDING_MODE_Gasket_srnd_en_MASK |
            ALU_ROUNDING_MODE_Packer_srnd_en_MASK>(alu.val);
    cfg_reg_rmw_tensix<THCON_SEC0_REG1_Unp_LF8_4b_exp_RMW>((sa & 0x1F) == to_underlying(DataFormat::Fp8_e4m3) ? 1 : 0);
    cfg_reg_rmw_tensix<THCON_SEC1_REG1_Unp_LF8_4b_exp_RMW>((sb & 0x1F) == to_underlying(DataFormat::Fp8_e4m3) ? 1 : 0);
    t6_mutex_release(mutex::REG_RMW);

    unpack_tile_descriptor_u td;
    for (std::uint32_t i = 0; i < TILE_DESC_SIZE; i++) {
        td.val[i] = 0;
    }
    td.f.uncompressed = 1;
    td.f.y_dim = 1;
    td.f.in_data_format = masked_data_format(sa);
    td.f.x_dim = 0;
    td.f.z_dim = get_operand_num_faces(ia);
    for (std::uint32_t i = 0; i < TILE_DESC_SIZE; i++) {
        cfg[THCON_SEC0_REG0_TileDescriptor_ADDR32 + i] = td.val[i];
    }
    td.f.in_data_format = masked_data_format(sb);
    td.f.x_dim = rb * FACE_C_DIM;
    td.f.z_dim = get_operand_num_faces(ib);
    for (std::uint32_t i = 0; i < TILE_DESC_SIZE; i++) {
        cfg[THCON_SEC1_REG0_TileDescriptor_ADDR32 + i] = td.val[i];
    }

    unpack_config_u uc;
    for (std::uint32_t i = 0; i < CONFIG_SIZE; i++) {
        uc.val[i] = 0;
    }
    uc.f.throttle_mode = 2;
    uc.f.uncompress_cntx0_3 = 0xf;
    uc.f.uncompress_cntx4_7 = 0xf;
    uc.f.out_data_format = da;
    for (std::uint32_t i = 0; i < CONFIG_SIZE; i++) {
        cfg[THCON_SEC0_REG2_Out_data_format_ADDR32 + i] = uc.val[i];
    }
    uc.f.out_data_format = db;
    for (std::uint32_t i = 0; i < CONFIG_SIZE; i++) {
        cfg[THCON_SEC1_REG2_Out_data_format_ADDR32 + i] = uc.val[i];
    }

    const std::uint32_t dest_row0 = ra == 0 ? 22 * 16 : 4 * 16;
    cfg[THCON_SEC0_REG5_Dest_cntx0_address_ADDR32] = dest_row0 | (dest_row0 << 16);
    cfg[THCON_SEC0_REG5_Tile_x_dim_cntx0_ADDR32] = (ra * FACE_C_DIM) | ((ra * FACE_C_DIM) << 16);
    regfile[p_gpr_unpack::FACE_DIM_16x16] = 256 | (256 << 16);
    regfile[p_gpr_unpack::FACE_DIM_8x16] = 128 | (128 << 16);
    regfile[p_gpr_unpack::FACE_DIM_4x16] = 64 | (64 << 16);
    regfile[p_gpr_unpack::FACE_DIM_2x16] = 32 | (32 << 16);
    regfile[p_gpr_unpack::FACE_DIM_1x16] = 16 | (16 << 16);
    sync_regfile_write(p_gpr_unpack::FACE_DIM_1x16);
    TTI_SETC16(SRCA_SET_Base_ADDR32, 0x4);
    cfg[UNP0_ADD_DEST_ADDR_CNTR_add_dest_addr_cntr_ADDR32] = 0x1 << UNP0_ADD_DEST_ADDR_CNTR_add_dest_addr_cntr_SHAMT;
    unp_cfg_context = 0;
    TTI_SETC16(UNPACK_MISC_CFG_CfgContextOffset_0_ADDR32, 0x0000);
    TT_SETDMAREG(0, LOWER_HALFWORD(get_local_cb_interface(ia).fifo_page_size), 0, LO_16(p_gpr_unpack::TILE_SIZE_A));
    TT_SETDMAREG(0, LOWER_HALFWORD(get_local_cb_interface(ib).fifo_page_size), 0, LO_16(p_gpr_unpack::TILE_SIZE_B));
}
#endif

#ifdef TRISC_MATH
ALWI void sdo_hw_math(const std::uint32_t a_cb, const std::uint32_t b_cb) {
    const std::uint32_t fa = unpack_dst_format[get_operand_id(a_cb)];
    const std::uint32_t fb = unpack_dst_format[get_operand_id(b_cb)];
    tensix_sync();
    while (semaphore_read(semaphore::MATH_PACK) > 0) {
    }
    TTI_SEMINIT(2, 0, p_stall::SEMAPHORE_1);
    dest_offset_id = 0;
    TTI_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, 0);

    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::MATH | p_stall::WAIT_SFPU);
    cfg_reg_rmw_tensix<DEST_ACCESS_CFG_zeroacc_absolute_tile_mode_RMW>(0);
    cfg_reg_rmw_tensix<ALU_ACC_CTRL_INT8_math_enabled_RMW>(
        (is_int8_or_int32_format(fa) || is_int8_or_int32_format(fb)) ? 1 : 0);
    cfg_reg_rmw_tensix<ALU_ACC_CTRL_Fp32_enabled_RMW>(DST_ACCUM_MODE ? 1 : 0);
    cfg_reg_rmw_tensix<ALU_ACC_CTRL_SFPU_Fp32_enabled_RMW>(DST_ACCUM_MODE ? 1 : 0);
    cfg_reg_rmw_tensix<
        ALU_FORMAT_SPEC_REG_SrcA_val_ADDR32,
        0,
        ALU_FORMAT_SPEC_REG_SrcA_val_MASK | ALU_FORMAT_SPEC_REG_SrcA_override_MASK | ALU_FORMAT_SPEC_REG_SrcB_val_MASK |
            ALU_FORMAT_SPEC_REG_SrcB_override_MASK>(0);
    math::src_zero_flag_srca_fmt = fa;
    math::src_zero_flag_srcb_fmt = fb;
    math::_configure_default_zero_flag_state_();
}
#endif

#ifdef TRISC_PACK
ALWI void sdo_hw_pack(const std::uint32_t out_cb) {
    const std::uint32_t id = get_output_id(out_cb);
    const std::uint32_t sf = pack_src_format[id];
    const std::uint32_t df = pack_dst_format[id];
    const std::uint32_t s = masked_data_format(sf);
    const std::uint32_t d = masked_data_format(df);
    const std::uint32_t fr = get_output_face_r_dim(id);
    const std::uint32_t faces = get_output_num_faces(id);
    const std::uint32_t sec = get_output_partial_face(id) ? 1 : faces;
    const bool e4m3 = (df & 0x1F) == to_underlying(DataFormat::Fp8_e4m3);
    volatile std::uint32_t tt_reg_ptr* cfg = get_cfg_pointer();
    const auto strides = [&]() {
        const std::uint32_t ys = FACE_C_DIM * datum_size_in_bytes(sf);
        const std::uint32_t zs = FACE_R_DIM * ys;
        const std::uint32_t ws = TILE_NUM_FACES * FACE_R_DIM * ys;
        TT_SETDMAREG(0, LOWER_HALFWORD(ys << PCK0_ADDR_CTRL_XY_REG_0_Ystride_SHAMT), 0, LO_16(p_gpr_pack::TMP0));
        TT_SETDMAREG(0, UPPER_HALFWORD(ys << PCK0_ADDR_CTRL_XY_REG_0_Ystride_SHAMT), 0, HI_16(p_gpr_pack::TMP0));
        TT_SETDMAREG(0, LOWER_HALFWORD(zs << PCK0_ADDR_CTRL_ZW_REG_0_Zstride_SHAMT), 0, LO_16(p_gpr_pack::TMP1));
        TT_SETDMAREG(0, UPPER_HALFWORD(ws << PCK0_ADDR_CTRL_ZW_REG_0_Wstride_SHAMT), 0, HI_16(p_gpr_pack::TMP1));
        TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::THCON);
        TTI_WRCFG(p_gpr_pack::TMP0, p_cfg::WRCFG_32b, PCK0_ADDR_CTRL_XY_REG_0_Xstride_ADDR32);
        TTI_WRCFG(p_gpr_pack::TMP1, p_cfg::WRCFG_32b, PCK0_ADDR_CTRL_ZW_REG_0_Zstride_ADDR32);
        TTI_WRCFG(p_gpr::ZERO, p_cfg::WRCFG_32b, PCK0_ADDR_CTRL_XY_REG_1_Xstride_ADDR32);
        TTI_WRCFG(p_gpr::ZERO, p_cfg::WRCFG_32b, PCK0_ADDR_CTRL_ZW_REG_1_Zstride_ADDR32);
        TTI_NOP;
        TTI_NOP;
    };

    TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::PACK);
    tensix_sync();
    strides();
    t6_mutex_acquire(mutex::REG_RMW);
    cfg_reg_rmw_tensix<THCON_SEC0_REG1_Pac_LF8_4b_exp_RMW>(e4m3 ? 1 : 0);
    cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG2_Dstacc_RMW>(e4m3 ? to_underlying(DataFormat::Float16) : s);
    cfg_reg_rmw_tensix<
        ALU_FORMAT_SPEC_REG_Dstacc_val_ADDR32,
        0,
        ALU_FORMAT_SPEC_REG_Dstacc_val_MASK | ALU_FORMAT_SPEC_REG_Dstacc_override_MASK>(0);
    cfg_reg_rmw_tensix<STACC_RELU_ApplyRelu_ADDR32, 0, STACC_RELU_ApplyRelu_MASK | STACC_RELU_ReluThreshold_MASK>(0);
    t6_mutex_release(mutex::REG_RMW);

    pack_config_u pc;
    pc.val[0] = 0;
    pc.val[1] = 0;
    pc.val[2] = 0;
    pc.f.exp_section_size = (d == to_underlying(DataFormat::Lf8) || d == to_underlying(DataFormat::Int8)) ? 0 : sec;
    pc.f.uncompress = 1;
    pc.f.out_data_format = d;
    pc.f.in_data_format = e4m3 ? to_underlying(DataFormat::Float16) : s;
    const bool thr = DST_ACCUM_MODE && IS_BFP_A_FORMAT(d);
    cfg_reg_rmw_tensix<
        THCON_SEC0_REG1_Exp_threshold_ADDR32,
        0,
        THCON_SEC0_REG1_Exp_threshold_en_MASK | THCON_SEC0_REG1_Exp_threshold_MASK>(
        ((thr ? 113u : 0u) << THCON_SEC0_REG1_Exp_threshold_SHAMT) |
        ((thr ? 1u : 0u) << THCON_SEC0_REG1_Exp_threshold_en_SHAMT));
    cfg[THCON_SEC0_REG1_Row_start_section_size_ADDR32 + 0] = pc.val[0];
    cfg[THCON_SEC0_REG1_Row_start_section_size_ADDR32 + 2] = pc.val[2];
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::PACK);
    cfg_reg_rmw_tensix<THCON_SEC0_REG1_Pack_L1_Acc_RMW>(0);
#if !defined(TT_METAL_TTSIM)
    cfg_reg_rmw_tensix<PACK_GLOBAL_CFG_CTL_pack_disable_fast_tile_end_drain_RMW>(0);
#endif

    const bool wide = sf == to_underlying(DataFormat::Int32) || sf == to_underlying(DataFormat::UInt32) ||
                      sf == to_underlying(DataFormat::Float32);
    const bool narrow = sf == to_underlying(DataFormat::Int8) || sf == to_underlying(DataFormat::UInt8);
    dest_rd_ctrl_u rd;
    rd.val = 0;
    rd.f.PCK_DEST_RD_CTRL_Read_32b_data = wide || DST_ACCUM_MODE;
    rd.f.PCK_DEST_RD_CTRL_Read_int8 = !(DST_ACCUM_MODE || wide) && narrow;
    rd.f.PCK_DEST_RD_CTRL_Read_unsigned = df == to_underlying(DataFormat::UInt8) ? 1 : 0;
    rd.f.PCK_DEST_RD_CTRL_Round_10b_mant =
        ((DST_ACCUM_MODE && sf == to_underlying(DataFormat::Float16)) || (!IS_A_FORMAT(sf) && e4m3)) ? 1 : 0;
    cfg[PCK_DEST_RD_CTRL_Read_32b_data_ADDR32] = rd.val;
    regfile[p_gpr_pack::EXP0_SEC_SIZE_BFP] = sec << THCON_SEC0_REG8_Exp_section_size_SHAMT;
    sync_regfile_write(p_gpr_pack::EXP0_SEC_SIZE_BFP);

    pack_counters_u cnt;
    cnt.val = 0;
    cnt.f.pack_reads_per_xy_plane = 1;
    for (std::uint32_t i = 0; i < NUM_PACKERS; i++) {
        cfg[PACK_COUNTERS_SEC0_pack_per_xy_plane_ADDR32 + i] = cnt.val;
    }
    pck_edge_offset_u edge;
    edge.val = 0;
    edge.f.mask = 0xffff;
    cfg[PCK_EDGE_OFFSET_SEC0_mask_ADDR32] = edge.val;
    cfg[TILE_ROW_SET_MAPPING_0_row_set_mapping_0_ADDR32] = 0x0;
    cfg[PCK_EDGE_TILE_FACE_SET_SELECT_select_ADDR32] = 0x0;
    regfile[p_gpr_pack::TILE_HEADER] = get_local_cb_interface(id).fifo_page_size;
    regfile[p_gpr_pack::TILE_HEADER + 1] = 0;
    regfile[p_gpr_pack::TILE_HEADER + 2] = 0;
    regfile[p_gpr_pack::TILE_HEADER + 3] = 0;
    sync_regfile_write(p_gpr_pack::TILE_HEADER + 3);

    addr_mod_pack_t{.y_src = {.incr = 4}}.set(ADDR_MOD_0);
    addr_mod_pack_t{.y_src = {.incr = 0, .clr = 1, .cr = 0}, .z_src = {.incr = 0, .clr = 1}}.set(ADDR_MOD_1);
    addr_mod_pack_t{.y_src = {.incr = 0, .clr = 1, .cr = 0}, .z_src = {.incr = 1, .clr = 0}}.set(ADDR_MOD_2);
    const std::uint32_t intf =
        fr == 1 ? p_pacr::SINGLE_INTF_ACTIVE : (fr == 2 ? p_pacr::TWO_INTFS_ACTIVE : p_pacr::ALL_INTF_ACTIVE);
    ckernel::ckernel_template mop(
        faces,
        fr < 4 ? 1 : fr >> 2,
        TT_OP_PACR(
            p_pacr::CFG_CTXT_0,
            p_pacr::NO_ROW_PAD_ZERO,
            p_pacr::DST_ACCESS_NORMAL_MODE,
            ADDR_MOD_0,
            p_pacr::ADDR_CNT_CTXT_0,
            p_pacr::P_ZERO_OUTPUT_DISABLED,
            intf,
            0,
            0,
            0,
            0,
            0));
    mop.set_last_inner_loop_instr(TT_OP_PACR(
        p_pacr::CFG_CTXT_0,
        p_pacr::NO_ROW_PAD_ZERO,
        p_pacr::DST_ACCESS_NORMAL_MODE,
        ADDR_MOD_2,
        p_pacr::ADDR_CNT_CTXT_0,
        p_pacr::P_ZERO_OUTPUT_DISABLED,
        intf,
        0,
        0,
        0,
        0,
        0));
    mop.set_last_outer_loop_instr(TT_OP_PACR(
        p_pacr::CFG_CTXT_0,
        p_pacr::NO_ROW_PAD_ZERO,
        p_pacr::DST_ACCESS_NORMAL_MODE,
        ADDR_MOD_1,
        p_pacr::ADDR_CNT_CTXT_0,
        p_pacr::P_ZERO_OUTPUT_DISABLED,
        intf,
        0,
        0,
        0,
        0,
        1));
    mop.program();
    strides();
    TTI_SETADCXX(p_setadc::PAC, FACE_C_DIM - 1, 0x0);

    dest_offset_id = 0;
    TTI_STALLWAIT(p_stall::STALL_TDMA | p_stall::STALL_THCON, p_stall::PACK);
    TTI_SETDMAREG(0, 0x00, 0, LO_16(p_gpr_pack::DEST_OFFSET_LO));
    TTI_SETDMAREG(0, DEST_REGISTER_HALF_SIZE, 0, LO_16(p_gpr_pack::DEST_OFFSET_HI));
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::THCON);
    TTI_WRCFG(p_gpr_pack::DEST_OFFSET_LO, p_cfg::WRCFG_32b, DEST_TARGET_REG_CFG_PACK_SEC0_Offset_ADDR32);
    TTI_DMANOP;
    TTI_DMANOP;
    TTI_SETADCXY(0b100, 0, 0, 0, 0, 0b1011);
    TTI_SETADCZW(0b100, 0, 0, 0, 0, 0b1111);
    pack_sync_tile_dst_ptr = 0;
}
#endif

ALWI void sdo_hw_start(const std::uint32_t cb_a, const std::uint32_t cb_b, const std::uint32_t cb_out) {
    UNPACK((sdo_hw_unpack(cb_b, cb_a)));
    MATH((sdo_hw_math(cb_b, cb_a)));
    PACK((sdo_hw_pack(cb_out)));
}
