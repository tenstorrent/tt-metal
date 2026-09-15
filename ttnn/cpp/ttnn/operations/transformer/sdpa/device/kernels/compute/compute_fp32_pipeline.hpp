// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once

// Investigation-only fixed-half ownership. The caller drains the ordinary
// half-sync queue first. MATH uses lower DST; UNPACK/PACK own scores in upper
// DST, with a resident BF16-derived, broadcast FP32 maximum at logical tile 6.
ALWI void sdpa_pipeline_reload(uint32_t score_cb, uint32_t index) {
    UNPACK((t6_semaphore_wait_on_max < p_stall::STALL_UNPACK | p_stall::STALL_SYNC > (semaphore::UNPACK_TO_DEST)));
    UNPACK((tensix_sync()));
    UNPACK((llk_unpack_reconfig_data_format<true, p_dim_stride_target::IGNORE, true>(score_cb, score_cb)));
    UNPACK((
        llk_unpack_A_init<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, true>(false, false, score_cb)));
    sdpa_score_unpack_mop(8);
    UNPACK({
        TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111);
        wait_for_next_context(2);
        auto& cb = get_local_cb_interface(score_cb);
        volatile uint32_t tt_reg_ptr* cfg = get_cfg_pointer();
        const uint32_t base =
            unp_cfg_context == 0 ? THCON_SEC0_REG3_Base_address_ADDR32 : THCON_SEC0_REG3_Base_cntx1_address_ADDR32;
        cfg[base] = cb.fifo_rd_ptr - 1 + index * cb.fifo_page_size;
        semaphore_post(semaphore::UNPACK_SYNC);
        TTI_SETC16(SRCA_SET_Base_ADDR32, 0);
        const uint32_t dst_byte_addr = 16 * (4 + 4 * 64);
        if (unp_cfg_context == 0) {
            cfg_reg_rmw_tensix<THCON_SEC0_REG2_Unpack_if_sel_cntx0_RMW>(1);
            cfg_reg_rmw_tensix<THCON_SEC0_REG5_Dest_cntx0_address_RMW>(dst_byte_addr);
        } else {
            cfg_reg_rmw_tensix<THCON_SEC0_REG2_Unpack_if_sel_cntx1_RMW>(1);
            cfg_reg_rmw_tensix<THCON_SEC0_REG5_Dest_cntx1_address_RMW>(dst_byte_addr);
        }
        TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG);
        ckernel_template::run();
        t6_semaphore_get(semaphore::UNPACK_SYNC);
        unpack_to_dest_tile_done(unp_cfg_context, static_cast<uint32_t>(DataFormat::Float32));
        switch_config_context(unp_cfg_context);
    });
}

template <uint32_t scale_fp32>
ALWI void sdpa_pipeline_exp(uint32_t inout_cb, uint32_t index) {
    PACK(
        (t6_semaphore_wait_on_zero < p_stall::STALL_SYNC | p_stall::STALL_SFPU |
         p_stall::STALL_MATH > (semaphore::UNPACK_TO_DEST)));
    // ZEROACC uses a bank-local face index; SFPU uses the full-DST logical index.
    PACK(TTI_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, 512));
    PACK(for (uint32_t tile = 0; tile < 2; ++tile) {
        for (uint32_t face = 0; face < 4; ++face) {
            TT_ZEROACC(p_zeroacc::CLR_16, 1, 1, ADDR_MOD_3, get_dest_index_in_faces(tile, face));
        }
    });
    PACK((SFPU_UNARY_CALL(
        DstSync::SyncFull, true, calculate_sdpa_fused_sub_exp, (scale_fp32, false, true), 4, VectorMode::None)));
    PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
    configure_pack_width(inout_cb, 2);
    PACK((llk_pack_relu_config(ReluConfig::zero())));
    PACK({
        // This region explicitly owns full-DST addresses, independent of the
        // surrounding kernel's ordinary half-sync pack API capacity check.
        const uint32_t address = get_output_tile_address<true, PackMode::Default>(get_output_id(inout_cb), index);
        _llk_pack_<DstSync::SyncFull, true, PackMode::Default>(4, address);
    });
    PACK((t6_semaphore_get<p_stall::PACK>(semaphore::UNPACK_TO_DEST)));
}

template <uint32_t scale_fp32, uint32_t stride>
static void sdpa_pipeline_qk_row(
    uint32_t q_cb,
    uint32_t k_cb,
    uint32_t score_cb,
    uint32_t inout_cb,
    uint32_t max_cb,
    uint32_t q_index,
    uint32_t row,
    uint32_t cols,
    bool overlap_first_half) {
    // Drain existing work, then establish one lower-half matmul slot. PACK
    // release below clears only that half; the maximum remains resident above.
    MATH((_llk_math_pack_sync_init_<DstSync::SyncFull, true>()));
    PACK((_llk_pack_dest_init_<DstSync::SyncFull, true>()));
    tile_regs_acquire();
    MATH(dest_offset_id = 1;);
    sdpa_stream_reconfig(max_cb, max_cb);
    unary_bcast_init<BroadcastType::COL>(max_cb);
    unary_bcast<BroadcastType::COL>(max_cb, row - 1, 2);
    unary_bcast_uninit<BroadcastType::COL>(max_cb);
    MATH(dest_offset_id = 0;);
    MATH((_llk_math_dest_section_done_<DstSync::SyncFull, true>()));
    tile_regs_wait();
    PACK((_llk_packer_set_math_semaphore_<p_stall::MATH>()));
    UNPACK(get_local_cb_interface(score_cb).fifo_rd_ptr = get_local_cb_interface(inout_cb).fifo_rd_ptr;);
    MATH((llk_math_matmul_init_no_mop<MATH_FIDELITY, MM_THROTTLE>(q_cb, k_cb, true, 4, 1)));
    for (uint32_t col = 0; col < cols; col += 4) {
        const uint32_t prev_index = (row - 1) * stride + col;
        sdpa_pipeline_reload(score_cb, prev_index);
        sdpa_pipeline_exp<scale_fp32>(inout_cb, prev_index);
        sdpa_stream_reconfig(k_cb, q_cb);
        UNPACK((llk_unpack_AB_matmul_init(q_cb, k_cb, true, 4, 1, 4)));
        tile_regs_acquire();
        MATH(TTI_ZEROACC(p_zeroacc::CLR_HALF, 1, 0, ADDR_MOD_1, 0));
        for (uint32_t k = 0; k < 4; ++k) {
            matmul_block_no_mop(q_cb, k_cb, q_index + k, col + k * stride, 0, true, 4, 1, 4);
        }
        MATH((_llk_math_dest_section_done_<DstSync::SyncFull, true>()));
        sdpa_pipeline_reload(score_cb, prev_index + 2);
        tile_regs_wait();
        configure_pack_width(inout_cb, 4);
        PACK((llk_pack_relu_config(ReluConfig::none())));
        pack_tile<true>(0, inout_cb, row * stride + col);
        PACK(TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::PACK));
        PACK(TTI_ZEROACC(p_zeroacc::CLR_HALF, 1, 0, ADDR_MOD_1, 0));
        PACK((_llk_packer_set_math_semaphore_<p_stall::NONE>()));
        sdpa_pipeline_exp<scale_fp32>(inout_cb, prev_index + 2);
        if (overlap_first_half && col + 4 == cols / 2) {
            PACK((t6_semaphore_post<p_stall::STALL_PACK>(semaphore::UNPACK_MATH_DONE)));
        }
    }
    MATH((_llk_math_pack_sync_init_<DST_SYNC_MODE, true>()));
    PACK((_llk_pack_dest_init_<DST_SYNC_MODE, true>()));
}
