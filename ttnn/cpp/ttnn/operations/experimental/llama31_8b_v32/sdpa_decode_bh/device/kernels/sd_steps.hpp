// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#ifdef SDB_C0
template <bool MUL, uint32_t P>
ALWI void sdk_rows4_sfpu(uint32_t idst, uint32_t row0) {
#ifdef TRISC_MATH
    sdo_sfpu_start(idst);
    sdv_seek_rows4(row0);
    if constexpr (MUL) {
        for (int face = 0; face < 2; face++) {
            sdv_mul_rows<P, SDB_N / 2>();
            sdo_sfpu_next_face();
        }
    } else {
        sdv_exp<false, SDB_N / 2>(2, 1, 0, P);
    }
    sdo_sfpu_done();
#endif
}

template <uint32_t in0_cb, uint32_t scale_fp32>
void sdk_sub_exp_rows4(uint32_t in1_cb, uint32_t cols, uint32_t row0) {
    CircularBuffer cb_in0(in0_cb);
    CircularBuffer cb_in1(in1_cb);
    sdo_reconfig(in0_cb, in1_cb);
    sdh_binop_init<EltwiseBinaryType::ELWSUB, BroadcastType::COL>();
    PACK((sdo_pack_relu(true)));
    cb_in0.wait_front(cols);
    cb_in1.wait_front(1);
    sdo_regs_acquire();
    for (uint32_t j = 0; j < cols; ++j) {
        sd_sub_tiles_bcast_cols(in0_cb, in1_cb, j, 0, j);
        sd_binop_with_scalar_tile_init();
        sdk_rows4_sfpu<true, scale_fp32>(j, row0);
        sd_exp_tile_init<false, 0x3F800000, true>();
        sdk_rows4_sfpu<false, p_sfpu::kCONST_1_FP16B>(j, row0);
    }
    sdo_regs_commit();
    cb_in0.pop_front(cols);
    cb_in0.reserve_back(cols);
    sdo_regs_wait();
    sdo_pack_reconfig(in0_cb);
    for (uint32_t j = 0; j < cols; ++j) {
        sd_pack_tile(j, in0_cb);
    }
    cb_in0.push_back(cols);
    sdo_regs_release();
    PACK((sdo_pack_relu(false)));
}
#endif

#ifdef SDB_C1
#ifdef TRISC_MATH

template <uint32_t S>
ALWI void sdk_fused_rows4() {
    sdv_corr_body<S>();
}

ALWI void sdk_recip_rows4() {
    static_assert(APPROX);
    sdv_recip_one();
}
#endif

template <uint32_t scale_fp32>
void sdk_correction_block_rows4(
    uint32_t cb_worker_max,
    uint32_t cb_worker_sum,
    uint32_t cb_cur_max,
    uint32_t cb_prev_max,
    uint32_t cb_cur_sum,
    uint32_t cb_prev_sum,
    uint32_t cb_exp_max_diff,
    uint32_t cb_exp_max_diff_2,
    uint32_t num_head_tiles,
    uint32_t row0) {
    CircularBuffer cb_worker_max_obj(cb_worker_max);
    CircularBuffer cb_worker_sum_obj(cb_worker_sum);
    CircularBuffer cb_cur_max_obj(cb_cur_max);
    CircularBuffer cb_prev_max_obj(cb_prev_max);
    CircularBuffer cb_cur_sum_obj(cb_cur_sum);
    CircularBuffer cb_prev_sum_obj(cb_prev_sum);
    CircularBuffer cb_exp_max_diff_obj(cb_exp_max_diff);
    CircularBuffer cb_exp_max_diff_2_obj(cb_exp_max_diff_2);
    cb_worker_max_obj.wait_front(num_head_tiles);
    cb_worker_sum_obj.wait_front(num_head_tiles);
    cb_prev_max_obj.wait_front(num_head_tiles);
    cb_prev_sum_obj.wait_front(num_head_tiles);
    cb_cur_max_obj.reserve_back(num_head_tiles);
    cb_cur_sum_obj.reserve_back(num_head_tiles);
    cb_exp_max_diff_obj.reserve_back(num_head_tiles);
    cb_exp_max_diff_2_obj.reserve_back(num_head_tiles);
    constexpr bool reuse_cur_max_tile = DST_ACCUM_MODE && DST_SYNC_MODE == DstSync::SyncHalf;
    static_assert(reuse_cur_max_tile);
    constexpr uint32_t worker_sum_dst = reuse_cur_max_tile ? 2 : 4;
    constexpr uint16_t scale_bf16 = scale_fp32 >> 16;
    for (uint32_t i = 0; i < num_head_tiles; i++) {
        sdo_regs_acquire();
        sdo_copy_init(cb_worker_max);
        sd_exp_tile_init<EXP_APPROX_MODE>();
        sdo_copy_tile(cb_prev_max, i, 0);
        sdo_copy_tile(cb_worker_max, i, 1);
        sdo_copy_tile(cb_prev_sum, i, 3);
        sdo_copy_tile(cb_worker_sum, i, worker_sum_dst);
#ifdef TRISC_MATH
        sdo_sfpu_start(0);
        sdv_seek_rows4(row0);
        for (uint32_t k = 0; k < SDB_N / 4; ++k) {
            if (k) {
                ckernel::math::inc_dst_addr<4>();
            }
            sdk_fused_rows4<scale_bf16>();
        }
        sdo_sfpu_done();
#endif
        sdo_regs_commit();
        sdo_regs_wait();
        sd_pack_tile(0, cb_exp_max_diff);
        sd_pack_tile(1, cb_exp_max_diff_2);
        sd_pack_tile(2, cb_cur_max);
        sd_pack_tile(3, cb_cur_sum);
        sdo_regs_release();
        cb_cur_max_obj.push_back(1);
        cb_cur_sum_obj.push_back(1);
        cb_exp_max_diff_obj.push_back(1);
        cb_exp_max_diff_2_obj.push_back(1);
    }
    cb_prev_sum_obj.pop_front(num_head_tiles);
    cb_worker_sum_obj.pop_front(num_head_tiles);
}

void sdk_recip_block_rows4(uint32_t in_cb, uint32_t num_tiles, uint32_t row0) {
    CircularBuffer cb_in(in_cb);
    sdo_reconfig_srca(in_cb);
    sdo_copy_init(in_cb);
    MATH((sdo_sfpu_init_unary<true>(sdv_recip_init)));
    sdo_pack_reconfig(in_cb);
    cb_in.wait_front(num_tiles);
    for (uint32_t i = 0; i < num_tiles; ++i) {
        sdo_regs_acquire();
        sdo_copy_tile(in_cb, i, 0);
#ifdef TRISC_MATH
        sdo_sfpu_start(0);
        sdv_seek_rows4(row0);
        for (uint32_t k = 0; k < SDB_N / 4; ++k) {
            if (k) {
                ckernel::math::inc_dst_addr<4>();
            }
            sdk_recip_rows4();
        }
        sdo_sfpu_done();
#endif
        sdo_regs_commit();
        sdo_regs_wait();
        sd_pack_tile(0, in_cb);
        sdo_regs_release();
    }
    cb_in.pop_front(num_tiles);
    cb_in.reserve_back(num_tiles);
    cb_in.push_back(num_tiles);
}
#endif

#ifdef SDB_C4
#ifndef SDB_C0
#error "SDB_C4 needs SDB_C0"
#endif
ALWI void sdk_trs_rnd_rows4(uint32_t idst, uint32_t row0) {
#ifdef TRISC_MATH
    sdv_rnd_rows<SDB_N / 2>(idst, row0);
#endif
}

template <uint32_t cols>
void sdk_trs_rescale(
    uint32_t cb_prev_sum,
    uint32_t cb_cur_sum,
    uint32_t cb_exp_max_diff,
    uint32_t cb_out_acc,
    uint32_t cb_out_im,
    uint32_t row0) {
    static_assert(cols <= 4);
    CircularBuffer cps(cb_prev_sum);
    CircularBuffer ccs(cb_cur_sum);
    CircularBuffer cem(cb_exp_max_diff);
    CircularBuffer coa(cb_out_acc);
    CircularBuffer coi(cb_out_im);

    sdo_reconfig(cb_prev_sum, cb_exp_max_diff);
    sdo_pack_reconfig(cb_cur_sum);
    sdh_binop_init<EltwiseBinaryType::ELWMUL, BroadcastType::NONE>();
    cps.wait_front(1);
    cem.wait_front(1);
    ccs.wait_front(1);
    sdo_regs_acquire();
    sd_mul_tiles(cb_prev_sum, cb_exp_max_diff, 0, 0, 0);
    sd_binop_with_scalar_tile_init();
    sdk_trs_rnd_rows4(0, row0);
    MATH((TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::WAIT_SFPU)));
    sdh_reuse_init<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCB>();
    sd_add_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCB>(cb_cur_sum, 0, 0);
    sdo_regs_commit();
    cps.pop_front(1);
    ccs.pop_front(1);
    ccs.reserve_back(1);
    sdo_regs_wait();
    sd_pack_tile(0, cb_cur_sum);
    sdo_regs_release();
    ccs.push_back(1);

    sdo_reconfig(cb_out_acc, cb_exp_max_diff);
    sdo_pack_reconfig(cb_out_acc);
    sdh_binop_init<EltwiseBinaryType::ELWMUL, BroadcastType::COL>();
    coa.wait_front(cols);
    sdo_regs_acquire();
    for (uint32_t j = 0; j < cols; ++j) {
        sd_mul_tiles_bcast_cols(cb_out_acc, cb_exp_max_diff, j, 0, j);
    }
    sd_binop_with_scalar_tile_init();
    for (uint32_t j = 0; j < cols; ++j) {
        sdk_trs_rnd_rows4(j, row0);
    }
    MATH((TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::WAIT_SFPU)));
    sdo_reconfig(cb_out_acc, cb_out_im);
    sdh_reuse_init<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>();
    coi.wait_front(cols);
    for (uint32_t j = 0; j < cols; ++j) {
        sd_add_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_out_im, j, j);
    }
    sdo_regs_commit();
    coa.pop_front(cols);
    cem.pop_front(1);
    coi.pop_front(cols);
    coa.reserve_back(cols);
    sdo_regs_wait();
    for (uint32_t j = 0; j < cols; ++j) {
        sd_pack_tile(j, cb_out_acc);
    }
    sdo_regs_release();
    coa.push_back(cols);
}
#endif

#ifdef SDB_C4
template <uint32_t cols>
void sdk_trm_child(uint32_t cb_child_o, uint32_t cb_e2, uint32_t cb_acc, uint32_t row0, uint32_t cb_dst) {
    static_assert(cols <= 4);
    CircularBuffer cco(cb_child_o);
    CircularBuffer ce2(cb_e2);
    CircularBuffer cac(cb_acc);
    sdo_reconfig(cb_child_o, cb_e2);
    sdo_pack_reconfig(cb_acc);
    sdh_binop_init<EltwiseBinaryType::ELWMUL, BroadcastType::COL>();
    cco.wait_front(cols);
    ce2.wait_front(1);
    sdo_regs_acquire();
    for (uint32_t j = 0; j < cols; ++j) {
        sd_mul_tiles_bcast_cols(cb_child_o, cb_e2, j, 0, j);
    }
    sd_binop_with_scalar_tile_init();
    for (uint32_t j = 0; j < cols; ++j) {
        sdk_trs_rnd_rows4(j, row0);
    }
    MATH((TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::WAIT_SFPU)));
    sdo_reconfig(cb_acc, cb_child_o);
    sdh_reuse_init<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCB>();
    cac.wait_front(cols);
    for (uint32_t j = 0; j < cols; ++j) {
        sd_add_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCB>(cb_acc, j, j);
    }
    sdo_regs_commit();
    cco.pop_front(cols);
    ce2.pop_front(1);
    cac.pop_front(cols);
    CircularBuffer cds(cb_dst);
    cds.reserve_back(cols);
    sdo_regs_wait();
    for (uint32_t j = 0; j < cols; ++j) {
        sd_pack_tile(j, cb_dst);
    }
    sdo_regs_release();
    cds.push_back(cols);
}
#endif
