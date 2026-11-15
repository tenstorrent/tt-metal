// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#define REDUCE_OP (PoolType::MAX)
#define REDUCE_DIM (ReduceDim::REDUCE_ROW)

#include "api/debug/assert.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/binary_max_min.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/eltwise_unary/softplus.h"
#include "api/compute/eltwise_unary/negative.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/bcast.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/matmul.h"
#include "api/compute/reduce.h"
#include "api/compute/reduce_custom.h"
#include "api/dataflow/circular_buffer.h"
#include "q_chunk_remapping.hpp"

#include "cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/chunked_prefill_utils.hpp"
#include "cpp/ttnn/kernel_lib/dest_helpers.hpp"
#if defined(TRISC_MATH) || defined(TRISC_PACK)
#include "experimental/llk_sfpu/ckernel_sfpu_sdpa.h"
#endif

inline uint32_t last_pack_format = 0xFFFFFFFFu;
template <typename... PackArgs>
ALWI void pack_reconfig_data_format_cached(PackArgs... a) {
#if defined(UCK_CHLKC_PACK)
    if constexpr (sizeof...(PackArgs) == 1) {
        const uint32_t v[1] = {uint32_t(a)...};
        const uint32_t f = uint32_t(pack_src_format[v[0]]) | (uint32_t(pack_dst_format[v[0]]) << 8);
        if (f != last_pack_format) {
            pack_reconfig_data_format(a...);
            last_pack_format = f;
        }
        return;
    }
    last_pack_format = 0xFFFFFFFFu;
#endif
    pack_reconfig_data_format(a...);
}
#define pack_reconfig_data_format pack_reconfig_data_format_cached
#if defined(TRISC_MATH)
namespace ckernel::sfpu {
template <bool APPROXIMATE, bool IS_FP32, int ITERATIONS>
inline void scaled_exp_fp32(const std::uint32_t scale_bits) {
    TT_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_LOWER, scale_bits & 0xFFFFu);
    TT_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_UPPER, scale_bits >> 16);
    TTI_SFPCONFIG(0, 11, 0);
    _sfpu_exp_fp32_load_imm32_<p_sfpu::LREG4, __builtin_bit_cast(std::uint32_t, EXP_FP32_P1)>();
    _sfpu_exp_fp32_load_imm32_<p_sfpu::LREG5, __builtin_bit_cast(std::uint32_t, EXP_FP32_C2)>();
    _sfpu_exp_fp32_load_imm32_<p_sfpu::LREG6, __builtin_bit_cast(std::uint32_t, EXP_FP32_C3)>();
    _sfpu_exp_fp32_load_imm32_<p_sfpu::LREG7, __builtin_bit_cast(std::uint32_t, EXP_FP32_C4)>();
    constexpr std::uint32_t HEAD_LEN = 16;
    static_assert(HEAD_LEN <= ckernel::math::replay_buf_offset);
    constexpr std::uint32_t scale_reg = p_sfpu::LREG11;
    auto head = [] {
        TTI_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::DEFAULT, ADDR_MOD_7, 0);
        TTI_SFPMUL(p_sfpu::LREG1, scale_reg, p_sfpu::LCONST_0, p_sfpu::LREG1, 0);
        TTI_SFPMUL(p_sfpu::LREG1, p_sfpu::LREG12, p_sfpu::LCONST_0, p_sfpu::LREG3, 0);
        TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_FLOATA, EXP_FP32_R0_FP16A);
        TTI_SFP_STOCH_RND(
            sfpi::SFPSTOCHRND_RND_NEAREST,
            0,
            p_sfpu::LREG0,
            p_sfpu::LREG3,
            p_sfpu::LREG3,
            sfpi::SFPSTOCHRND_MOD1_FP32_TO_SMAG16);
        TTI_SFPCAST(p_sfpu::LREG3, p_sfpu::LREG2, sfpi::SFPCAST_MOD1_SM32_TO_FP32_RNE);
        TTI_SFPMAD(p_sfpu::LREG2, p_sfpu::LREG13, p_sfpu::LREG1, p_sfpu::LREG1, 0);
        TTI_SFPMAD(p_sfpu::LREG2, p_sfpu::LREG14, p_sfpu::LREG1, p_sfpu::LREG1, 0);
        TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LREG4, p_sfpu::LREG0, 0);
        TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LREG5, p_sfpu::LREG0, 0);
        TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LREG6, p_sfpu::LREG0, 0);
        TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LREG7, p_sfpu::LREG0, 0);
        TTI_SFPCAST(p_sfpu::LREG3, p_sfpu::LREG3, sfpi::SFPCAST_MOD1_INT32_TO_SM32);
        TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LCONST_1, p_sfpu::LREG0, 0);
        TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LCONST_1, p_sfpu::LREG1, 0);
        TTI_SFPMULI(EXP_FP32_INF_FP16B, p_sfpu::LREG0, 0);
    };
    auto tail = [] {
        TTI_SFPEXEXP(0, p_sfpu::LREG1, p_sfpu::LREG2, sfpi::SFPEXEXP_MOD1_NODEBIAS);
        TTI_SFPIADD(0, p_sfpu::LREG3, p_sfpu::LREG2, sfpi::SFPIADD_MOD1_ARG_LREG_DST | sfpi::SFPIADD_MOD1_CC_NONE);
        TTI_SFPIADD(0xF01, p_sfpu::LREG2, p_sfpu::LREG3, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_LT0);
        TTI_SFPMOV(0, p_sfpu::LREG2, p_sfpu::LREG0, 0);
        TTI_SFPSETEXP(0, p_sfpu::LREG1, p_sfpu::LREG0, 0);
        TTI_SFPIADD(0xFFF, p_sfpu::LREG2, p_sfpu::LREG2, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_LT0);
        TTI_SFPMOV(0, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);
        TTI_SFPENCC(sfpi::SFPENCC_IMM12_BOTH, 0, 0, sfpi::SFPENCC_MOD1_EI_RI);
        TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_6, 0);
    };
    load_replay_buf<Exec>(0, HEAD_LEN, head);
    tail();
#pragma GCC unroll 8
    for (int d = 1; d < ITERATIONS; d++) {
        TTI_REPLAY(0, HEAD_LEN, 0, 0);
        tail();
    }
    TTI_SFPCONFIG(0, 11, 1);
}
}  // namespace ckernel::sfpu
#endif
template <uint32_t SCALE_BITS>
ALWI void scaled_exp_tile(uint32_t idst) {
    MATH((SFPU_UNARY_CALL(
        DST_SYNC_MODE, DST_ACCUM_MODE, scaled_exp_fp32, (false, true, 32), idst, VectorMode::None, SCALE_BITS)));
}

#ifdef TRISC_UNPACK
inline void unpack_reduce_scaler_dummy_a(const uint32_t cb) {
    const uint32_t id = get_operand_id(cb);
    const uint32_t addr = get_local_cb_interface(id).fifo_rd_ptr - 1;
    TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111);
    volatile uint32_t* cfg = get_cfg_pointer();
    wait_for_next_context(2);
    _llk_unpack_configure_addresses_(addr, addr, cfg);
    semaphore_post(semaphore::UNPACK_SYNC);
    TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG);
    for (uint32_t f = 0; f < 4; f++) {
        TTI_UNPACR_NOP(
            SrcA, 0, 0, p_unpacr_nop::SET_DVALID, 0, p_unpacr_nop::WAIT_LIKE_UNPACR, 0, 0, p_unpacr_nop::UNP_ZEROSRC);
        TTI_UNPACR(SrcB, 0b01, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    }
    t6_semaphore_get(semaphore::UNPACK_SYNC);
    switch_config_context(unp_cfg_context);
}
#endif
#ifdef TRISC_MATH
template <uint32_t DB>
inline void face_to_srca(const uint32_t s_tile, const uint32_t m_tile) {
    math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(s_tile);
    TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::MATH | p_stall::SRCA_VLD | p_stall::SRCB_VLD);
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_AB);
    TTI_MOVD2B(0, p_movd2b::SRC_ROW16_OFFSET + 0, ADDR_MOD_7, p_movd2b::MOV_4_ROWS, DB + 0);
    TTI_MOVD2B(0, p_movd2b::SRC_ROW16_OFFSET + 4, ADDR_MOD_7, p_movd2b::MOV_4_ROWS, DB + 4);
    TTI_MOVD2B(0, p_movd2b::SRC_ROW16_OFFSET + 8, ADDR_MOD_7, p_movd2b::MOV_4_ROWS, DB + 8);
    TTI_MOVD2B(0, p_movd2b::SRC_ROW16_OFFSET + 12, ADDR_MOD_7, p_movd2b::MOV_4_ROWS, DB + 12);
    TTI_TRNSPSRCB;
    TTI_MOVB2A(p_movb2a::SRCA_ZERO_OFFSET + 0, ADDR_MOD_7, p_movb2a::MOV_4_ROWS, p_movb2a::SRCB_ROW16_OFFSET + 0);
    TTI_MOVB2A(p_movb2a::SRCA_ZERO_OFFSET + 4, ADDR_MOD_7, p_movb2a::MOV_4_ROWS, p_movb2a::SRCB_ROW16_OFFSET + 4);
    TTI_MOVB2A(p_movb2a::SRCA_ZERO_OFFSET + 8, ADDR_MOD_7, p_movb2a::MOV_4_ROWS, p_movb2a::SRCB_ROW16_OFFSET + 8);
    TTI_MOVB2A(p_movb2a::SRCA_ZERO_OFFSET + 12, ADDR_MOD_7, p_movb2a::MOV_4_ROWS, p_movb2a::SRCB_ROW16_OFFSET + 12);
    math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(m_tile);
}
inline void rowmax_from_dest(const uint32_t s_tile, const uint32_t m_tile) {
    math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(m_tile);
    face_to_srca<0>(s_tile, m_tile);
    reduce_pool_op<PoolType::MAX, false, p_setrwc::CLR_AB, 0>();
    face_to_srca<16>(s_tile, m_tile);
    reduce_pool_op<PoolType::MAX, false, p_setrwc::CLR_NONE, 0>();
    reduce_row_perform_transpose<false>();
    reduce_row_advance_dest(false);
    face_to_srca<0>(s_tile, m_tile);
    reduce_pool_op<PoolType::MAX, false, p_setrwc::CLR_AB, 0>();
    face_to_srca<16>(s_tile, m_tile);
    reduce_pool_op<PoolType::MAX, false, p_setrwc::CLR_NONE, 0>();
    reduce_row_perform_transpose<false>();
    TTI_SETRWC(p_setrwc::CLR_AB, 0, 0, 0, 0, p_setrwc::SET_BD);
}
inline void addr_mod7_zero() {
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}}.set(ADDR_MOD_7);
}
inline void math_take_dest() {
    tensix_sync();
    while (semaphore_read(semaphore::MATH_PACK) > 0) {
    }
    reset_dest_offset_id();
    set_dest_section_base<StartZero>();
    TTI_ZEROACC(p_zeroacc::CLR_ALL, DST_ACCUM_MODE, 0, ADDR_MOD_1, 0);
    TTI_STALLWAIT(p_stall::STALL_SFPU | p_stall::STALL_SYNC, p_stall::MATH);
}
#endif
#ifdef TRISC_PACK
inline void pack_take_dest() {
    reset_dest_offset_id();
    select_packer_dest_registers<DstSync::SyncHalf>();
}
#endif
template <uint32_t cb_qk_im, uint32_t cb_scale, uint32_t Sq, uint32_t Sk, uint32_t DHt, uint32_t scale_fp32>
void sdpa_qk_max_sub_exp(
    const uint32_t cb_q,
    const uint32_t cb_k,
    const uint32_t cb_cur_max,
    const uint32_t cb_prev_max,
    const uint32_t cb_cur_sum,
    const bool do_eltwise_max) {
    constexpr uint32_t S0 = 4;
    MATH((math_take_dest()));
    PACK((pack_take_dest()));
    CircularBuffer(cb_q).wait_front(Sq * DHt);
    CircularBuffer(cb_k).wait_front(DHt * Sk);
    reconfig_data_format(cb_k, cb_q);
    matmul_block_init(cb_q, cb_k, true, Sk, Sq, DHt);
    for (uint32_t inner = 0; inner < DHt; inner++) {
        matmul_block(cb_q, cb_k, inner, inner * Sk, S0, true, Sk, Sq, DHt);
    }
    CircularBuffer(cb_k).pop_front(DHt * Sk);
    CircularBuffer(cb_scale).wait_front(1);
    CircularBuffer(cb_cur_max).reserve_back(Sq);
    binary_max_tile_init();
    reconfig_data_format(cb_qk_im, cb_scale);
    reduce_init<PoolType::MAX, ReduceDim::REDUCE_ROW>(cb_qk_im, cb_scale, cb_cur_max);
    MATH((addr_mod7_zero()));
    for (uint32_t r = 0; r < Sq; r++) {
        for (uint32_t j = 0; j < Sk; j++) {
            UNPACK((unpack_reduce_scaler_dummy_a(cb_scale)));
            MATH((rowmax_from_dest(S0 + r * Sk + j, r)));
        }
    }
    reduce_uninit();
    if (do_eltwise_max) {
        reconfig_data_format_srca(cb_prev_max);
        copy_init(cb_prev_max);
        for (uint32_t r = 0; r < Sq; r++) {
            copy_tile(cb_prev_max, r, Sq + r);
        }
        for (uint32_t r = 0; r < Sq; r++) {
            binary_max_tile(r, Sq + r, r, VectorMode::C);
        }
    }
    MATH((set_math_semaphores()));
    PACK((_llk_packer_wait_for_math_done_()));
    pack_reconfig_data_format(cb_cur_max);
    for (uint32_t r = 0; r < Sq; r++) {
        pack_tile(r, cb_cur_max);
    }
    PACK((_llk_packer_set_math_semaphore_<p_stall::PACK>()));
    CircularBuffer(cb_cur_max).push_back(Sq);
    reconfig_data_format(cb_qk_im, cb_cur_max);
    UNPACK((llk_unpack_A_init<BroadcastType::COL, true, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(
        false, false, cb_cur_max)));
    MATH((llk_math_eltwise_binary_init<
          EltwiseBinaryType::ELWSUB,
          BroadcastType::COL,
          MathFidelity::LoFi,
          EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_cur_max, cb_cur_max, false)));
    PACK((llk_pack_relu_config(ReluConfig::zero())));
    CircularBuffer(cb_cur_max).wait_front(Sq);
    CircularBuffer(cb_cur_sum).reserve_back(Sq);
    for (uint32_t i = 0; i < Sq; i++) {
        MATH((math_dest_wait()));
        for (uint32_t j = 0; j < Sk; j++) {
            UNPACK((llk_unpack_A<BroadcastType::COL, true, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_cur_max, i)));
            MATH((llk_math_eltwise_binary<
                  EltwiseBinaryType::ELWSUB,
                  BroadcastType::COL,
                  DST_ACCUM_MODE,
                  MathFidelity::LoFi,
                  EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_cur_max, cb_cur_max, S0 + i * Sk + j, true)));
            exp_tile_init<false, 0x3F800000, InputClamping::ClampToNegative>();
            scaled_exp_tile<scale_fp32>(S0 + i * Sk + j);
        }
        MATH((set_math_semaphores()));
        PACK((_llk_packer_wait_for_math_done_()));
        CircularBuffer(cb_qk_im).reserve_back(Sk);
        pack_reconfig_data_format(cb_qk_im);
        for (uint32_t j = 0; j < Sk; j++) {
            pack_tile(S0 + i * Sk + j, cb_qk_im);
        }
        CircularBuffer(cb_qk_im).push_back(Sk);
        pack_reconfig_data_format(cb_cur_sum);
        for (uint32_t j = 0; j < Sk; j++) {
            pack_tile<true>(S0 + i * Sk + j, cb_cur_sum, i);
            if (j == 0) {
                PACK((llk_pack_reconfig_l1_acc(1)));
            }
        }
        PACK((_llk_packer_set_math_semaphore_<p_stall::PACK>()));
        PACK((llk_pack_reconfig_l1_acc(0)));
    }
    CircularBuffer(cb_cur_sum).push_back(Sq);
    PACK((llk_pack_relu_config(ReluConfig::none())));
    MATH((math_take_dest()));
    PACK((pack_take_dest()));
}

ALWI void sdpa_reduce_copy_tile_to_dst_init_short(uint32_t cbid, uint32_t transpose = 0) {
    UNPACK((llk_unpack_A_init<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, UnpackToDestEn>(
        transpose, true, cbid)));

    MATH((llk_math_eltwise_unary_datacopy_init<
          DataCopyType::A2D,
          DST_ACCUM_MODE,
          BroadcastType::NONE,
          false,
          PackMode::Default>(cbid)));
}

template <uint32_t num_tiles>
void max_block_inplace(uint32_t in0, uint32_t in1) {
    CircularBuffer cb_in0(in0);
    CircularBuffer cb_in1(in1);
    copy_init(in0);
    copy_init(in1);
    binary_max_tile_init();
    constexpr uint32_t dst_reg_0 = 0;
    constexpr uint32_t dst_reg_1 = 1;
    cb_in0.wait_front(num_tiles);
    cb_in1.wait_front(num_tiles);
    for (uint32_t i = 0; i < num_tiles; ++i) {
        tile_regs_acquire();
        copy_tile(in0, i, dst_reg_0);
        copy_tile(in1, i, dst_reg_1);
        binary_max_tile(dst_reg_0, dst_reg_1, dst_reg_0, VectorMode::C);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(dst_reg_0, in0);
        tile_regs_release();
    }
    cb_in0.pop_front(num_tiles);
    cb_in0.reserve_back(num_tiles);
    cb_in0.push_back(num_tiles);
}

#ifdef TRISC_MATH
inline void sdpa_max_sfpi(const std::uint32_t in0, const std::uint32_t in1, const std::uint32_t out) {
    constexpr std::uint32_t dst_tile_size_sfpi = 32;
#pragma GCC unroll 8
    for (int d = 0; d < 8; d++) {
        sfpi::vFloat a = sfpi::dst_reg[in0 * dst_tile_size_sfpi];
        const sfpi::vFloat b = sfpi::dst_reg[in1 * dst_tile_size_sfpi];
        v_if(b > a) { a = b; }
        v_endif;
        sfpi::dst_reg[out * dst_tile_size_sfpi] = a;
        sfpi::dst_reg++;
    }
}
#endif

void max_block_sfpi(uint32_t in0, uint32_t in1, uint32_t out_cb, uint32_t num_tiles) {
    CircularBuffer cb_in0(in0);
    CircularBuffer cb_in1(in1);
    CircularBuffer cb_out(out_cb);
    copy_init(in0);
    add_binary_tile_init();
    cb_in0.wait_front(num_tiles);
    cb_in1.wait_front(num_tiles);
    cb_out.reserve_back(num_tiles);
    for (uint32_t i = 0; i < num_tiles; ++i) {
        tile_regs_acquire();
        copy_tile(in0, i, 0);
        copy_tile(in1, i, 1);
        MATH((_llk_math_eltwise_binary_sfpu_params_(sdpa_max_sfpi, 0, 1, 0, VectorMode::RC)));
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, out_cb, i);
        tile_regs_release();
    }
    cb_out.push_back(num_tiles);
}

void zero_block(uint32_t like_cb, uint32_t out_cb, uint32_t num_tiles) {
    CircularBuffer cb_like(like_cb);
    CircularBuffer cb_out(out_cb);
    sub_init(like_cb, like_cb);
    cb_like.wait_front(1);
    cb_out.reserve_back(num_tiles);
    tile_regs_acquire();
    sub_tiles(like_cb, like_cb, 0, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t i = 0; i < num_tiles; ++i) {
        pack_tile(0, out_cb);
    }
    tile_regs_release();
    cb_out.push_back(num_tiles);
}

template <VectorMode vector_mode = VectorMode::RC>
void max_block(uint32_t in0, uint32_t in1, uint32_t out_cb, uint32_t num_tiles) {
    CircularBuffer cb_in0(in0);
    CircularBuffer cb_in1(in1);
    CircularBuffer cb_out(out_cb);
    copy_init(in0);
    binary_max_tile_init();

    constexpr uint32_t dst_reg_0 = 0;
    constexpr uint32_t dst_reg_1 = 1;
    cb_in0.wait_front(num_tiles);
    cb_in1.wait_front(num_tiles);
    cb_out.reserve_back(num_tiles);
    for (uint32_t i = 0; i < num_tiles; ++i) {
        tile_regs_acquire();
        copy_tile(in0, i, dst_reg_0);
        copy_tile(in1, i, dst_reg_1);
        binary_max_tile(dst_reg_0, dst_reg_1, dst_reg_0, vector_mode);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(dst_reg_0, out_cb, i);
        tile_regs_release();
    }
    cb_out.push_back(num_tiles);
}

template <
    PoolType pool_type,
    ReduceDim reduce_dim,
    uint32_t in0_cb,
    uint32_t scale_cb,
    uint32_t rows,
    uint32_t cols,
    VectorMode vector_mode = VectorMode::C>
void reduce_c(uint32_t out_cb, uint32_t prev_cb, bool do_eltwise_max = false) {
    CircularBuffer cb_in0(in0_cb);
    CircularBuffer cb_scale(scale_cb);
    CircularBuffer cb_out(out_cb);
    CircularBuffer cb_prev(prev_cb);

#if defined REDUCE_GRANULARITY
    constexpr uint32_t dst_tiles = (rows < REDUCE_GRANULARITY) ? rows : REDUCE_GRANULARITY;
    constexpr uint32_t granularity = (rows >= REDUCE_GRANULARITY) ? (rows / REDUCE_GRANULARITY) : 1;
#else
    constexpr uint32_t dst_tiles = 1;
    constexpr uint32_t granularity = rows;
#endif

    cb_scale.wait_front(1);
    cb_out.reserve_back(rows);

    const uint32_t num_tiles_to_wait = dst_tiles * cols;
    uint32_t in0_wait_tiles = num_tiles_to_wait;

    uint32_t row_start_idx = 0;
    for (uint32_t g = 0; g < granularity; g++) {
        cb_in0.wait_front(in0_wait_tiles);
        tile_regs_acquire();

        if (do_eltwise_max) {
            cb_prev.wait_front(g * dst_tiles);
            reconfig_data_format_srca(prev_cb);
            sdpa_reduce_copy_tile_to_dst_init_short(prev_cb);
            for (uint32_t i = 0; i < dst_tiles; i++) {
                const uint32_t cur_max_dst_idx = i;
                copy_tile(prev_cb, (row_start_idx + i), cur_max_dst_idx);
            }
            reconfig_data_format_srca(in0_cb);
        }

        reduce_block_max_row_init<cols>(out_cb);
        for (uint32_t i = 0; i < dst_tiles; i++) {
            const uint32_t reduce_dst_idx = i;
            reduce_block_max_row<cols>(in0_cb, scale_cb, (row_start_idx + i) * cols, reduce_dst_idx);
        }
        reduce_block_max_row_uninit(in0_cb);

        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(out_cb);
        for (uint32_t i = 0; i < dst_tiles; i++) {
            const uint32_t cur_max_dst_idx = i;
            pack_tile<true>(cur_max_dst_idx, out_cb, (row_start_idx + i));
        }
        tile_regs_release();

        row_start_idx += dst_tiles;
        in0_wait_tiles += num_tiles_to_wait;
    }

    cb_out.push_back(rows);
}

template <
    PoolType pool_type,
    ReduceDim reduce_dim,
    uint32_t in0_cb,
    uint32_t scale_cb,
    uint32_t rows,
    VectorMode vector_mode = VectorMode::C>
void reduce_c(uint32_t out_cb, uint32_t prev_cb, uint32_t cols, bool do_eltwise_max = false) {
    CircularBuffer cb_in0(in0_cb);
    CircularBuffer cb_scale(scale_cb);
    CircularBuffer cb_out(out_cb);

    uint32_t num_tiles = rows * cols;
    cb_scale.wait_front(1);
    cb_in0.wait_front(num_tiles);
    cb_out.reserve_back(rows);

    pack_reconfig_data_format(out_cb);

    binary_max_tile_init();
    constexpr uint32_t reduce_dst_idx = 0;
    constexpr uint32_t prev_max_dst_idx = 1;

    constexpr uint32_t rows_per_group =
        (compute_kernel_lib::DEST_AUTO_LIMIT / 2 < rows) ? compute_kernel_lib::DEST_AUTO_LIMIT / 2 : rows;
    for (uint32_t i0 = 0; i0 < rows; i0 += rows_per_group) {
        const uint32_t g = (rows - i0 < rows_per_group) ? rows - i0 : rows_per_group;
        reconfig_data_format_srca(in0_cb);
        tile_regs_acquire();
        reduce_init<pool_type, reduce_dim>(in0_cb, scale_cb, out_cb);
        for (uint32_t r = 0; r < g; r++) {
            for (uint32_t j = 0; j < cols; j++) {
                reduce_tile<pool_type, reduce_dim>(in0_cb, scale_cb, (i0 + r) * cols + j, 0, r);
            }
        }
        reduce_uninit();
        if (do_eltwise_max) {
            reconfig_data_format_srca(prev_cb);
            copy_init(prev_cb);
            for (uint32_t r = 0; r < g; r++) {
                copy_tile(prev_cb, i0 + r, rows_per_group + r);
            }
            for (uint32_t r = 0; r < g; r++) {
                binary_max_tile(r, rows_per_group + r, r, vector_mode);
            }
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t r = 0; r < g; r++) {
            pack_tile(r, out_cb);
        }
        tile_regs_release();
    }

    cb_out.push_back(rows);
}

#ifdef TRISC_MATH
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
void recip_tile_first_column(uint32_t idst) {
    SFPU_UNARY_CALL(
        DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_recip_first_column, (is_fp32_dest_acc_en), idst, VectorMode::C);
}
#endif

void recip_block_inplace(uint32_t in_cb, uint32_t num_tiles) {
    CircularBuffer cb_in(in_cb);
    reconfig_data_format_srca(in_cb);
    copy_init(in_cb);
    MATH(SFPU_UNARY_INIT_FN(reciprocal, sfpu::sfpu_reciprocal_init, (APPROX)));
    pack_reconfig_data_format(in_cb);

    cb_in.wait_front(num_tiles);
    for (uint32_t i = 0; i < num_tiles; ++i) {
        tile_regs_acquire();
        copy_tile(in_cb, i, 0);
        MATH((recip_tile_first_column(0)));
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, in_cb);
        tile_regs_release();
    }
    cb_in.pop_front(num_tiles);
    cb_in.reserve_back(num_tiles);
    cb_in.push_back(num_tiles);
}

template <
    uint32_t in0_cb,
    uint32_t rows,
    uint32_t scale_fp32,
    bool write_result_inplace = true,
    bool do_reduce = true,
    VectorMode vector_mode = VectorMode::RC>
void sub_exp_block_bcast_cols_inplace(uint32_t in1_cb, uint32_t reduce_cb, uint32_t cols) {
    CircularBuffer cb_in0(in0_cb);
    CircularBuffer cb_in1(in1_cb);
    CircularBuffer cb_reduce(reduce_cb);
    reconfig_data_format(in0_cb, in1_cb);
    sub_bcast_cols_init(in0_cb, in1_cb);

    if constexpr (EXP_APPROX_MODE || vector_mode != VectorMode::RC) {
        exp_tile_init<true, scale_fp32, InputClamping::None>();
    }
    PACK((llk_pack_relu_config(ReluConfig::zero())));

    cb_in0.wait_front(rows * cols);
    cb_in1.wait_front(rows);
    if constexpr (do_reduce) {
        cb_reduce.reserve_back(rows);
    }

#ifdef SUB_EXP_GRANULARITY
    uint32_t dst_tiles = (cols < SUB_EXP_GRANULARITY) ? cols : SUB_EXP_GRANULARITY;
    uint32_t granularity = (cols >= SUB_EXP_GRANULARITY) ? (cols / SUB_EXP_GRANULARITY) : 1;
#else
    uint32_t dst_tiles = cols;
    uint32_t granularity = 1;
#endif

    for (uint32_t i = 0; i < rows; ++i) {
        for (uint32_t u = 0; u < granularity; u++) {
            tile_regs_acquire();
            for (uint32_t j = 0; j < dst_tiles; ++j) {
                sub_tiles_bcast_cols(in0_cb, in1_cb, j, i, j);
                constexpr int iterations = (vector_mode == VectorMode::RC) ? 32 : 8;
                constexpr VectorMode vector_mode_exp = (vector_mode == VectorMode::RC) ? VectorMode::None : vector_mode;
                if constexpr (EXP_APPROX_MODE || vector_mode != VectorMode::RC) {
                    exp_tile<true, false, InputClamping::None, iterations>(j, vector_mode_exp);
                } else {
                    exp_tile_init<false, 0x3F800000, InputClamping::ClampToNegative>();
                    scaled_exp_tile<scale_fp32>(j);
                }
            }
            tile_regs_commit();

            if constexpr (write_result_inplace) {
                cb_in0.pop_front(dst_tiles);
                cb_in0.reserve_back(dst_tiles);
            }

            tile_regs_wait();

            if constexpr (write_result_inplace) {
                pack_reconfig_data_format(in0_cb);
                for (uint32_t j = 0; j < dst_tiles; ++j) {
                    pack_tile(j, in0_cb);
                }
                cb_in0.push_back(dst_tiles);
            }

            if constexpr (do_reduce) {
                pack_reconfig_data_format(reduce_cb);
                if (u > 0) {
                    PACK((llk_pack_reconfig_l1_acc(1)));
                }
                for (uint32_t j = 0; j < dst_tiles; ++j) {
                    pack_tile<true>(j, reduce_cb, i);
                    if (u == 0 && j == 0) {
                        PACK((llk_pack_reconfig_l1_acc(1)));
                    }
                }
            }
            tile_regs_release();
            if constexpr (do_reduce) {
                PACK((llk_pack_reconfig_l1_acc(0)));
            }
        }
    }
    if constexpr (do_reduce) {
        cb_reduce.push_back(rows);
    }

    PACK((llk_pack_relu_config(ReluConfig::none())));
}

template <uint32_t rows, uint32_t cols, bool immediate_pop, bool pack_accumulate>
void mul_block_bcast_cols(uint32_t in0_cb, uint32_t in1_cb, uint32_t out_cb) {
    CircularBuffer cb_in0(in0_cb);
    CircularBuffer cb_in1(in1_cb);
    CircularBuffer cb_out(out_cb);

    constexpr uint32_t num_tiles = rows * cols;

    reconfig_data_format(in0_cb, in1_cb);
    pack_reconfig_data_format(out_cb);
    mul_bcast_cols_init(in0_cb, in1_cb);
    cb_in0.wait_front(num_tiles);
    cb_in1.wait_front(rows);

    if constexpr (immediate_pop) {
        static_assert(!pack_accumulate, "Unsupported parameter configuration");
        for (uint32_t i = 0; i < rows; ++i) {
            for (uint32_t j = 0; j < cols; ++j) {
                tile_regs_acquire();
                mul_tiles_bcast_cols(in0_cb, in1_cb, 0, i, 0);
                tile_regs_commit();
                cb_in0.pop_front(1);
                cb_out.reserve_back(1);
                tile_regs_wait();
                pack_tile(0, out_cb);
                tile_regs_release();
                cb_out.push_back(1);
            }
        }
        cb_in1.pop_front(rows);
    } else {
#ifdef DHT_GRANULARITY
        constexpr uint32_t dst_tiles = (cols < DHT_GRANULARITY) ? cols : DHT_GRANULARITY;
        constexpr uint32_t granularity = (cols >= DHT_GRANULARITY) ? (cols / DHT_GRANULARITY) : 1;
#else
        constexpr uint32_t dst_tiles = 1;
        constexpr uint32_t granularity = cols;
#endif
        PACK((llk_pack_reconfig_l1_acc(pack_accumulate)));
        if (!pack_accumulate) {
            cb_out.reserve_back(num_tiles);
        }
        uint32_t in0_index = 0;
        for (uint32_t i = 0; i < rows; ++i) {
            for (uint32_t u = 0; u < granularity; ++u) {
                tile_regs_acquire();
                for (uint32_t j = 0; j < dst_tiles; ++j) {
                    mul_tiles_bcast_cols(in0_cb, in1_cb, in0_index, i, j);
                    in0_index++;
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t j = 0; j < dst_tiles; ++j) {
                    pack_tile(j, out_cb);
                }
                tile_regs_release();
            }
        }
        cb_in1.pop_front(rows);
        cb_in0.pop_front(num_tiles);
        if (pack_accumulate) {
            PACK((llk_pack_reconfig_l1_acc(false)));
            cb_out.pop_front(num_tiles);
            cb_out.reserve_back(num_tiles);
            cb_out.push_back(num_tiles);
        } else {
            cb_out.push_back(num_tiles);
        }
    }
}

template <uint32_t rows, uint32_t cols>
void mul_block_bcast_cols_inplace(uint32_t in0_cb, uint32_t in1_cb) {
    CircularBuffer cb_in0(in0_cb);
    CircularBuffer cb_in1(in1_cb);

    constexpr uint32_t num_tiles = rows * cols;

#ifdef DHT_GRANULARITY
    constexpr uint32_t dst_tiles = (cols < DHT_GRANULARITY) ? cols : DHT_GRANULARITY;
    constexpr uint32_t granularity = (cols >= DHT_GRANULARITY) ? (cols / DHT_GRANULARITY) : 1;
#else
    constexpr uint32_t dst_tiles = 1;
    constexpr uint32_t granularity = cols;
#endif

    reconfig_data_format(in0_cb, in1_cb);
    mul_bcast_cols_init(in0_cb, in1_cb);
    pack_reconfig_data_format(in0_cb);
    cb_in0.wait_front(num_tiles);
    cb_in1.wait_front(rows);
    for (uint32_t i = 0; i < rows; ++i) {
        for (uint32_t u = 0; u < granularity; ++u) {
            tile_regs_acquire();
            for (uint32_t j = 0; j < dst_tiles; ++j) {
                mul_tiles_bcast_cols(in0_cb, in1_cb, j, i, j);
            }
            tile_regs_commit();
            cb_in0.pop_front(dst_tiles);
            cb_in0.reserve_back(dst_tiles);
            tile_regs_wait();
            for (uint32_t j = 0; j < dst_tiles; ++j) {
                pack_tile(j, in0_cb);
            }
            cb_in0.push_back(dst_tiles);
            tile_regs_release();
        }
    }
    cb_in1.pop_front(rows);
    reconfig_data_format_srcb(in0_cb);
}

template <uint32_t in1_scalar_cb, uint32_t num_tiles>
void mul_block_bcast_scalar_inplace(uint32_t in0_cb) {
    CircularBuffer cb_in0(in0_cb);
    CircularBuffer cb_in1_scalar(in1_scalar_cb);

#ifdef STATS_GRANULARITY
    constexpr uint32_t dst_tiles = STATS_GRANULARITY;
    constexpr uint32_t granularity = num_tiles / STATS_GRANULARITY;
#else
    constexpr uint32_t dst_tiles = 1;
    constexpr uint32_t granularity = num_tiles;
#endif

    reconfig_data_format(in0_cb, in1_scalar_cb);
    mul_bcast_scalar_init(in0_cb, in1_scalar_cb);
    cb_in0.wait_front(num_tiles);
    cb_in1_scalar.wait_front(1);
    uint32_t in0_index = 0;
    for (uint32_t g = 0; g < granularity; ++g) {
        tile_regs_acquire();
        for (uint32_t i = 0; i < dst_tiles; ++i) {
            mul_tiles_bcast_scalar(in0_cb, in1_scalar_cb, in0_index, 0, i);
            in0_index++;
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t i = 0; i < dst_tiles; ++i) {
            pack_tile(i, in0_cb);
        }
        tile_regs_release();
    }
    cb_in0.pop_front(num_tiles);
    cb_in0.reserve_back(num_tiles);
    cb_in0.push_back(num_tiles);
}

template <bool pop_in1 = true>
void add_block_inplace(uint32_t in0_cb, uint32_t in1_cb, uint32_t num_tiles) {
    CircularBuffer cb_in0(in0_cb);
    CircularBuffer cb_in1(in1_cb);

    reconfig_data_format(in0_cb, in1_cb);
    pack_reconfig_data_format(in0_cb);
    add_init(in0_cb, in1_cb);
    cb_in0.wait_front(num_tiles);
    cb_in1.wait_front(num_tiles);
    for (uint32_t i = 0; i < num_tiles; i++) {
        tile_regs_acquire();
        add_tiles(in0_cb, in1_cb, i, i, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, in0_cb);
        tile_regs_release();
    }

    cb_in0.pop_front(num_tiles);
    if (pop_in1) {
        cb_in1.pop_front(num_tiles);
    }
    cb_in0.reserve_back(num_tiles);
    cb_in0.push_back(num_tiles);
}

void mul_tiles_bcast_cols_inplace(uint32_t in0_cb, uint32_t in1_cb, uint32_t num_tiles) {
    CircularBuffer cb_in0(in0_cb);
    CircularBuffer cb_in1(in1_cb);

    reconfig_data_format(in0_cb, in1_cb);
    mul_bcast_cols_init(in0_cb, in1_cb);
    pack_reconfig_data_format(in0_cb);
    cb_in0.wait_front(num_tiles);
    cb_in1.wait_front(num_tiles);
    for (uint32_t i = 0; i < num_tiles; i++) {
        tile_regs_acquire();
        mul_tiles_bcast_cols(in0_cb, in1_cb, 0, i, 0);
        tile_regs_commit();
        cb_in0.pop_front(1);
        cb_in0.reserve_back(1);
        tile_regs_wait();
        pack_tile(0, in0_cb);
        tile_regs_release();
        cb_in0.push_back(1);
    }
}

void mul_block_inplace(uint32_t in0_cb, uint32_t in1_cb, uint32_t num_tiles) {
    CircularBuffer cb_in0(in0_cb);
    CircularBuffer cb_in1(in1_cb);

    mul_init(in0_cb, in1_cb);
    cb_in0.wait_front(num_tiles);
    cb_in1.wait_front(num_tiles);
    for (uint32_t i = 0; i < num_tiles; i++) {
        invalidate_l1_cache();
        tile_regs_acquire();
        mul_tiles(in0_cb, in1_cb, 0, i, 0);
        tile_regs_commit();
        cb_in0.pop_front(1);
        cb_in0.reserve_back(1);
        tile_regs_wait();
        pack_tile(0, in0_cb);
        tile_regs_release();
        cb_in0.push_back(1);
    }
}

#if defined(TRISC_MATH) || defined(TRISC_PACK)

template <bool SDPA_EXP_APPROX_MODE, uint16_t scale_bf16, bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
void exp_tile_first_column(uint32_t idst) {
    SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        is_fp32_dest_acc_en,
        calculate_exponential_first_column,
        (SDPA_EXP_APPROX_MODE, scale_bf16, is_fp32_dest_acc_en),
        idst,
        VectorMode::C);
}
#endif

template <uint32_t scale_fp32>
void sub_exp_block(uint32_t in0_cb, uint32_t in1_cb, uint32_t out_cb, uint32_t num_tiles) {
    CircularBuffer cb_in0(in0_cb);
    CircularBuffer cb_in1(in1_cb);
    CircularBuffer cb_out(out_cb);

    sub_init(in0_cb, in1_cb);
    exp_tile_init<EXP_APPROX_MODE>();
    cb_in0.wait_front(num_tiles);
    cb_in1.wait_front(num_tiles);
    cb_out.reserve_back(num_tiles);

    constexpr uint16_t scale_bf16 = scale_fp32 >> 16;

    for (uint32_t i = 0; i < num_tiles; i++) {
        invalidate_l1_cache();
        tile_regs_acquire();
        sub_tiles(in0_cb, in1_cb, i, i, 0);
        MATH((exp_tile_first_column<EXP_APPROX_MODE, scale_bf16>(0)));
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, out_cb);
        tile_regs_release();
        cb_out.push_back(1);
    }
}

#ifdef TRISC_MATH
template <VectorMode vector_mode = VectorMode::C, bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
void fused_max_sub_exp_add_tile(uint32_t idst, int scale_bf16) {
    constexpr bool reuse_cur_max_tile = is_fp32_dest_acc_en && DST_SYNC_MODE == DstSync::SyncHalf;
    SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        is_fp32_dest_acc_en,
        calculate_fused_max_sub_exp_add_tile,
        (is_fp32_dest_acc_en, reuse_cur_max_tile),
        idst,
        vector_mode,
        scale_bf16);
}
#endif

template <uint32_t scale_fp32, VectorMode vector_mode = VectorMode::C>
void correction_block(
    uint32_t cb_worker_max,
    uint32_t cb_worker_sum,
    uint32_t cb_cur_max,
    uint32_t cb_prev_max,
    uint32_t cb_cur_sum,
    uint32_t cb_prev_sum,
    uint32_t cb_exp_max_diff,
    uint32_t cb_exp_max_diff_2,
    uint32_t num_head_tiles) {
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

    constexpr uint32_t dst_reg_0 = 0;
    constexpr uint32_t dst_reg_1 = 1;
    constexpr uint32_t dst_reg_2 = 2;
    constexpr uint32_t dst_reg_3 = 3;
    constexpr uint32_t dst_reg_4 = 4;
    constexpr uint32_t worker_sum_dst = (DST_ACCUM_MODE && DST_SYNC_MODE == DstSync::SyncHalf) ? dst_reg_2 : dst_reg_4;
    static_assert(
        worker_sum_dst < compute_kernel_lib::DEST_AUTO_LIMIT,
        "correction_block DST layout exceeds DEST capacity for this sync/accum mode");

    constexpr uint16_t scale_bf16 = scale_fp32 >> 16;

    for (uint32_t i = 0; i < num_head_tiles; i++) {
        tile_regs_acquire();
        copy_init(cb_worker_max);
        exp_tile_init<EXP_APPROX_MODE>();
        copy_tile(cb_prev_max, i, dst_reg_0);
        copy_tile(cb_worker_max, i, dst_reg_1);
        copy_tile(cb_prev_sum, i, dst_reg_3);
        copy_tile(cb_worker_sum, i, worker_sum_dst);
        MATH((fused_max_sub_exp_add_tile<vector_mode>(0, scale_bf16)));
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(dst_reg_0, cb_exp_max_diff);
        pack_tile(dst_reg_1, cb_exp_max_diff_2);
        pack_tile(dst_reg_2, cb_cur_max);
        pack_tile(dst_reg_3, cb_cur_sum);
        tile_regs_release();
        cb_cur_max_obj.push_back(1);
        cb_cur_sum_obj.push_back(1);
        cb_exp_max_diff_obj.push_back(1);
        cb_exp_max_diff_2_obj.push_back(1);
    }
    cb_prev_sum_obj.pop_front(num_head_tiles);
    cb_worker_sum_obj.pop_front(num_head_tiles);
}

template <bool pop_in_cb>
void move_block(uint32_t in_cb, uint32_t out_cb, uint32_t num_tiles) {
    CircularBuffer cb_in(in_cb);
    CircularBuffer cb_out(out_cb);

    copy_init(in_cb);

    cb_in.wait_front(num_tiles);
    cb_out.reserve_back(num_tiles);

#pragma GCC unroll 0
    for (uint32_t i = 0; i < num_tiles; i++) {
        tile_regs_acquire();
        copy_tile(in_cb, i, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, out_cb);
        tile_regs_release();
        cb_out.push_back(1);
    }
    if (pop_in_cb) {
        cb_in.pop_front(num_tiles);
    }
}

void copy_block(uint32_t in_cb, uint32_t out_cb, uint32_t num_tiles) {
    CircularBuffer cb_in(in_cb);
    CircularBuffer cb_out(out_cb);
    copy_init(in_cb);
    cb_in.wait_front(num_tiles);
    cb_out.reserve_back(num_tiles);
#pragma GCC unroll 0
    for (uint32_t i = 0; i < num_tiles; i++) {
        tile_regs_acquire();
        copy_tile(in_cb, i, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, out_cb);
        tile_regs_release();
        cb_out.push_back(1);
    }
    cb_in.pop_front(num_tiles);
}

void log_block(uint32_t in_cb, uint32_t out_cb, uint32_t num_tiles) {
    reconfig_data_format_srca(in_cb);
    pack_reconfig_data_format(out_cb);
    CircularBuffer cb_in(in_cb);
    CircularBuffer cb_out(out_cb);
    copy_init(in_cb);
    log_tile_init();
    cb_in.wait_front(num_tiles);
    cb_out.reserve_back(num_tiles);

    for (uint32_t i = 0; i < num_tiles; i++) {
        tile_regs_acquire();
        copy_tile(in_cb, i, 0);
        log_tile(0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, out_cb);
        tile_regs_release();
        cb_out.push_back(1);
    }
}

void sigmoid_sub(uint32_t in0_cb, uint32_t in1_cb, uint32_t out_cb, uint32_t num_tiles) {
    CircularBuffer cb_in0(in0_cb);
    CircularBuffer cb_in1(in1_cb);
    CircularBuffer cb_out(out_cb);
    cb_in0.wait_front(num_tiles);
    cb_in1.wait_front(num_tiles);
    cb_out.reserve_back(num_tiles);
    sub_init(in0_cb, in1_cb);
    exp_tile_init<false>();
    MATH((ckernel::sfpu::sfpu_reciprocal_init<false>()));

    for (uint32_t i = 0; i < num_tiles; i++) {
        tile_regs_acquire();
        sub_tiles(in0_cb, in1_cb, i, i, 0);
        MATH((exp_tile_first_column<false, (uint16_t)0xBF80>(0)));
        MATH(SFPU_UNARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_binop_with_scalar,
            (APPROX, ADD_UNARY, 8, DST_ACCUM_MODE),
            0,
            VectorMode::C,
            0x3F800000));
        MATH((recip_tile_first_column(0)));
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, out_cb);
        tile_regs_release();
    }
    cb_out.push_back(num_tiles);
}

#ifdef TRISC_MATH
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
void softplus_tile_first_column(uint32_t idst, uint beta, uint beta_reciprocal, uint threshold) {
    SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        is_fp32_dest_acc_en,
        calculate_softplus_first_column,
        (is_fp32_dest_acc_en),
        idst,
        VectorMode::C,
        beta,
        beta_reciprocal,
        threshold);
}
#endif

void logsigmoid_sub(uint32_t in0_cb, uint32_t in1_cb, uint32_t out_cb, uint32_t num_tiles) {
    CircularBuffer cb_in0(in0_cb);
    CircularBuffer cb_in1(in1_cb);
    CircularBuffer cb_out(out_cb);
    cb_in0.wait_front(num_tiles);
    cb_in1.wait_front(num_tiles);
    cb_out.reserve_back(num_tiles);
    sub_init(in0_cb, in1_cb);
    softplus_tile_init();
    constexpr uint32_t const_1_fp32 = 0x3F800000;
    constexpr uint32_t const_20_fp32 = 0x41A00000;

    for (uint32_t i = 0; i < num_tiles; i++) {
        tile_regs_acquire();
        sub_tiles(in1_cb, in0_cb, i, i, 0);

        MATH((softplus_tile_first_column(0, const_1_fp32, const_1_fp32, const_20_fp32)));
        negative_tile(0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, out_cb);
        tile_regs_release();
    }
    cb_out.push_back(num_tiles);
}

__attribute__((optimize("Os"))) void sub_block(uint32_t in0_cb, uint32_t in1_cb, uint32_t out_cb, uint32_t num_tiles) {
    CircularBuffer cb_in0(in0_cb);
    CircularBuffer cb_in1(in1_cb);
    CircularBuffer cb_out(out_cb);
    cb_in0.wait_front(num_tiles);
    cb_in1.wait_front(num_tiles);
    cb_out.reserve_back(num_tiles);
    sub_init(in0_cb, in1_cb);

    for (uint32_t i = 0; i < num_tiles; i++) {
        tile_regs_acquire();
        sub_tiles(in0_cb, in1_cb, i, i, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, out_cb);
        tile_regs_release();
    }
    cb_out.push_back(num_tiles);
}

ALWI void matmul_blocks(
    const uint32_t& in0_cb,
    const uint32_t& in1_cb,
    const uint32_t& out_cb,
    const uint32_t& M,
    const uint32_t& N,
    const uint32_t& K,
    const uint32_t& in0_num_subblocks,
    const uint32_t& in1_num_subblocks,
    const uint32_t& in0_block_w,
    const uint32_t& subblock_h,
    const uint32_t& subblock_w,
    const bool& transpose,
    const bool& add_mask = false,
    const uint32_t& mask_cb = 0,
    const uint32_t& zero_cb = 0) {
    CircularBuffer cb_in0(in0_cb);
    CircularBuffer cb_in1(in1_cb);
    CircularBuffer cb_out(out_cb);
    CircularBuffer cb_mask(mask_cb);
    CircularBuffer cb_zero(zero_cb);

    matmul_block_init(in0_cb, in1_cb, transpose, subblock_w, subblock_h, in0_block_w);

    const uint32_t output_num_tiles = M * N;
    const uint32_t out_subblock_num_tiles = subblock_h * subblock_w;
    const uint32_t in0_subblock_all_cols_num_tiles = subblock_h * N;

    uint32_t in0_index_offset = 0;

    const uint32_t in0_subblock_num_tiles = subblock_h * in0_block_w;
    uint32_t in0_wait_tiles = in0_subblock_num_tiles;

    reconfig_data_format(in1_cb, in0_cb);
    cb_in1.wait_front(K * N);
    cb_out.reserve_back(output_num_tiles);

    for (uint32_t in0_subblock = 0; in0_subblock < in0_num_subblocks; ++in0_subblock) {
        cb_in0.wait_front(in0_wait_tiles);
        uint32_t in1_index_offset = 0;
        for (uint32_t in1_subblock = 0; in1_subblock < in1_num_subblocks; ++in1_subblock) {
            tile_regs_acquire();

            uint32_t dst_index = 0;
            uint32_t in0_index = in0_index_offset;
            uint32_t in1_index = in1_index_offset;

            for (uint32_t inner_dim = 0; inner_dim < in0_block_w; inner_dim++) {
                matmul_block(
                    in0_cb, in1_cb, in0_index, in1_index, dst_index, transpose, subblock_w, subblock_h, in0_block_w);
                in0_index++;
                in1_index += N;
            }
            if (add_mask) {
                cb_mask.wait_front(out_subblock_num_tiles);
                cb_zero.wait_front(1);
                reconfig_data_format(zero_cb, mask_cb);
                add_init(zero_cb, mask_cb, true);
                for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {
                    add_tiles(zero_cb, mask_cb, 0, i, i);
                }
                reconfig_data_format(in1_cb, in0_cb);
                matmul_block_init(in0_cb, in1_cb, transpose, subblock_w, subblock_h, in0_block_w);
            }
            tile_regs_commit();
            tile_regs_wait();
            uint32_t dst_idx = 0;
            uint32_t out_col_offset = in1_subblock * subblock_w;
            for (uint32_t r = 0; r < subblock_h; r++) {
                uint32_t out_row_offset = r * N;
                for (uint32_t c = 0; c < subblock_w; c++) {
                    pack_tile<true>(dst_idx, out_cb, out_row_offset + out_col_offset + c);
                    dst_idx++;
                }
            }
            tile_regs_release();
            in1_index_offset += subblock_w;
        }
        in0_index_offset += subblock_h * in0_block_w;
        in0_wait_tiles += in0_subblock_num_tiles;
        cb_out.push_back(in0_subblock_all_cols_num_tiles);
    }
    cb_in1.pop_front(K * N);
}

template <uint32_t M>
void matmul_reduce(uint32_t in1_cb, const uint32_t& out_cb) {
    CircularBuffer cb_in1(in1_cb);
    CircularBuffer cb_out(out_cb);

    constexpr uint32_t N = 1;
    constexpr uint32_t in0_block_w = N;
    constexpr uint32_t subblock_w = N;
#ifdef STATS_GRANULARITY
    constexpr uint32_t subblock_h = STATS_GRANULARITY;
    constexpr uint32_t in0_num_subblocks = M / STATS_GRANULARITY;
#else
    constexpr uint32_t subblock_h = 1;
    constexpr uint32_t in0_num_subblocks = M;
#endif

    reconfig_data_format(in1_cb, out_cb);
    matmul_block_init(out_cb, in1_cb, 0, subblock_w, subblock_h, in0_block_w);

    pack_reconfig_data_format(out_cb);
    cb_in1.wait_front(N);
    cb_out.wait_front(M);

    for (uint32_t in0_subblock = 0; in0_subblock < in0_num_subblocks; ++in0_subblock) {
        tile_regs_acquire();

        uint32_t dst_index = 0;
        uint32_t in0_index = 0;
        uint32_t in1_index = 0;

        matmul_block(out_cb, in1_cb, in0_index, in1_index, dst_index, 0, subblock_w, subblock_h, in0_block_w);

        tile_regs_commit();
        cb_out.pop_front(subblock_h);

        tile_regs_wait();
        for (uint32_t i = 0; i < subblock_h; i++) {
            pack_tile(i, out_cb);
        }
        tile_regs_release();
        cb_out.push_back(subblock_h);
    }
}

template <uint32_t dst_batch>
void stamp_tile_range_l1_acc(
    uint32_t src_cb, uint32_t src_tile_idx, uint32_t out_cb, uint32_t out_offset, uint32_t count) {
    for (uint32_t base = 0; base < count; base += dst_batch) {
        uint32_t batch = (count - base < dst_batch) ? (count - base) : dst_batch;
        tile_regs_acquire();
        for (uint32_t i = 0; i < batch; i++) {
            copy_tile(src_cb, src_tile_idx, i);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t i = 0; i < batch; i++) {
            pack_tile<true>(i, out_cb, out_offset + base + i);
        }
        tile_regs_release();
    }
}

template <uint32_t dst_batch>
void apply_padded_mask_lightweight_runtime(
    uint32_t neginf_cb,
    uint32_t neginf_tile_idx,
    uint32_t out_cb,
    uint32_t num_padded,
    uint32_t num_cols,
    uint32_t num_rows,
    uint32_t row_base = 0) {
    uint32_t start = num_cols - num_padded;

    reconfig_data_format_srca(neginf_cb);
    pack_reconfig_data_format(out_cb);
    copy_init(neginf_cb);
    PACK((llk_pack_reconfig_l1_acc(1)));

    for (uint32_t row = 0; row < num_rows; row++) {
        stamp_tile_range_l1_acc<dst_batch>(
            neginf_cb, neginf_tile_idx, out_cb, (row_base + row) * num_cols + start, num_padded);
    }

    PACK((llk_pack_reconfig_l1_acc(0)));
}

void apply_partial_mask_lightweight(
    uint32_t mask_cb,
    uint32_t partial_tile_idx,
    uint32_t out_cb,
    uint32_t boundary_col,
    uint32_t num_cols,
    uint32_t num_rows,
    uint32_t row_base = 0) {
    reconfig_data_format_srca(mask_cb);
    pack_reconfig_data_format(out_cb);
    copy_init(mask_cb);
    PACK((llk_pack_reconfig_l1_acc(1)));

    for (uint32_t row = 0; row < num_rows; row++) {
        tile_regs_acquire();
        copy_tile(mask_cb, partial_tile_idx, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile<true>(0, out_cb, (row_base + row) * num_cols + boundary_col);
        tile_regs_release();
    }

    PACK((llk_pack_reconfig_l1_acc(0)));
}

template <uint32_t dst_batch>
void apply_causal_mask_lightweight(
    uint32_t mask_cb,
    uint32_t neginf_idx,
    uint32_t diag_idx,
    uint32_t out_cb,
    uint32_t q_start_tile,
    uint32_t k_start_tile,
    uint32_t num_rows,
    uint32_t num_cols,
    uint32_t straddle_col = 0,
    uint32_t straddle_jump = 0) {
    reconfig_data_format_srca(mask_cb);
    pack_reconfig_data_format(out_cb);
    copy_init(mask_cb);
    PACK((llk_pack_reconfig_l1_acc(1)));

    for (uint32_t row = 0; row < num_rows; row++) {
        uint32_t row_offset = row * num_cols;
        const int32_t q_pos = static_cast<int32_t>(q_start_tile + row);

        if (straddle_col == 0) {
            int32_t diag_col = q_pos - static_cast<int32_t>(k_start_tile);
            if (diag_col < 0) {
                stamp_tile_range_l1_acc<dst_batch>(mask_cb, neginf_idx, out_cb, row_offset, num_cols);
            } else if (static_cast<uint32_t>(diag_col) < num_cols) {
                tile_regs_acquire();
                copy_tile(mask_cb, diag_idx, 0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile<true>(0, out_cb, row_offset + static_cast<uint32_t>(diag_col));
                tile_regs_release();

                uint32_t neginf_start = static_cast<uint32_t>(diag_col) + 1;
                if (neginf_start < num_cols) {
                    stamp_tile_range_l1_acc<dst_batch>(
                        mask_cb, neginf_idx, out_cb, row_offset + neginf_start, num_cols - neginf_start);
                }
            }
        } else {
            for (uint32_t col = 0; col < num_cols; col++) {
                int32_t k_pos = static_cast<int32_t>(k_start_tile) + static_cast<int32_t>(col);
                if (col >= straddle_col) {
                    k_pos += static_cast<int32_t>(straddle_jump);
                }
                if (k_pos > q_pos) {
                    stamp_tile_range_l1_acc<dst_batch>(mask_cb, neginf_idx, out_cb, row_offset + col, 1);
                } else if (k_pos == q_pos) {
                    tile_regs_acquire();
                    copy_tile(mask_cb, diag_idx, 0);
                    tile_regs_commit();
                    tile_regs_wait();
                    pack_tile<true>(0, out_cb, row_offset + col);
                    tile_regs_release();
                }
            }
        }
    }

    PACK((llk_pack_reconfig_l1_acc(0)));
}

struct LightweightMaskContext {
    bool is_causal = false;
    uint32_t neginf_tile_idx = 0;
    uint32_t causal_diag_tile_idx = 0;
    uint32_t primary_diag_tile_idx = 0;
    uint32_t sliding_leading_prev_tile_idx = 0;
    uint32_t sliding_leading_tile_idx = 0;
    uint32_t sliding_trailing_next_tile_idx = 0;
    uint32_t global_n_padded_tiles = 0;
    uint32_t local_n_padded_tiles = 0;
    uint32_t joint_n_padded_tiles = 0;
    uint32_t global_n_partial_col = 0;
    uint32_t joint_l_partial_col = 0;
    uint32_t global_n_partial_tile_idx = 0;
    uint32_t joint_l_partial_tile_idx = 0;
    uint32_t straddle_num_padded_tiles = 0;
    uint32_t straddle_mask_chunk_id = 0;

    void resolve_for_chunk(
        uint32_t Sk_chunk_t,
        uint32_t k_chunk,
        uint32_t num_local_k_chunks,
        bool ring_iter_needs_global_n_mask,
        bool ring_iter_needs_joint_n_mask,
        bool local_n_needs_masking,
        uint32_t global_n_mask_chunk_id,
        uint32_t local_n_mask_chunk_id,
        uint32_t joint_n_mask_chunk_id,
        uint32_t& out_num_padded,
        uint32_t& out_boundary_col,
        uint32_t& out_partial_tile_idx,
        bool& out_has_partial) const {
        out_num_padded = 0;
        out_has_partial = false;
        out_boundary_col = 0;
        out_partial_tile_idx = 0;

        if (ring_iter_needs_global_n_mask && k_chunk == global_n_mask_chunk_id) {
            out_num_padded = global_n_padded_tiles;
            out_has_partial = (global_n_partial_col > 0);
            out_boundary_col = Sk_chunk_t - out_num_padded - (out_has_partial ? 1 : 0);
            out_partial_tile_idx = global_n_partial_tile_idx;
        } else if (local_n_needs_masking && k_chunk == local_n_mask_chunk_id) {
            out_num_padded = local_n_padded_tiles;
        } else if (ring_iter_needs_joint_n_mask && (k_chunk - num_local_k_chunks) == joint_n_mask_chunk_id) {
            out_num_padded = joint_n_padded_tiles;
            out_has_partial = (joint_l_partial_col > 0);
            out_boundary_col = Sk_chunk_t - out_num_padded - (out_has_partial ? 1 : 0);
            out_partial_tile_idx = joint_l_partial_tile_idx;
        } else if (straddle_num_padded_tiles > 0 && k_chunk == straddle_mask_chunk_id) {
            out_num_padded = straddle_num_padded_tiles;
        }
    }

    template <uint32_t dst_size>
    void apply(
        uint32_t cb_mask_in,
        uint32_t cb_qk_im,
        uint32_t Sk_chunk_t,
        uint32_t Sq_chunk_t,
        uint32_t k_chunk,
        uint32_t num_local_k_chunks,
        bool ring_iter_needs_global_n_mask,
        bool ring_iter_needs_joint_n_mask,
        bool local_n_needs_masking,
        uint32_t global_n_mask_chunk_id,
        uint32_t local_n_mask_chunk_id,
        uint32_t joint_n_mask_chunk_id,
        uint32_t q_start_tile,
        uint32_t k_start_tile,
        uint32_t straddle_col = 0,
        uint32_t straddle_jump = 0) const {
        if (is_causal) {
            apply_causal_mask_lightweight<dst_size>(
                cb_mask_in,
                neginf_tile_idx,
                causal_diag_tile_idx,
                cb_qk_im,
                q_start_tile,
                k_start_tile,
                Sq_chunk_t,
                Sk_chunk_t,
                straddle_col,
                straddle_jump);
        }

        uint32_t num_padded, boundary_col, partial_tile_idx;
        bool has_partial;
        resolve_for_chunk(
            Sk_chunk_t,
            k_chunk,
            num_local_k_chunks,
            ring_iter_needs_global_n_mask,
            ring_iter_needs_joint_n_mask,
            local_n_needs_masking,
            global_n_mask_chunk_id,
            local_n_mask_chunk_id,
            joint_n_mask_chunk_id,
            num_padded,
            boundary_col,
            partial_tile_idx,
            has_partial);

        if (has_partial) {
            apply_partial_mask_lightweight(
                cb_mask_in, partial_tile_idx, cb_qk_im, boundary_col, Sk_chunk_t, Sq_chunk_t);
        }
        if (num_padded > 0) {
            apply_padded_mask_lightweight_runtime<dst_size>(
                cb_mask_in, neginf_tile_idx, cb_qk_im, num_padded, Sk_chunk_t, Sq_chunk_t);
        }
    }
};

enum SDPAType {
    STANDARD = 0,
    JOINT = 1,
    RING = 2,
};

template <
    SDPAType sdpa_type,
    uint32_t cb_qk_im,
    uint32_t cb_identity_scale_in,
    uint32_t cb_attention_sink,
    uint32_t cb_scale_in,
    uint32_t Sq_chunk_t,
    uint32_t Sk_chunk_t,
    uint32_t NH,
    uint32_t DHt,
    uint32_t vDHt,
    bool use_attention_sink,
    bool use_provided_mask,
    bool use_padded_mask,
    bool use_joint_mask,
    bool is_chunked,
    uint32_t scale_fp32,
    uint32_t sliding_window_size,
    bool lightweight_mask_enabled = false,
    bool chunked_enabled = false,
    uint32_t chunked_q_local_padded_Nt = 0,
    uint32_t chunked_chunk_size_t = 0,
    bool use_windowed_narrowing = false,
    uint32_t cb_windowed_k_range = 0,
    bool padded_q_slots_enabled = true>
void sdpa_inner_loop(
    const uint32_t Skt,
    const uint32_t qk_in0_block_w,
    const uint32_t qk_subblock_w,
    const uint32_t qk_subblock_h,
    const uint32_t qk_in0_num_subblocks,
    const uint32_t qk_in1_num_subblocks,
    const uint32_t out_in0_block_w,
    const uint32_t out_subblock_w,
    const uint32_t out_subblock_h,
    const uint32_t out_in0_num_subblocks,
    const uint32_t out_in1_num_subblocks,
    const uint32_t iter_q_start,
    const uint32_t iter_q_end,
    const uint32_t q_num_chunks,
    const uint32_t local_q_start,
    const uint32_t chunked_q_chunk_offset,
    const uint32_t iter_k_chunk_start,
    const uint32_t iter_k_chunk_end,
    const uint32_t q_chunk_tiles,
    const uint32_t k_chunk_tiles,
    const uint32_t v_chunk_tiles,
    const uint32_t qk_chunk_tiles,
    const uint32_t out_chunk_tiles,
    const uint32_t mask_chunk_0,
    const uint32_t mask_chunk_1,
    const uint32_t ring_iter,
    const uint32_t ring_id,
    const uint32_t num_local_k_chunks,
    const uint32_t local_padded_Nt,
    const uint32_t logical_nt,
    const bool ring_iter_needs_global_n_mask,
    const bool ring_iter_needs_joint_n_mask,
    const bool local_n_needs_masking,
    const uint32_t global_n_mask_chunk_id,
    const uint32_t local_n_mask_chunk_id,
    const uint32_t joint_n_mask_chunk_id,
    const uint32_t cb_q_in,
    const uint32_t cb_k_in,
    const uint32_t cb_v_in,
    const uint32_t cb_mask_in,
    const uint32_t cb_col_identity,
    const uint32_t cb_out_im_A,
    const uint32_t cb_out_im_B,
    const uint32_t cb_max_A,
    const uint32_t cb_max_B,
    const uint32_t cb_sum_A,
    const uint32_t cb_sum_B,
    const uint32_t cb_exp_max_diff,
    const uint32_t cb_lse_in,
    const uint32_t cb_lse_out,
    const uint32_t cb_prev_out,
    const uint32_t cb_out,
    const LightweightMaskContext& lw_mask = {},
    const bool is_causal = false,
    const bool is_balanced = false,
    const bool use_zigzag_balancing = false,
    const bool is_last_ring_iter = true,
    const ChunkedContext& chunked = {},
    [[maybe_unused]] const uint32_t padded_q_slots = 0) {
    CircularBuffer cb_q_in_obj(cb_q_in);
    CircularBuffer cb_k_in_obj(cb_k_in);
    CircularBuffer cb_v_in_obj(cb_v_in);
    CircularBuffer cb_qk_im_obj(cb_qk_im);
    CircularBuffer cb_attention_sink_obj(cb_attention_sink);
    CircularBuffer cb_lse_in_obj(cb_lse_in);
    CircularBuffer cb_prev_out_obj(cb_prev_out);
    constexpr uint32_t dst_size = compute_kernel_lib::DEST_AUTO_LIMIT;
    uint32_t KV_chunks_processed_in_iter = 0;
    const uint32_t q_per_core = iter_q_end - iter_q_start;
    auto ring_kv_global_start_tile = [&](uint32_t k_chunk) -> uint32_t {
        if constexpr (chunked_enabled) {
            return kv_global_tile_for_local<true, 0, chunked_chunk_size_t, chunked_q_local_padded_Nt>(
                ring_id, k_chunk * Sk_chunk_t);
        } else {
            return local_padded_Nt * ring_id + k_chunk * Sk_chunk_t;
        }
    };

    for (uint32_t q_iter = iter_q_start; q_iter < iter_q_end; ++q_iter) {
        uint32_t q_start_tile = 0;
        uint32_t q_high_tile = 0;
        uint32_t causal_k_limit = 0;
        if constexpr (sdpa_type == STANDARD) {
            const uint32_t linear_q_chunk = local_q_start + (q_iter - iter_q_start);
            uint32_t q_chunk = remap_q_index(linear_q_chunk, q_num_chunks, use_zigzag_balancing) % q_num_chunks;
            if constexpr (is_chunked) {
                q_chunk = chunked_q_chunk_offset + q_chunk;
            }
            q_start_tile = q_chunk * Sq_chunk_t;
            if (is_causal) {
                const uint32_t q_high_unclamped = q_start_tile + Sq_chunk_t;
                q_high_tile = q_high_unclamped < Skt ? q_high_unclamped : Skt;
            } else {
                q_high_tile = Skt;
            }
        } else if (sdpa_type == RING) {
            uint32_t q_chunk = remap_q_index(q_iter, q_num_chunks, use_zigzag_balancing) % q_num_chunks;

            if constexpr (chunked_enabled) {
                q_start_tile =
                    chunked.q_start_idx_t + chunked.ring_index * chunked_q_local_padded_Nt + q_chunk * Sq_chunk_t;
            } else if (is_causal) {
                q_start_tile = q_chunk * Sq_chunk_t;
                causal_k_limit = (q_start_tile + Sq_chunk_t + Sk_chunk_t - 1) / Sk_chunk_t;
            }
            if (is_balanced && (q_chunk < q_num_chunks / 2)) {
                continue;
            }
        }

        uint32_t alias_prev_sum = cb_sum_A;
        uint32_t alias_cur_sum = cb_sum_B;
        uint32_t alias_prev_max = cb_max_A;
        uint32_t alias_cur_max = cb_max_B;
        uint32_t alias_mm2_prev_out = cb_out_im_A;
        uint32_t alias_mm2_cur_out = cb_out_im_B;

        uint32_t k_chunk_end;
        if constexpr (sdpa_type == STANDARD) {
            k_chunk_end = (q_high_tile + Sk_chunk_t - 1) / Sk_chunk_t;
        } else {
            k_chunk_end = iter_k_chunk_end;
        }

        uint32_t k_chunk_start = iter_k_chunk_start;
        if constexpr (use_windowed_narrowing) {
            CircularBuffer cb_k_range_obj(cb_windowed_k_range);
            cb_k_range_obj.wait_front(1);
            k_chunk_start = ckernel::read_tile_value(cb_windowed_k_range, 0, 0);
            k_chunk_end = ckernel::read_tile_value(cb_windowed_k_range, 0, 1);
            cb_k_range_obj.pop_front(1);
        }

        uint32_t processed_k_chunks = 0;

        for (uint32_t k_chunk = k_chunk_start; k_chunk < k_chunk_end; ++k_chunk) {
            uint32_t kv_global_start_tile = 0;
            if constexpr (sdpa_type == RING) {
                const bool kv_chunk_is_joint = k_chunk >= num_local_k_chunks;
                kv_global_start_tile = ring_kv_global_start_tile(k_chunk);
                if (!kv_chunk_is_joint && (kv_global_start_tile >= logical_nt)) {
                    continue;
                }
            }

            KV_chunks_processed_in_iter++;

            if (sdpa_type == RING && !chunked_enabled && k_chunk >= causal_k_limit && is_causal) {
                cb_k_in_obj.wait_front(k_chunk_tiles);
                cb_v_in_obj.wait_front(v_chunk_tiles);
                cb_k_in_obj.pop_front(k_chunk_tiles);
                cb_v_in_obj.pop_front(v_chunk_tiles);

                continue;
            }

            const bool qk_fused = (sdpa_type == STANDARD) && !use_provided_mask && !use_padded_mask &&
                                  (sliding_window_size == 0) &&
                                  !(is_causal && (q_start_tile < (k_chunk + 1) * Sk_chunk_t));
            if (qk_fused) {
                sdpa_qk_max_sub_exp<cb_qk_im, cb_identity_scale_in, Sq_chunk_t, Sk_chunk_t, DHt, scale_fp32>(
                    cb_q_in, cb_k_in, alias_cur_max, alias_prev_max, alias_cur_sum, processed_k_chunks > 0);
            } else {
                reconfig_data_format(cb_k_in, cb_q_in);
                pack_reconfig_data_format(cb_qk_im);
                matmul_blocks(
                    cb_q_in,
                    cb_k_in,
                    cb_qk_im,
                    Sq_chunk_t,
                    Sk_chunk_t,
                    DHt,
                    qk_in0_num_subblocks,
                    qk_in1_num_subblocks,
                    qk_in0_block_w,
                    qk_subblock_h,
                    qk_subblock_w,
                    true);

                bool apply_mask = false;
                bool needs_padding_mask =
                    (sdpa_type == RING) &&
                    ((ring_iter_needs_global_n_mask && k_chunk == global_n_mask_chunk_id) ||
                     (local_n_needs_masking && k_chunk == local_n_mask_chunk_id) ||
                     (ring_iter_needs_joint_n_mask && (k_chunk - num_local_k_chunks) == joint_n_mask_chunk_id) ||
                     (lw_mask.straddle_num_padded_tiles > 0 && k_chunk == lw_mask.straddle_mask_chunk_id));
                if (sdpa_type == RING && !is_causal) {
                    apply_mask = needs_padding_mask;
                } else if (is_causal || sliding_window_size > 0) {
                    uint32_t k_low_idx;
                    uint32_t k_high_idx;
                    if constexpr (sdpa_type == RING && chunked_enabled) {
                        k_low_idx = kv_global_start_tile;
                        k_high_idx = k_low_idx + Sk_chunk_t;
                        if (chunked_q_local_padded_Nt > 0) {
                            const uint32_t local_start = k_chunk * Sk_chunk_t;
                            const uint32_t slab_end_local =
                                (local_start / chunked_q_local_padded_Nt + 1) * chunked_q_local_padded_Nt;
                            if (local_start + Sk_chunk_t > slab_end_local) {
                                k_high_idx += (chunked_chunk_size_t - chunked_q_local_padded_Nt);
                            }
                        }
                    } else {
                        k_low_idx = k_chunk * Sk_chunk_t;
                        k_high_idx = k_low_idx + Sk_chunk_t;
                    }
                    apply_mask = (q_start_tile < k_high_idx) || (sliding_window_size > 0) || needs_padding_mask;
                } else if constexpr (use_provided_mask) {
                    apply_mask = true;
                } else if constexpr (use_padded_mask) {
                    apply_mask = (k_chunk == iter_k_chunk_end - 1);
                } else if constexpr (use_joint_mask) {
                    apply_mask = (k_chunk == mask_chunk_0) || (k_chunk == mask_chunk_1);
                }

                if (apply_mask) {
                    reconfig_data_format(cb_qk_im, cb_mask_in);
                    if constexpr (lightweight_mask_enabled) {
                        cb_qk_im_obj.wait_front(Sk_chunk_t * Sq_chunk_t);
                        cb_qk_im_obj.pop_front(Sk_chunk_t * Sq_chunk_t);
                        cb_qk_im_obj.reserve_back(Sk_chunk_t * Sq_chunk_t);
                        uint32_t k_start_tile_for_mask;
                        uint32_t lw_straddle_col = 0;
                        uint32_t lw_straddle_jump = 0;
                        if constexpr (sdpa_type == RING && chunked_enabled) {
                            k_start_tile_for_mask = kv_global_start_tile;
                            if (chunked_q_local_padded_Nt > 0) {
                                const uint32_t local_start = k_chunk * Sk_chunk_t;
                                const uint32_t slab_end_local =
                                    (local_start / chunked_q_local_padded_Nt + 1) * chunked_q_local_padded_Nt;
                                if (local_start + Sk_chunk_t > slab_end_local) {
                                    lw_straddle_col = slab_end_local - local_start;
                                    lw_straddle_jump = chunked_chunk_size_t - chunked_q_local_padded_Nt;
                                }
                            }
                        } else {
                            k_start_tile_for_mask = k_chunk * Sk_chunk_t;
                        }
                        lw_mask.template apply<dst_size>(
                            cb_mask_in,
                            cb_qk_im,
                            Sk_chunk_t,
                            Sq_chunk_t,
                            k_chunk,
                            num_local_k_chunks,
                            ring_iter_needs_global_n_mask,
                            ring_iter_needs_joint_n_mask,
                            local_n_needs_masking,
                            global_n_mask_chunk_id,
                            local_n_mask_chunk_id,
                            joint_n_mask_chunk_id,
                            q_start_tile,
                            k_start_tile_for_mask,
                            lw_straddle_col,
                            lw_straddle_jump);
                        cb_qk_im_obj.push_back(Sk_chunk_t * Sq_chunk_t);
                    } else {
                        add_block_inplace(cb_qk_im, cb_mask_in, qk_chunk_tiles);
                    }
                }

                reconfig_data_format(cb_qk_im, cb_identity_scale_in);
                reduce_c<PoolType::MAX, ReduceDim::REDUCE_ROW, cb_qk_im, cb_identity_scale_in, Sq_chunk_t>(
                    alias_cur_max, alias_prev_max, Sk_chunk_t, processed_k_chunks > 0);

                sub_exp_block_bcast_cols_inplace<cb_qk_im, Sq_chunk_t, scale_fp32, true>(
                    alias_cur_max, alias_cur_sum, Sk_chunk_t);
            }

            reconfig_data_format(cb_v_in, cb_qk_im);
            pack_reconfig_data_format(alias_mm2_cur_out);

            matmul_blocks(
                cb_qk_im,
                cb_v_in,
                alias_mm2_cur_out,
                Sq_chunk_t,
                vDHt,
                Sk_chunk_t,
                out_in0_num_subblocks,
                out_in1_num_subblocks,
                out_in0_block_w,
                out_subblock_h,
                out_subblock_w,
                false);

            cb_qk_im_obj.pop_front(qk_chunk_tiles);
            reconfig_data_format(alias_prev_max, alias_cur_max);

            if (processed_k_chunks > 0) {
                sub_exp_block<scale_fp32>(alias_prev_max, alias_cur_max, cb_exp_max_diff, Sq_chunk_t);
                CircularBuffer(alias_prev_max).pop_front(Sq_chunk_t);

                mul_tiles_bcast_cols_inplace(alias_prev_sum, cb_exp_max_diff, Sq_chunk_t);

                add_block_inplace(alias_cur_sum, alias_prev_sum, Sq_chunk_t);

                mul_block_bcast_cols<Sq_chunk_t, vDHt, false, true>(
                    alias_mm2_prev_out, cb_exp_max_diff, alias_mm2_cur_out);
            }

            std::swap(alias_prev_sum, alias_cur_sum);
            std::swap(alias_mm2_prev_out, alias_mm2_cur_out);
            std::swap(alias_prev_max, alias_cur_max);

            processed_k_chunks++;
        }

        matmul_reduce<Sq_chunk_t>(cb_col_identity, alias_prev_sum);

        if constexpr (use_attention_sink) {
            reconfig_data_format(cb_attention_sink, cb_identity_scale_in);

            reduce_c<PoolType::MAX, ReduceDim::REDUCE_ROW, cb_attention_sink, cb_identity_scale_in, Sq_chunk_t, 1>(
                alias_cur_max, alias_prev_max, true);

            sub_exp_block<scale_fp32>(alias_prev_max, alias_cur_max, cb_exp_max_diff, Sq_chunk_t);
            CircularBuffer(alias_prev_max).pop_front(Sq_chunk_t);

            mul_tiles_bcast_cols_inplace(alias_prev_sum, cb_exp_max_diff, Sq_chunk_t);
            sub_exp_block_bcast_cols_inplace<cb_attention_sink, Sq_chunk_t, scale_fp32, false>(
                alias_cur_max, alias_cur_sum, 1);

            add_block_inplace(alias_cur_sum, alias_prev_sum, Sq_chunk_t);

            std::swap(alias_prev_sum, alias_cur_sum);
            std::swap(alias_prev_max, alias_cur_max);

            mul_block_bcast_cols<Sq_chunk_t, vDHt, false, false>(
                alias_mm2_prev_out, cb_exp_max_diff, alias_mm2_cur_out);
            std::swap(alias_mm2_prev_out, alias_mm2_cur_out);
        }

        if constexpr (sdpa_type == RING) {
            log_block(alias_prev_sum, alias_cur_max, Sq_chunk_t);

            mul_block_bcast_scalar_inplace<cb_scale_in, Sq_chunk_t>(alias_prev_max);
            add_block_inplace(alias_prev_max, alias_cur_max, Sq_chunk_t);

            recip_block_inplace(alias_prev_sum, Sq_chunk_t);
            mul_block_bcast_cols_inplace<Sq_chunk_t, vDHt>(alias_mm2_prev_out, alias_prev_sum);

            if (ring_iter > 0) {
                cb_lse_in_obj.wait_front(Sq_chunk_t);
                cb_prev_out_obj.wait_front(out_chunk_tiles);

                uint32_t alias_cur_lse = alias_prev_max;
                uint32_t alias_sig = alias_cur_max;
                uint32_t alias_cur_out = alias_mm2_prev_out;
                uint32_t alias_sub = alias_mm2_cur_out;

                sigmoid_sub(alias_cur_lse, cb_lse_in, alias_sig, Sq_chunk_t);

                reconfig_data_format(cb_prev_out, alias_cur_out);
                sub_block(cb_prev_out, alias_cur_out, alias_sub, out_chunk_tiles);
                reconfig_data_format(alias_sub, alias_sig);
                mul_block_bcast_cols_inplace<Sq_chunk_t, vDHt>(alias_sub, alias_sig);
                reconfig_data_format(cb_prev_out, alias_sub);
                pack_reconfig_data_format(cb_out);
                sub_block(cb_prev_out, alias_sub, cb_out, out_chunk_tiles);
                cb_prev_out_obj.pop_front(out_chunk_tiles);
                CircularBuffer(alias_cur_out).pop_front(out_chunk_tiles);
                CircularBuffer(alias_sub).pop_front(out_chunk_tiles);

                pack_reconfig_data_format(alias_sig);
                reconfig_data_format(cb_lse_in, alias_cur_lse);
                logsigmoid_sub(cb_lse_in, alias_cur_lse, alias_sig, Sq_chunk_t);
                sub_block(cb_lse_in, alias_sig, cb_lse_out, Sq_chunk_t);
                CircularBuffer(alias_sig).pop_front(Sq_chunk_t);
                CircularBuffer(alias_cur_lse).pop_front(Sq_chunk_t);
                cb_lse_in_obj.pop_front(Sq_chunk_t);
            } else {
                pack_reconfig_data_format(cb_out);
                copy_block(alias_mm2_prev_out, cb_out, out_chunk_tiles);

                pack_reconfig_data_format(cb_lse_out);
                copy_block(alias_prev_max, cb_lse_out, Sq_chunk_t);
            }
        } else {
            recip_block_inplace(alias_prev_sum, Sq_chunk_t);

            pack_reconfig_data_format(cb_out);
            mul_block_bcast_cols<Sq_chunk_t, vDHt, false, false>(alias_mm2_prev_out, alias_prev_sum, cb_out);

            CircularBuffer(alias_prev_max).pop_front(Sq_chunk_t);
        }

        if (q_per_core > 1 || is_last_ring_iter) {
            cb_q_in_obj.pop_front(q_chunk_tiles);
        }

        if constexpr (use_attention_sink) {
            cb_attention_sink_obj.pop_front(Sq_chunk_t);
        }
    }

    if constexpr (sdpa_type == RING) {
        if constexpr (padded_q_slots_enabled) {
            for (uint32_t slot = 0; slot < padded_q_slots; ++slot) {
                const uint32_t q_flat = padded_slot_flat_q<false>({}, iter_q_start, q_per_core, q_per_core + slot, 1);
                const uint32_t q_chunk = remap_q_index(q_flat, q_num_chunks, use_zigzag_balancing) % q_num_chunks;
                if (is_balanced && (q_chunk < q_num_chunks / 2)) {
                    continue;
                }
                for (uint32_t k_chunk = iter_k_chunk_start; k_chunk < iter_k_chunk_end; ++k_chunk) {
                    if (k_chunk < num_local_k_chunks && ring_kv_global_start_tile(k_chunk) >= logical_nt) {
                        continue;
                    }
                    KV_chunks_processed_in_iter++;
                    cb_k_in_obj.wait_front(k_chunk_tiles);
                    cb_v_in_obj.wait_front(v_chunk_tiles);
                    cb_k_in_obj.pop_front(k_chunk_tiles);
                    cb_v_in_obj.pop_front(v_chunk_tiles);
                }
            }
        }
        if (KV_chunks_processed_in_iter % 2 == 0) {
            cb_k_in_obj.wait_front(k_chunk_tiles);
            cb_v_in_obj.wait_front(v_chunk_tiles);
            cb_k_in_obj.pop_front(k_chunk_tiles);
            cb_v_in_obj.pop_front(v_chunk_tiles);
        }
    }
}

template <
    uint32_t cb_qk_im,
    uint32_t cb_identity_scale_in,
    uint32_t cb_attention_sink,
    uint32_t Sq_chunk_t,
    uint32_t Sk_chunk_t,
    uint32_t DHt,
    uint32_t vDHt,
    bool use_attention_sink,
    bool is_causal,
    bool use_provided_mask,
    bool use_padded_mask,
    bool is_chunked,
    uint32_t scale_fp32,
    uint32_t sliding_window_size,
    bool lightweight_mask_enabled = false,
    bool use_windowed_narrowing = false,
    uint32_t cb_windowed_k_range = 0>
void sdpa_standard(
    const uint32_t Skt,
    const uint32_t qk_in0_block_w,
    const uint32_t qk_subblock_w,
    const uint32_t qk_subblock_h,
    const uint32_t qk_in0_num_subblocks,
    const uint32_t qk_in1_num_subblocks,
    const uint32_t out_in0_block_w,
    const uint32_t out_subblock_w,
    const uint32_t out_subblock_h,
    const uint32_t out_in0_num_subblocks,
    const uint32_t out_in1_num_subblocks,
    const uint32_t iter_q_start,
    const uint32_t iter_q_end,
    const uint32_t q_num_chunks,
    const uint32_t local_q_start,
    const uint32_t chunked_q_chunk_offset,
    const uint32_t k_num_chunks,
    const uint32_t q_chunk_tiles,
    const uint32_t k_chunk_tiles,
    const uint32_t v_chunk_tiles,
    const uint32_t qk_chunk_tiles,
    const uint32_t out_chunk_tiles,
    const uint32_t cb_q_in,
    const uint32_t cb_k_in,
    const uint32_t cb_v_in,
    const uint32_t cb_mask_in,
    const uint32_t cb_col_identity,
    const uint32_t cb_out_im_A,
    const uint32_t cb_out_im_B,
    const uint32_t cb_max_A,
    const uint32_t cb_max_B,
    const uint32_t cb_sum_A,
    const uint32_t cb_sum_B,
    const uint32_t cb_exp_max_diff,
    const uint32_t cb_out,
    const LightweightMaskContext& lw_mask = {},
    const bool use_zigzag_balancing = false) {
    sdpa_inner_loop<
        STANDARD,
        cb_qk_im,
        cb_identity_scale_in,
        cb_attention_sink,
        0,
        Sq_chunk_t,
        Sk_chunk_t,
        0,
        DHt,
        vDHt,
        use_attention_sink,
        use_provided_mask,
        use_padded_mask,
        false,
        is_chunked,
        scale_fp32,
        sliding_window_size,
        lightweight_mask_enabled,
        false,
        0,
        0,
        use_windowed_narrowing,
        cb_windowed_k_range>(
        Skt,
        qk_in0_block_w,
        qk_subblock_w,
        qk_subblock_h,
        qk_in0_num_subblocks,
        qk_in1_num_subblocks,
        out_in0_block_w,
        out_subblock_w,
        out_subblock_h,
        out_in0_num_subblocks,
        out_in1_num_subblocks,
        iter_q_start,
        iter_q_end,
        q_num_chunks,
        local_q_start,
        chunked_q_chunk_offset,
        0,
        k_num_chunks,
        q_chunk_tiles,
        k_chunk_tiles,
        v_chunk_tiles,
        qk_chunk_tiles,
        out_chunk_tiles,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        false,
        false,
        false,
        0,
        0,
        0,
        cb_q_in,
        cb_k_in,
        cb_v_in,
        cb_mask_in,
        cb_col_identity,
        cb_out_im_A,
        cb_out_im_B,
        cb_max_A,
        cb_max_B,
        cb_sum_A,
        cb_sum_B,
        cb_exp_max_diff,
        0,
        0,
        0,
        cb_out,
        lw_mask,
        is_causal,
        false,
        use_zigzag_balancing);
}

template <
    uint32_t cb_qk_im,
    uint32_t cb_identity_scale_in,
    uint32_t Sq_chunk_t,
    uint32_t Sk_chunk_t,
    uint32_t DHt,
    bool use_joint_mask,
    uint32_t scale_fp32>
void sdpa_joint(
    const uint32_t Skt,
    const uint32_t qk_in0_block_w,
    const uint32_t qk_subblock_w,
    const uint32_t qk_subblock_h,
    const uint32_t qk_in0_num_subblocks,
    const uint32_t qk_in1_num_subblocks,
    const uint32_t out_in0_block_w,
    const uint32_t out_subblock_w,
    const uint32_t out_subblock_h,
    const uint32_t out_in0_num_subblocks,
    const uint32_t out_in1_num_subblocks,
    const uint32_t local_q_start,
    const uint32_t local_q_end,
    const uint32_t k_num_chunks,
    const uint32_t q_chunk_tiles,
    const uint32_t k_chunk_tiles,
    const uint32_t qk_chunk_tiles,
    const uint32_t out_chunk_tiles,
    const uint32_t mask_chunk_0,
    const uint32_t mask_chunk_1,
    const uint32_t cb_q_in,
    const uint32_t cb_k_in,
    const uint32_t cb_v_in,
    const uint32_t cb_mask_in,
    const uint32_t cb_col_identity,
    const uint32_t cb_out_im_A,
    const uint32_t cb_out_im_B,
    const uint32_t cb_max_A,
    const uint32_t cb_max_B,
    const uint32_t cb_sum_A,
    const uint32_t cb_sum_B,
    const uint32_t cb_exp_max_diff,
    const uint32_t cb_out) {
    sdpa_inner_loop<
        JOINT,
        cb_qk_im,
        cb_identity_scale_in,
        0,
        0,
        Sq_chunk_t,
        Sk_chunk_t,
        0,
        DHt,
        DHt,
        false,
        false,
        false,
        use_joint_mask,
        false,
        scale_fp32,
        0>(
        Skt,
        qk_in0_block_w,
        qk_subblock_w,
        qk_subblock_h,
        qk_in0_num_subblocks,
        qk_in1_num_subblocks,
        out_in0_block_w,
        out_subblock_w,
        out_subblock_h,
        out_in0_num_subblocks,
        out_in1_num_subblocks,
        local_q_start,
        local_q_end,
        0,
        local_q_start,
        0,
        0,
        k_num_chunks,
        q_chunk_tiles,
        k_chunk_tiles,
        k_chunk_tiles,
        qk_chunk_tiles,
        out_chunk_tiles,
        mask_chunk_0,
        mask_chunk_1,
        0,
        0,
        0,
        0,
        0,
        false,
        false,
        false,
        0,
        0,
        0,
        cb_q_in,
        cb_k_in,
        cb_v_in,
        cb_mask_in,
        cb_col_identity,
        cb_out_im_A,
        cb_out_im_B,
        cb_max_A,
        cb_max_B,
        cb_sum_A,
        cb_sum_B,
        cb_exp_max_diff,
        0,
        0,
        0,
        cb_out);
}

template <
    uint32_t cb_qk_im,
    uint32_t cb_identity_scale_in,
    uint32_t cb_scale_in,
    uint32_t Sq_chunk_t,
    uint32_t Sk_chunk_t,
    uint32_t NH,
    uint32_t DHt,
    uint32_t vDHt,
    uint32_t scale_fp32,
    bool lightweight_mask_enabled = false,
    bool chunked_enabled = false,
    uint32_t chunked_q_local_padded_Nt = 0,
    uint32_t chunked_chunk_size_t = 0,
    bool padded_q_slots_enabled = true>
void sdpa_ring(
    const uint32_t qk_in0_block_w,
    const uint32_t qk_subblock_w,
    const uint32_t qk_subblock_h,
    const uint32_t qk_in0_num_subblocks,
    const uint32_t qk_in1_num_subblocks,
    const uint32_t out_in0_block_w,
    const uint32_t out_subblock_w,
    const uint32_t out_subblock_h,
    const uint32_t out_in0_num_subblocks,
    const uint32_t out_in1_num_subblocks,
    const uint32_t global_q_start,
    const uint32_t global_q_end,
    const uint32_t q_num_chunks,
    const uint32_t iter_k_chunk_start,
    const uint32_t iter_k_chunk_end,
    const uint32_t q_chunk_tiles,
    const uint32_t k_chunk_tiles,
    const uint32_t v_chunk_tiles,
    const uint32_t qk_chunk_tiles,
    const uint32_t out_chunk_tiles,
    const uint32_t ring_iter,
    const uint32_t ring_id,
    const uint32_t num_local_k_chunks,
    const uint32_t local_padded_Nt,
    const uint32_t logical_nt,
    const bool ring_iter_needs_global_n_mask,
    const bool ring_iter_needs_joint_n_mask,
    const bool local_n_needs_masking,
    const uint32_t global_n_mask_chunk_id,
    const uint32_t local_n_mask_chunk_id,
    const uint32_t joint_n_mask_chunk_id,
    const uint32_t cb_q_in,
    const uint32_t cb_k_in,
    const uint32_t cb_v_in,
    const uint32_t cb_mask_in,
    const uint32_t cb_col_identity,
    const uint32_t cb_out_im_A,
    const uint32_t cb_out_im_B,
    const uint32_t cb_max_A,
    const uint32_t cb_max_B,
    const uint32_t cb_sum_A,
    const uint32_t cb_sum_B,
    const uint32_t cb_exp_max_diff,
    const uint32_t cb_lse_in,
    const uint32_t cb_lse_out,
    const uint32_t cb_prev_out,
    const uint32_t cb_out,
    const LightweightMaskContext& lw_mask,
    const bool is_causal_ring_iter,
    const bool skip_first_half_q,
    const bool is_last_ring_iter,
    const bool use_zigzag_balancing = false,
    const ChunkedContext& chunked = {},
    const uint32_t padded_q_slots = 0) {
    sdpa_inner_loop<
        RING,
        cb_qk_im,
        cb_identity_scale_in,
        0,
        cb_scale_in,
        Sq_chunk_t,
        Sk_chunk_t,
        NH,
        DHt,
        vDHt,
        false,
        false,
        false,
        false,
        false,
        scale_fp32,
        0,
        lightweight_mask_enabled,
        chunked_enabled,
        chunked_q_local_padded_Nt,
        chunked_chunk_size_t,
        false,
        0,
        padded_q_slots_enabled>(
        0,
        qk_in0_block_w,
        qk_subblock_w,
        qk_subblock_h,
        qk_in0_num_subblocks,
        qk_in1_num_subblocks,
        out_in0_block_w,
        out_subblock_w,
        out_subblock_h,
        out_in0_num_subblocks,
        out_in1_num_subblocks,
        global_q_start,
        global_q_end,
        q_num_chunks,
        0,
        0,
        0,
        iter_k_chunk_end,
        q_chunk_tiles,
        k_chunk_tiles,
        v_chunk_tiles,
        qk_chunk_tiles,
        out_chunk_tiles,
        0,
        0,
        ring_iter,
        ring_id,
        num_local_k_chunks,
        local_padded_Nt,
        logical_nt,
        ring_iter_needs_global_n_mask,
        ring_iter_needs_joint_n_mask,
        local_n_needs_masking,
        global_n_mask_chunk_id,
        local_n_mask_chunk_id,
        joint_n_mask_chunk_id,
        cb_q_in,
        cb_k_in,
        cb_v_in,
        cb_mask_in,
        cb_col_identity,
        cb_out_im_A,
        cb_out_im_B,
        cb_max_A,
        cb_max_B,
        cb_sum_A,
        cb_sum_B,
        cb_exp_max_diff,
        cb_lse_in,
        cb_lse_out,
        cb_prev_out,
        cb_out,
        lw_mask,
        is_causal_ring_iter,
        skip_first_half_q,
        use_zigzag_balancing,
        is_last_ring_iter,
        chunked,
        padded_q_slots);
}
