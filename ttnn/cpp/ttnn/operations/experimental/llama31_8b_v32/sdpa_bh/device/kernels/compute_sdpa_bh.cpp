// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/circular_buffer.h"
#include "attn_steps.hpp"

namespace {

inline uint32_t q_slots[Q_UNIT_TABLE_SIZE];

constexpr uint32_t dest_tiles = 4;

#ifdef TRISC_MATH
template <uint32_t row0>
inline void scores_face_to_srca(uint32_t s_tile, uint32_t m_tile) {
    using namespace ckernel;
    math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(s_tile);
    TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::MATH | p_stall::SRCA_VLD | p_stall::SRCB_VLD);
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_AB);
    TTI_MOVD2B(0, p_movd2b::SRC_ROW16_OFFSET + 0, ADDR_MOD_7, p_movd2b::MOV_4_ROWS, row0 + 0);
    TTI_MOVD2B(0, p_movd2b::SRC_ROW16_OFFSET + 4, ADDR_MOD_7, p_movd2b::MOV_4_ROWS, row0 + 4);
    TTI_MOVD2B(0, p_movd2b::SRC_ROW16_OFFSET + 8, ADDR_MOD_7, p_movd2b::MOV_4_ROWS, row0 + 8);
    TTI_MOVD2B(0, p_movd2b::SRC_ROW16_OFFSET + 12, ADDR_MOD_7, p_movd2b::MOV_4_ROWS, row0 + 12);
    TTI_TRNSPSRCB;
    TTI_MOVB2A(p_movb2a::SRCA_ZERO_OFFSET + 0, ADDR_MOD_7, p_movb2a::MOV_4_ROWS, p_movb2a::SRCB_ROW16_OFFSET + 0);
    TTI_MOVB2A(p_movb2a::SRCA_ZERO_OFFSET + 4, ADDR_MOD_7, p_movb2a::MOV_4_ROWS, p_movb2a::SRCB_ROW16_OFFSET + 4);
    TTI_MOVB2A(p_movb2a::SRCA_ZERO_OFFSET + 8, ADDR_MOD_7, p_movb2a::MOV_4_ROWS, p_movb2a::SRCB_ROW16_OFFSET + 8);
    TTI_MOVB2A(p_movb2a::SRCA_ZERO_OFFSET + 12, ADDR_MOD_7, p_movb2a::MOV_4_ROWS, p_movb2a::SRCB_ROW16_OFFSET + 12);
    math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(m_tile);
}

inline void scores_row_max(uint32_t s_tile, uint32_t m_tile) {
    using namespace ckernel;
    math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(m_tile);
    scores_face_to_srca<0>(s_tile, m_tile);
    attn::row_max_face<p_setrwc::CLR_AB>();
    scores_face_to_srca<16>(s_tile, m_tile);
    attn::row_max_face<p_setrwc::CLR_NONE>();
    attn::row_max_fold();
    attn::row_max_next_rows();
    scores_face_to_srca<0>(s_tile, m_tile);
    attn::row_max_face<p_setrwc::CLR_AB>();
    scores_face_to_srca<16>(s_tile, m_tile);
    attn::row_max_face<p_setrwc::CLR_NONE>();
    attn::row_max_fold();
    TTI_SETRWC(p_setrwc::CLR_AB, 0, 0, 0, 0, p_setrwc::SET_BD);
}

inline void zero_step_addr7() {
    using namespace ckernel;
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}}.set(ADDR_MOD_7);
}

inline void math_whole_dest() {
    using namespace ckernel;
    using namespace ckernel::math;
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
inline void pack_whole_dest() {
    using namespace ckernel;
    using namespace ckernel::packer;
    reset_dest_offset_id();
    select_packer_dest_registers<DstSync::SyncHalf>();
}
#endif

template <uint32_t cb_qk_im, uint32_t cb_scale, uint32_t Sq, uint32_t Sk, uint32_t DHt, uint32_t scale_fp32>
void scores_block_in_dest(
    uint32_t cb_q, uint32_t cb_k, uint32_t cb_cur_max, uint32_t cb_prev_max, uint32_t cb_cur_sum, bool with_prev_max) {
    constexpr uint32_t s0 = 4;
    MATH((math_whole_dest()));
    PACK((pack_whole_dest()));
    CircularBuffer(cb_q).wait_front(Sq * DHt);
    CircularBuffer(cb_k).wait_front(DHt * Sk);
    attn::formats(cb_k, cb_q);
    attn::mm_setup(cb_q, cb_k, true, Sk, Sq, DHt);
    for (uint32_t inner = 0; inner < DHt; inner++) {
        attn::mm(cb_q, cb_k, inner, inner * Sk, s0, true, Sk, Sq, DHt);
    }
    CircularBuffer(cb_k).pop_front(DHt * Sk);
    CircularBuffer(cb_scale).wait_front(1);
    CircularBuffer(cb_cur_max).reserve_back(Sq);
    attn::max_setup();
    attn::formats(cb_qk_im, cb_scale);
    attn::rowmax_setup(cb_cur_max);
    MATH((zero_step_addr7()));
    for (uint32_t r = 0; r < Sq; r++) {
        for (uint32_t j = 0; j < Sk; j++) {
            UNPACK((attn::unpack_scaler_only(cb_scale)));
            MATH((scores_row_max(s0 + r * Sk + j, r)));
        }
    }
    attn::rowmax_done();
    if (with_prev_max) {
        attn::format_a(cb_prev_max);
        attn::copy_setup(cb_prev_max);
        for (uint32_t r = 0; r < Sq; r++) {
            attn::copy(cb_prev_max, r, Sq + r);
        }
        for (uint32_t r = 0; r < Sq; r++) {
            attn::max_cols(r, Sq + r, r);
        }
    }
    MATH((ckernel::math::set_math_semaphores()));
    attn::drain();
    PACK((attn::pack_format(cb_cur_max)));
    for (uint32_t r = 0; r < Sq; r++) {
        attn::out(r, cb_cur_max);
    }
    PACK((attn::pack_give_back_to_math()));
    CircularBuffer(cb_cur_max).push_back(Sq);
    attn::formats(cb_qk_im, cb_cur_max);
    attn::sub_cols_over_dest_setup();
    PACK((attn::pack_clamp_negative(true)));
    CircularBuffer(cb_cur_max).wait_front(Sq);
    CircularBuffer(cb_cur_sum).reserve_back(Sq);
    for (uint32_t i = 0; i < Sq; i++) {
        MATH((ckernel::math::math_dest_wait()));
        for (uint32_t j = 0; j < Sk; j++) {
            attn::sub_cols_over_dest(cb_cur_max, i, s0 + i * Sk + j);
            attn::exp_setup();
            attn::exp_scaled<scale_fp32>(s0 + i * Sk + j);
        }
        MATH((ckernel::math::set_math_semaphores()));
        attn::drain();
        CircularBuffer(cb_qk_im).reserve_back(Sk);
        PACK((attn::pack_format(cb_qk_im)));
        for (uint32_t j = 0; j < Sk; j++) {
            attn::out(s0 + i * Sk + j, cb_qk_im);
        }
        CircularBuffer(cb_qk_im).push_back(Sk);
        PACK((attn::pack_format(cb_cur_sum)));
        for (uint32_t j = 0; j < Sk; j++) {
            attn::out_at(s0 + i * Sk + j, cb_cur_sum, i);
            if (j == 0) {
                PACK((attn::pack_accumulate(1)));
            }
        }
        PACK((attn::pack_give_back_to_math()));
        PACK((attn::pack_accumulate(0)));
    }
    CircularBuffer(cb_cur_sum).push_back(Sq);
    PACK((attn::pack_clamp_negative(false)));
    MATH((math_whole_dest()));
    PACK((pack_whole_dest()));
}

void block_product(
    uint32_t in0_cb,
    uint32_t in1_cb,
    uint32_t out_cb,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    uint32_t in0_num_subblocks,
    uint32_t in1_num_subblocks,
    uint32_t in0_block_w,
    uint32_t subblock_h,
    uint32_t subblock_w,
    bool transpose) {
    CircularBuffer cb_in0(in0_cb);
    CircularBuffer cb_in1(in1_cb);
    CircularBuffer cb_out(out_cb);
    attn::mm_setup(in0_cb, in1_cb, transpose, subblock_w, subblock_h, in0_block_w);
    const uint32_t output_num_tiles = M * N;
    const uint32_t in0_subblock_all_cols_num_tiles = subblock_h * N;
    uint32_t in0_index_offset = 0;
    const uint32_t in0_subblock_num_tiles = subblock_h * in0_block_w;
    uint32_t in0_wait_tiles = in0_subblock_num_tiles;
    attn::formats(in1_cb, in0_cb);
    cb_in1.wait_front(K * N);
    cb_out.reserve_back(output_num_tiles);
    for (uint32_t in0_subblock = 0; in0_subblock < in0_num_subblocks; ++in0_subblock) {
        cb_in0.wait_front(in0_wait_tiles);
        uint32_t in1_index_offset = 0;
        for (uint32_t in1_subblock = 0; in1_subblock < in1_num_subblocks; ++in1_subblock) {
            attn::open();
            uint32_t in0_index = in0_index_offset;
            uint32_t in1_index = in1_index_offset;
            for (uint32_t inner_dim = 0; inner_dim < in0_block_w; inner_dim++) {
                attn::mm(in0_cb, in1_cb, in0_index, in1_index, 0, transpose, subblock_w, subblock_h, in0_block_w);
                in0_index++;
                in1_index += N;
            }
            attn::close();
            attn::drain();
            uint32_t dst_idx = 0;
            const uint32_t out_col_offset = in1_subblock * subblock_w;
            for (uint32_t r = 0; r < subblock_h; r++) {
                const uint32_t out_row_offset = r * N;
                for (uint32_t c = 0; c < subblock_w; c++) {
                    attn::out_at(dst_idx, out_cb, out_row_offset + out_col_offset + c);
                    dst_idx++;
                }
            }
            attn::release();
            in1_index_offset += subblock_w;
        }
        in0_index_offset += subblock_h * in0_block_w;
        in0_wait_tiles += in0_subblock_num_tiles;
        cb_out.push_back(in0_subblock_all_cols_num_tiles);
    }
    cb_in1.pop_front(K * N);
}

template <uint32_t M>
void row_sums(uint32_t in1_cb, uint32_t out_cb) {
    CircularBuffer cb_in1(in1_cb);
    CircularBuffer cb_out(out_cb);
    constexpr uint32_t rows = STATS_GRANULARITY;
    attn::formats(in1_cb, out_cb);
    attn::mm_setup(out_cb, in1_cb, 0, 1, rows, 1);
    PACK((attn::pack_format(out_cb)));
    cb_in1.wait_front(1);
    cb_out.wait_front(M);
    for (uint32_t g = 0; g < M / rows; ++g) {
        attn::open();
        attn::mm(out_cb, in1_cb, 0, 0, 0, 0, 1, rows, 1);
        attn::close();
        cb_out.pop_front(rows);
        attn::drain();
        for (uint32_t i = 0; i < rows; i++) {
            attn::out(i, out_cb);
        }
        attn::release();
        cb_out.push_back(rows);
    }
}

void fill_from(uint32_t src_cb, uint32_t src_tile, uint32_t out_cb, uint32_t out_offset, uint32_t count) {
    for (uint32_t base = 0; base < count; base += dest_tiles) {
        const uint32_t batch = (count - base < dest_tiles) ? (count - base) : dest_tiles;
        attn::open();
        for (uint32_t i = 0; i < batch; i++) {
            attn::copy(src_cb, src_tile, i);
        }
        attn::close();
        attn::drain();
        for (uint32_t i = 0; i < batch; i++) {
            attn::out_at(i, out_cb, out_offset + base + i);
        }
        attn::release();
    }
}

void causal_mask(
    uint32_t mask_cb,
    uint32_t neginf_idx,
    uint32_t diag_idx,
    uint32_t out_cb,
    uint32_t q_start_tile,
    uint32_t k_start_tile,
    uint32_t num_rows,
    uint32_t num_cols) {
    attn::format_a(mask_cb);
    PACK((attn::pack_format(out_cb)));
    attn::copy_setup(mask_cb);
    PACK((attn::pack_accumulate(1)));
    for (uint32_t row = 0; row < num_rows; row++) {
        const uint32_t row_offset = row * num_cols;
        const int32_t diag_col = static_cast<int32_t>(q_start_tile + row) - static_cast<int32_t>(k_start_tile);
        if (diag_col < 0) {
            fill_from(mask_cb, neginf_idx, out_cb, row_offset, num_cols);
        } else if (static_cast<uint32_t>(diag_col) < num_cols) {
            attn::open();
            attn::copy(mask_cb, diag_idx, 0);
            attn::close();
            attn::drain();
            attn::out_at(0, out_cb, row_offset + static_cast<uint32_t>(diag_col));
            attn::release();
            const uint32_t neginf_start = static_cast<uint32_t>(diag_col) + 1;
            if (neginf_start < num_cols) {
                fill_from(mask_cb, neginf_idx, out_cb, row_offset + neginf_start, num_cols - neginf_start);
            }
        }
    }
    PACK((attn::pack_accumulate(0)));
}

template <uint32_t in0_cb, uint32_t scale_cb, uint32_t rows>
void running_row_max(uint32_t out_cb, uint32_t prev_cb, uint32_t cols, bool with_prev) {
    CircularBuffer cb_in0(in0_cb);
    CircularBuffer cb_scale(scale_cb);
    CircularBuffer cb_out(out_cb);
    cb_scale.wait_front(1);
    cb_in0.wait_front(rows * cols);
    cb_out.reserve_back(rows);
    PACK((attn::pack_format(out_cb)));
    attn::max_setup();
    constexpr uint32_t group = (dest_tiles / 2 < rows) ? dest_tiles / 2 : rows;
    for (uint32_t i0 = 0; i0 < rows; i0 += group) {
        const uint32_t g = (rows - i0 < group) ? rows - i0 : group;
        attn::format_a(in0_cb);
        attn::open();
        attn::rowmax_setup(out_cb);
        for (uint32_t r = 0; r < g; r++) {
            for (uint32_t j = 0; j < cols; j++) {
                attn::rowmax(in0_cb, scale_cb, (i0 + r) * cols + j, 0, r);
            }
        }
        attn::rowmax_done();
        if (with_prev) {
            attn::format_a(prev_cb);
            attn::copy_setup(prev_cb);
            for (uint32_t r = 0; r < g; r++) {
                attn::copy(prev_cb, i0 + r, group + r);
            }
            for (uint32_t r = 0; r < g; r++) {
                attn::max_cols(r, group + r, r);
            }
        }
        attn::close();
        attn::drain();
        for (uint32_t r = 0; r < g; r++) {
            attn::out(r, out_cb);
        }
        attn::release();
    }
    cb_out.push_back(rows);
}

template <uint32_t in0_cb, uint32_t rows, uint32_t scale_fp32>
void exp_shifted_in_place(uint32_t in1_cb, uint32_t sum_cb, uint32_t cols) {
    CircularBuffer cb_in0(in0_cb);
    CircularBuffer cb_in1(in1_cb);
    CircularBuffer cb_sum(sum_cb);
    attn::formats(in0_cb, in1_cb);
    attn::sub_cols_setup(in0_cb, in1_cb);
    PACK((attn::pack_clamp_negative(true)));
    cb_in0.wait_front(rows * cols);
    cb_in1.wait_front(rows);
    cb_sum.reserve_back(rows);
    constexpr uint32_t step = SUB_EXP_GRANULARITY;
    const uint32_t dst_count = (cols < step) ? cols : step;
    const uint32_t groups = (cols >= step) ? (cols / step) : 1;
    for (uint32_t i = 0; i < rows; ++i) {
        for (uint32_t u = 0; u < groups; u++) {
            attn::open();
            for (uint32_t j = 0; j < dst_count; ++j) {
                attn::sub_cols(in0_cb, in1_cb, j, i, j);
                attn::exp_setup();
                attn::exp_scaled<scale_fp32>(j);
            }
            attn::close();
            cb_in0.pop_front(dst_count);
            cb_in0.reserve_back(dst_count);
            attn::drain();
            PACK((attn::pack_format(in0_cb)));
            for (uint32_t j = 0; j < dst_count; ++j) {
                attn::out(j, in0_cb);
            }
            cb_in0.push_back(dst_count);
            PACK((attn::pack_format(sum_cb)));
            if (u > 0) {
                PACK((attn::pack_accumulate(1)));
            }
            for (uint32_t j = 0; j < dst_count; ++j) {
                attn::out_at(j, sum_cb, i);
                if (u == 0 && j == 0) {
                    PACK((attn::pack_accumulate(1)));
                }
            }
            attn::release();
            PACK((attn::pack_accumulate(0)));
        }
    }
    cb_sum.push_back(rows);
    PACK((attn::pack_clamp_negative(false)));
}

template <uint32_t scale_fp32>
void exp_of_difference(uint32_t in0_cb, uint32_t in1_cb, uint32_t out_cb, uint32_t num_tiles) {
    CircularBuffer cb_in0(in0_cb);
    CircularBuffer cb_in1(in1_cb);
    CircularBuffer cb_out(out_cb);
    attn::sub_setup(in0_cb, in1_cb);
    attn::exp_setup();
    cb_in0.wait_front(num_tiles);
    cb_in1.wait_front(num_tiles);
    cb_out.reserve_back(num_tiles);
    constexpr uint16_t scale_bf16 = scale_fp32 >> 16;
    for (uint32_t i = 0; i < num_tiles; i++) {
        invalidate_l1_cache();
        attn::open();
        attn::sub(in0_cb, in1_cb, i, i, 0);
        MATH((attn::math_exp_cols<scale_bf16>(0)));
        attn::close();
        attn::drain();
        attn::out(0, out_cb);
        attn::release();
        cb_out.push_back(1);
    }
}

void scale_rows_in_place(uint32_t in0_cb, uint32_t in1_cb, uint32_t num_tiles) {
    CircularBuffer cb_in0(in0_cb);
    CircularBuffer cb_in1(in1_cb);
    attn::formats(in0_cb, in1_cb);
    attn::mul_cols_setup(in0_cb, in1_cb);
    PACK((attn::pack_format(in0_cb)));
    cb_in0.wait_front(num_tiles);
    cb_in1.wait_front(num_tiles);
    for (uint32_t i = 0; i < num_tiles; i++) {
        attn::open();
        attn::mul_cols(in0_cb, in1_cb, 0, i, 0);
        attn::close();
        cb_in0.pop_front(1);
        cb_in0.reserve_back(1);
        attn::drain();
        attn::out(0, in0_cb);
        attn::release();
        cb_in0.push_back(1);
    }
}

void add_in_place(uint32_t in0_cb, uint32_t in1_cb, uint32_t num_tiles) {
    CircularBuffer cb_in0(in0_cb);
    CircularBuffer cb_in1(in1_cb);
    attn::formats(in0_cb, in1_cb);
    PACK((attn::pack_format(in0_cb)));
    attn::add_setup(in0_cb, in1_cb);
    cb_in0.wait_front(num_tiles);
    cb_in1.wait_front(num_tiles);
    for (uint32_t i = 0; i < num_tiles; i++) {
        attn::open();
        attn::add(in0_cb, in1_cb, i, i, 0);
        attn::close();
        attn::drain();
        attn::out(0, in0_cb);
        attn::release();
    }
    cb_in0.pop_front(num_tiles);
    cb_in1.pop_front(num_tiles);
    cb_in0.reserve_back(num_tiles);
    cb_in0.push_back(num_tiles);
}

template <uint32_t rows, uint32_t cols, bool accumulate>
void scale_block_rows(uint32_t in0_cb, uint32_t in1_cb, uint32_t out_cb) {
    CircularBuffer cb_in0(in0_cb);
    CircularBuffer cb_in1(in1_cb);
    CircularBuffer cb_out(out_cb);
    constexpr uint32_t num_tiles = rows * cols;
    attn::formats(in0_cb, in1_cb);
    PACK((attn::pack_format(out_cb)));
    attn::mul_cols_setup(in0_cb, in1_cb);
    cb_in0.wait_front(num_tiles);
    cb_in1.wait_front(rows);
    constexpr uint32_t dst_count = (cols < DHT_GRANULARITY) ? cols : DHT_GRANULARITY;
    constexpr uint32_t groups = (cols >= DHT_GRANULARITY) ? (cols / DHT_GRANULARITY) : 1;
    PACK((attn::pack_accumulate(accumulate)));
    if (!accumulate) {
        cb_out.reserve_back(num_tiles);
    }
    uint32_t in0_index = 0;
    for (uint32_t i = 0; i < rows; ++i) {
        for (uint32_t u = 0; u < groups; ++u) {
            attn::open();
            for (uint32_t j = 0; j < dst_count; ++j) {
                attn::mul_cols(in0_cb, in1_cb, in0_index, i, j);
                in0_index++;
            }
            attn::close();
            attn::drain();
            for (uint32_t j = 0; j < dst_count; ++j) {
                attn::out(j, out_cb);
            }
            attn::release();
        }
    }
    cb_in1.pop_front(rows);
    cb_in0.pop_front(num_tiles);
    if (accumulate) {
        PACK((attn::pack_accumulate(false)));
        cb_out.pop_front(num_tiles);
        cb_out.reserve_back(num_tiles);
        cb_out.push_back(num_tiles);
    } else {
        cb_out.push_back(num_tiles);
    }
}

void reciprocal_in_place(uint32_t in_cb, uint32_t num_tiles) {
    CircularBuffer cb_in(in_cb);
    attn::format_a(in_cb);
    attn::copy_setup(in_cb);
    MATH((attn::math_recip_setup()));
    PACK((attn::pack_format(in_cb)));
    cb_in.wait_front(num_tiles);
    for (uint32_t i = 0; i < num_tiles; ++i) {
        attn::open();
        attn::copy(in_cb, i, 0);
        MATH((attn::math_recip_cols(0)));
        attn::close();
        attn::drain();
        attn::out(0, in_cb);
        attn::release();
    }
    cb_in.pop_front(num_tiles);
    cb_in.reserve_back(num_tiles);
    cb_in.push_back(num_tiles);
}

}  // namespace

void kernel_main() {
    constexpr uint32_t Skt = get_compile_time_arg_val(0);
    constexpr uint32_t DHt = get_compile_time_arg_val(1);
    constexpr uint32_t vDHt = get_compile_time_arg_val(2);
    constexpr uint32_t Sq_chunk_t = get_compile_time_arg_val(3);
    constexpr uint32_t q_num_chunks = get_compile_time_arg_val(4);
    constexpr uint32_t Sk_chunk_t = get_compile_time_arg_val(5);
    constexpr uint32_t qk_in0_block_w = get_compile_time_arg_val(7);
    constexpr uint32_t qk_subblock_w = get_compile_time_arg_val(8);
    constexpr uint32_t qk_subblock_h = get_compile_time_arg_val(9);
    constexpr uint32_t qk_in0_num_subblocks = get_compile_time_arg_val(10);
    constexpr uint32_t qk_in1_num_subblocks = get_compile_time_arg_val(11);
    constexpr uint32_t out_in0_block_w = get_compile_time_arg_val(12);
    constexpr uint32_t out_subblock_w = get_compile_time_arg_val(13);
    constexpr uint32_t out_subblock_h = get_compile_time_arg_val(14);
    constexpr uint32_t out_in0_num_subblocks = get_compile_time_arg_val(15);
    constexpr uint32_t out_in1_num_subblocks = get_compile_time_arg_val(16);
    constexpr uint32_t scale_fp32 = get_compile_time_arg_val(21);

    static_assert(get_compile_time_arg_val(17) == 1);
    static_assert(get_compile_time_arg_val(18) == 0);
    static_assert(get_compile_time_arg_val(19) == 0);
    static_assert(get_compile_time_arg_val(20) == 0);
    static_assert(get_compile_time_arg_val(22) == 0);
    static_assert(get_compile_time_arg_val(23) == 0);
    static_assert(get_compile_time_arg_val(24) == 0);
    static_assert(get_compile_time_arg_val(28) == 0);
    static_assert(EXP_APPROX_MODE == 0);
    static_assert(DST_ACCUM_MODE);

    const uint32_t num_phases = get_arg_val<uint32_t>(0);
    const uint32_t global_q_start = get_arg_val<uint32_t>(4);
    const uint32_t global_q_count = get_arg_val<uint32_t>(5);
    for (uint32_t unit = 0; unit < global_q_count; ++unit) {
        q_slots[unit] = get_arg_val<uint32_t>(6 + unit);
    }

    constexpr uint32_t q_chunk_tiles = Sq_chunk_t * DHt;
    constexpr uint32_t qk_chunk_tiles = Sq_chunk_t * Sk_chunk_t;

    constexpr uint32_t cb_arg_offset = 30;
    constexpr uint32_t cb_q_in = get_compile_time_arg_val(cb_arg_offset + 0);
    constexpr uint32_t cb_k_in = get_compile_time_arg_val(cb_arg_offset + 1);
    constexpr uint32_t cb_v_in = get_compile_time_arg_val(cb_arg_offset + 2);
    constexpr uint32_t cb_mask_in = get_compile_time_arg_val(cb_arg_offset + 3);
    constexpr uint32_t cb_identity_scale_in = get_compile_time_arg_val(cb_arg_offset + 5);
    constexpr uint32_t cb_col_identity = get_compile_time_arg_val(cb_arg_offset + 6);
    constexpr uint32_t cb_out = get_compile_time_arg_val(cb_arg_offset + 9);
    constexpr uint32_t cb_qk_im = get_compile_time_arg_val(cb_arg_offset + 10);
    constexpr uint32_t cb_out_im_A = get_compile_time_arg_val(cb_arg_offset + 11);
    constexpr uint32_t cb_out_im_B = get_compile_time_arg_val(cb_arg_offset + 12);
    constexpr uint32_t cb_max_A = get_compile_time_arg_val(cb_arg_offset + 13);
    constexpr uint32_t cb_max_B = get_compile_time_arg_val(cb_arg_offset + 14);
    constexpr uint32_t cb_sum_A = get_compile_time_arg_val(cb_arg_offset + 15);
    constexpr uint32_t cb_sum_B = get_compile_time_arg_val(cb_arg_offset + 16);
    constexpr uint32_t cb_exp_max_diff = get_compile_time_arg_val(cb_arg_offset + 17);

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_q_in, cb_k_in, cb_out);
    attn::mm_start(cb_q_in, cb_k_in);
    CircularBuffer(cb_mask_in).wait_front(2);

    for (uint32_t phase = 0; phase < num_phases; ++phase) {
        for (uint32_t q_iter = 0; q_iter < global_q_count; ++q_iter) {
            const uint32_t q_chunk = q_slots[global_q_start + q_iter] % q_num_chunks;
            const uint32_t q_start_tile = q_chunk * Sq_chunk_t;
            const uint32_t q_high_tile = (q_start_tile + Sq_chunk_t < Skt) ? q_start_tile + Sq_chunk_t : Skt;
            uint32_t prev_sum = cb_sum_A;
            uint32_t cur_sum = cb_sum_B;
            uint32_t prev_max = cb_max_A;
            uint32_t cur_max = cb_max_B;
            uint32_t prev_out = cb_out_im_A;
            uint32_t cur_out = cb_out_im_B;
            const uint32_t k_chunk_end = (q_high_tile + Sk_chunk_t - 1) / Sk_chunk_t;
            uint32_t done = 0;
            for (uint32_t k_chunk = 0; k_chunk < k_chunk_end; ++k_chunk) {
                if (!(q_start_tile < (k_chunk + 1) * Sk_chunk_t)) {
                    scores_block_in_dest<cb_qk_im, cb_identity_scale_in, Sq_chunk_t, Sk_chunk_t, DHt, scale_fp32>(
                        cb_q_in, cb_k_in, cur_max, prev_max, cur_sum, done > 0);
                } else {
                    attn::formats(cb_k_in, cb_q_in);
                    PACK((attn::pack_format(cb_qk_im)));
                    block_product(
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
                    attn::formats(cb_qk_im, cb_mask_in);
                    CircularBuffer(cb_qk_im).wait_front(qk_chunk_tiles);
                    CircularBuffer(cb_qk_im).pop_front(qk_chunk_tiles);
                    CircularBuffer(cb_qk_im).reserve_back(qk_chunk_tiles);
                    causal_mask(cb_mask_in, 0, 1, cb_qk_im, q_start_tile, k_chunk * Sk_chunk_t, Sq_chunk_t, Sk_chunk_t);
                    CircularBuffer(cb_qk_im).push_back(qk_chunk_tiles);
                    attn::formats(cb_qk_im, cb_identity_scale_in);
                    running_row_max<cb_qk_im, cb_identity_scale_in, Sq_chunk_t>(
                        cur_max, prev_max, Sk_chunk_t, done > 0);
                    exp_shifted_in_place<cb_qk_im, Sq_chunk_t, scale_fp32>(cur_max, cur_sum, Sk_chunk_t);
                }
                attn::formats(cb_v_in, cb_qk_im);
                PACK((attn::pack_format(cur_out)));
                block_product(
                    cb_qk_im,
                    cb_v_in,
                    cur_out,
                    Sq_chunk_t,
                    vDHt,
                    Sk_chunk_t,
                    out_in0_num_subblocks,
                    out_in1_num_subblocks,
                    out_in0_block_w,
                    out_subblock_h,
                    out_subblock_w,
                    false);
                CircularBuffer(cb_qk_im).pop_front(qk_chunk_tiles);
                attn::formats(prev_max, cur_max);
                if (done > 0) {
                    exp_of_difference<scale_fp32>(prev_max, cur_max, cb_exp_max_diff, Sq_chunk_t);
                    CircularBuffer(prev_max).pop_front(Sq_chunk_t);
                    scale_rows_in_place(prev_sum, cb_exp_max_diff, Sq_chunk_t);
                    add_in_place(cur_sum, prev_sum, Sq_chunk_t);
                    scale_block_rows<Sq_chunk_t, vDHt, true>(prev_out, cb_exp_max_diff, cur_out);
                }
                const uint32_t s = prev_sum;
                prev_sum = cur_sum;
                cur_sum = s;
                const uint32_t o = prev_out;
                prev_out = cur_out;
                cur_out = o;
                const uint32_t m = prev_max;
                prev_max = cur_max;
                cur_max = m;
                done++;
            }
            row_sums<Sq_chunk_t>(cb_col_identity, prev_sum);
            reciprocal_in_place(prev_sum, Sq_chunk_t);
            PACK((attn::pack_format(cb_out)));
            scale_block_rows<Sq_chunk_t, vDHt, false>(prev_out, prev_sum, cb_out);
            CircularBuffer(prev_max).pop_front(Sq_chunk_t);
            CircularBuffer(cb_q_in).pop_front(q_chunk_tiles);
        }
    }
}
