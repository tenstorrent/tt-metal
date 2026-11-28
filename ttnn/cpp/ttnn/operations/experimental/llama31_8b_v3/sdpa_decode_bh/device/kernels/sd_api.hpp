// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#define REDUCE_OP (PoolType::MAX)
#define REDUCE_DIM (ReduceDim::REDUCE_ROW)
#include "api/debug/assert.h"
#include "api/dataflow/circular_buffer.h"
#include "cpp/ttnn/kernel_lib/dest_helpers.hpp"
#include "ttnn/operations/experimental/llama31_8b_v3/sdpa_decode_bh/device/kernels/sd_own_dest.hpp"
#include "ttnn/operations/experimental/llama31_8b_v3/sdpa_decode_bh/device/kernels/sd_own_cfg.hpp"
#include "ttnn/operations/experimental/llama31_8b_v3/sdpa_decode_bh/device/kernels/sd_own_pack.hpp"
#include "ttnn/operations/experimental/llama31_8b_v3/sdpa_decode_bh/device/kernels/sd_own_copy.hpp"
#include "ttnn/operations/experimental/llama31_8b_v3/sdpa_decode_bh/device/kernels/sd_hand.hpp"
#include "ttnn/operations/experimental/llama31_8b_v3/sdpa_decode_bh/device/kernels/sd_own_sfpuinit.hpp"
#include "ttnn/operations/experimental/llama31_8b_v3/sdpa_decode_bh/device/kernels/sd_own_hwstart.hpp"
#include "ttnn/operations/experimental/llama31_8b_v3/sdpa_decode_bh/device/kernels/sd_sfpu.hpp"

#if defined(ARCH_BLACKHOLE)
#ifdef TRISC_UNPACK
using namespace ckernel::unpacker;
using namespace ckernel;
#endif
#ifdef TRISC_MATH
using namespace ckernel::math;
using namespace ckernel;
using namespace sfpi;
#endif
#ifdef TRISC_PACK
using namespace ckernel::packer;
using namespace ckernel;
#endif
#endif

namespace ckernel {
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void sd_add_tiles(
    std::uint32_t icb0, std::uint32_t icb1, std::uint32_t itile0, std::uint32_t itile1, std::uint32_t idst) {
    UNPACK((sdh_unpack_ab<BroadcastType::NONE>(icb0, icb1, itile0, itile1)));
    MATH((sdh_bin<EltwiseBinaryType::ELWADD, BroadcastType::NONE, EltwiseBinaryReuseDestType::NONE>(idst)));
}
template <bool out_of_order_output = false, bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void sd_pack_tile(std::uint32_t ifrom_dst, std::uint32_t icb, std::uint32_t output_tile_index = 0) {
    PACK((sdh_pack<out_of_order_output>(ifrom_dst, icb, output_tile_index)));
}
template <bool pop_in1 = true>
void sd_add_block_inplace(uint32_t in0_cb, uint32_t in1_cb, uint32_t num_tiles) {
    CircularBuffer cb_in0(in0_cb);
    CircularBuffer cb_in1(in1_cb);

    sdo_reconfig(in0_cb, in1_cb);
    sdo_pack_reconfig(in0_cb);
    sdh_binop_init<EltwiseBinaryType::ELWADD, BroadcastType::NONE>();
    cb_in0.wait_front(num_tiles);
    cb_in1.wait_front(num_tiles);
    for (uint32_t i = 0; i < num_tiles; i++) {
        sdo_regs_acquire();
        sd_add_tiles(in0_cb, in1_cb, i, i, 0);
        sdo_regs_commit();
        sdo_regs_wait();
        sd_pack_tile(0, in0_cb);
        sdo_regs_release();
    }

    cb_in0.pop_front(num_tiles);
    if (pop_in1) {
        cb_in1.pop_front(num_tiles);
    }
    cb_in0.reserve_back(num_tiles);
    cb_in0.push_back(num_tiles);
}
template <
    EltwiseBinaryType eltwise_binary_type,
    EltwiseBinaryReuseDestType reuse_dest,
    bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void sd_binary_reuse_dest_tiles(
    std::uint32_t in_cb_id, std::uint32_t in_tile_index, std::uint32_t dst_tile_index) {
    UNPACK((sdh_unpack_reuse<reuse_dest>(in_cb_id, in_tile_index)));

    MATH((sdh_bin<eltwise_binary_type, BroadcastType::NONE, reuse_dest>(dst_tile_index)));
}
template <EltwiseBinaryReuseDestType reuse_dest, bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void sd_add_reuse_dest_tiles(std::uint32_t in_cb_id, std::uint32_t in_tile_index, std::uint32_t dst_tile_index) {
    sd_binary_reuse_dest_tiles<EltwiseBinaryType::ELWADD, reuse_dest, is_fp32_dest_acc_en>(
        in_cb_id, in_tile_index, dst_tile_index);
}
ALWI void sd_binop_with_scalar_tile_init() { sdo_sfpu_init_unary(); }
template <
    bool approx = false,
    uint32_t scale = 0x3F800000,
    bool clamp_negative = true,
    bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void sd_exp_tile_init() {
    MATH((sdo_sfpu_init_unary(sdv_exp_init<approx, scale, clamp_negative, is_fp32_dest_acc_en>)));
}
#ifdef TRISC_MATH
template <uint32_t S, VectorMode vector_mode = VectorMode::C>
void sd_fused_max_sub_exp_add_tile(uint32_t idst) {
    static_assert(DST_ACCUM_MODE && DST_SYNC_MODE == DstSync::SyncHalf);
    sdv_run(sdv_corr_rows<S>, idst, vector_mode);
}
#endif
template <uint32_t scale_fp32, VectorMode vector_mode = VectorMode::C>
void sd_correction_block(
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
        sdo_regs_acquire();
        sdo_copy_init(cb_worker_max);
        sd_exp_tile_init<EXP_APPROX_MODE>();
        sdo_copy_tile(cb_prev_max, i, dst_reg_0);
        sdo_copy_tile(cb_worker_max, i, dst_reg_1);
        sdo_copy_tile(cb_prev_sum, i, dst_reg_3);
        sdo_copy_tile(cb_worker_sum, i, worker_sum_dst);
        MATH((sd_fused_max_sub_exp_add_tile<scale_bf16, vector_mode>(0)));
        sdo_regs_commit();
        sdo_regs_wait();
        sd_pack_tile(dst_reg_0, cb_exp_max_diff);
        sd_pack_tile(dst_reg_1, cb_exp_max_diff_2);
        sd_pack_tile(dst_reg_2, cb_cur_max);
        sd_pack_tile(dst_reg_3, cb_cur_sum);
        sdo_regs_release();
        cb_cur_max_obj.push_back(1);
        cb_cur_sum_obj.push_back(1);
        cb_exp_max_diff_obj.push_back(1);
        cb_exp_max_diff_2_obj.push_back(1);
    }
    cb_prev_sum_obj.pop_front(num_head_tiles);
    cb_worker_sum_obj.pop_front(num_head_tiles);
}
ALWI void sd_matmul_block_init(
    std::uint32_t in0_cb_id,
    std::uint32_t in1_cb_id,
    const std::uint32_t transpose = 0,
    std::uint32_t ct_dim = 1,
    std::uint32_t rt_dim = 1,
    std::uint32_t kt_dim = 1,
    std::uint32_t call_line = __builtin_LINE()) {
    UNPACK((sdh_unpack_mm_init(transpose, kt_dim)));
    MATH((sdh_mm_init(transpose, 0)));
}
ALWI void sd_matmul_block_math_dynamic_throttle(
    std::uint32_t in0_cb_id,
    std::uint32_t in1_cb_id,
    std::uint32_t idst,
    const std::uint32_t transpose,
    std::uint32_t ct_dim,
    std::uint32_t rt_dim) {
    MATH((sdh_mm_block(idst, transpose, ct_dim, rt_dim)));
}
ALWI void sd_matmul_block(
    std::uint32_t in0_cb_id,
    std::uint32_t in1_cb_id,
    std::uint32_t in0_tile_index,
    std::uint32_t in1_tile_index,
    std::uint32_t idst,
    const std::uint32_t transpose,
    std::uint32_t ct_dim,
    std::uint32_t rt_dim,
    std::uint32_t kt_dim,
    std::uint32_t call_line = __builtin_LINE()) {
    UNPACK((sdh_unpack_mm(in0_cb_id, in1_cb_id, in0_tile_index, in1_tile_index, ct_dim, rt_dim, kt_dim)));

    MATH((sd_matmul_block_math_dynamic_throttle(in0_cb_id, in1_cb_id, idst, transpose, ct_dim, rt_dim)));
}
ALWI void sd_matmul_blocks(
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

    sd_matmul_block_init(in0_cb, in1_cb, transpose, subblock_w, subblock_h, in0_block_w);

    const uint32_t output_num_tiles = M * N;
    const uint32_t out_subblock_num_tiles = subblock_h * subblock_w;
    const uint32_t in0_subblock_all_cols_num_tiles = subblock_h * N;

    uint32_t in0_index_offset = 0;

    const uint32_t in0_subblock_num_tiles = subblock_h * in0_block_w;
    uint32_t in0_wait_tiles = in0_subblock_num_tiles;

    sdo_reconfig(in1_cb, in0_cb);
    cb_in1.wait_front(K * N);
    cb_out.reserve_back(output_num_tiles);

    for (uint32_t in0_subblock = 0; in0_subblock < in0_num_subblocks; ++in0_subblock) {
        cb_in0.wait_front(in0_wait_tiles);
        uint32_t in1_index_offset = 0;
        for (uint32_t in1_subblock = 0; in1_subblock < in1_num_subblocks; ++in1_subblock) {
            sdo_regs_acquire();

            uint32_t dst_index = 0;
            uint32_t in0_index = in0_index_offset;
            uint32_t in1_index = in1_index_offset;

            for (uint32_t inner_dim = 0; inner_dim < in0_block_w; inner_dim++) {
                sd_matmul_block(
                    in0_cb, in1_cb, in0_index, in1_index, dst_index, transpose, subblock_w, subblock_h, in0_block_w);
                in0_index++;
                in1_index += N;
            }
            if (add_mask) {
                cb_mask.wait_front(out_subblock_num_tiles);
                cb_zero.wait_front(1);
                sdo_reconfig(zero_cb, mask_cb);
                sdh_binop_init<EltwiseBinaryType::ELWADD, BroadcastType::NONE>(true);
                for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {
                    sd_add_tiles(zero_cb, mask_cb, 0, i, i);
                }
                sdo_reconfig(in1_cb, in0_cb);
                sd_matmul_block_init(in0_cb, in1_cb, transpose, subblock_w, subblock_h, in0_block_w);
            }
            sdo_regs_commit();
            sdo_regs_wait();
            uint32_t dst_idx = 0;
            uint32_t out_col_offset = in1_subblock * subblock_w;
            for (uint32_t r = 0; r < subblock_h; r++) {
                uint32_t out_row_offset = r * N;
                for (uint32_t c = 0; c < subblock_w; c++) {
                    sd_pack_tile<true>(dst_idx, out_cb, out_row_offset + out_col_offset + c);
                    dst_idx++;
                }
            }
            sdo_regs_release();
            in1_index_offset += subblock_w;
        }
        in0_index_offset += subblock_h * in0_block_w;
        in0_wait_tiles += in0_subblock_num_tiles;

        cb_out.push_back(in0_subblock_all_cols_num_tiles);
    }
    cb_in1.pop_front(K * N);
}
ALWI void sd_matmul_init(
    std::uint32_t in0_cb_id,
    std::uint32_t in1_cb_id,
    const std::uint32_t transpose = 0,
    std::uint32_t call_line = __builtin_LINE()) {
    MATH((sdh_mm_init(transpose, 0)));
    UNPACK((sdh_unpack_mm_init(transpose, 1)));
}
ALWI void sd_binary_max_tile_init() { MATH((sdo_sfpu_init_binary(sdv_bmax_init))); }
ALWI void sd_binary_max_tile(uint32_t idst0, uint32_t idst1, uint32_t odst, VectorMode vector_mode = VectorMode::RC) {
    MATH((sdv_run(sdv_bmax, 0, vector_mode, idst0, idst1, odst)));
}
template <VectorMode vector_mode = VectorMode::RC>
void sd_max_block(uint32_t in0, uint32_t in1, uint32_t out_cb, uint32_t num_tiles) {
    CircularBuffer cb_in0(in0);
    CircularBuffer cb_in1(in1);
    CircularBuffer cb_out(out_cb);

    sdo_copy_init(in0);
    sd_binary_max_tile_init();

    constexpr uint32_t dst_reg_0 = 0;
    constexpr uint32_t dst_reg_1 = 1;
    cb_in0.wait_front(num_tiles);
    cb_in1.wait_front(num_tiles);
    cb_out.reserve_back(num_tiles);
    for (uint32_t i = 0; i < num_tiles; ++i) {
        sdo_regs_acquire();
        sdo_copy_tile(in0, i, dst_reg_0);
        sdo_copy_tile(in1, i, dst_reg_1);
        sd_binary_max_tile(dst_reg_0, dst_reg_1, dst_reg_0, vector_mode);
        sdo_regs_commit();
        sdo_regs_wait();
        sd_pack_tile(dst_reg_0, out_cb, i);
        sdo_regs_release();
    }
    cb_out.push_back(num_tiles);
}
template <bool pop_in_cb>
void sd_move_block(uint32_t in_cb, uint32_t out_cb, uint32_t num_tiles) {
    CircularBuffer cb_in(in_cb);
    CircularBuffer cb_out(out_cb);

    sdo_copy_init(in_cb);

    cb_in.wait_front(num_tiles);
    cb_out.reserve_back(num_tiles);

#pragma GCC unroll 0
    for (uint32_t i = 0; i < num_tiles; i++) {
        sdo_regs_acquire();
        sdo_copy_tile(in_cb, i, 0);
        sdo_regs_commit();
        sdo_regs_wait();
        sd_pack_tile(0, out_cb);
        sdo_regs_release();
        cb_out.push_back(1);
    }
    if (pop_in_cb) {
        cb_in.pop_front(num_tiles);
    }
}
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void sd_mul_tiles_bcast_cols(uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst) {
    MATH((sdh_bin<EltwiseBinaryType::ELWMUL, BroadcastType::COL, EltwiseBinaryReuseDestType::NONE>(idst)));
    UNPACK((sdh_unpack_ab<BroadcastType::COL>(icb0, icb1, itile0, itile1)));
}
template <uint32_t rows, uint32_t cols, bool immediate_pop, bool pack_accumulate>
void sd_mul_block_bcast_cols(uint32_t in0_cb, uint32_t in1_cb, uint32_t out_cb) {
    CircularBuffer cb_in0(in0_cb);
    CircularBuffer cb_in1(in1_cb);
    CircularBuffer cb_out(out_cb);

    constexpr uint32_t num_tiles = rows * cols;

    sdo_reconfig(in0_cb, in1_cb);
    sdo_pack_reconfig(out_cb);
    sdh_binop_init<EltwiseBinaryType::ELWMUL, BroadcastType::COL>();
    cb_in0.wait_front(num_tiles);
    cb_in1.wait_front(rows);

    if constexpr (immediate_pop) {
        static_assert(!pack_accumulate, "Unsupported parameter configuration");
        for (uint32_t i = 0; i < rows; ++i) {
            for (uint32_t j = 0; j < cols; ++j) {
                sdo_regs_acquire();
                sd_mul_tiles_bcast_cols(in0_cb, in1_cb, 0, i, 0);
                sdo_regs_commit();
                cb_in0.pop_front(1);
                cb_out.reserve_back(1);
                sdo_regs_wait();
                sd_pack_tile(0, out_cb);
                sdo_regs_release();
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
        PACK((sdo_pack_l1_acc(pack_accumulate)));
        if (!pack_accumulate) {
            cb_out.reserve_back(num_tiles);
        }
        uint32_t in0_index = 0;
        for (uint32_t i = 0; i < rows; ++i) {
            for (uint32_t u = 0; u < granularity; ++u) {
                sdo_regs_acquire();
                for (uint32_t j = 0; j < dst_tiles; ++j) {
                    sd_mul_tiles_bcast_cols(in0_cb, in1_cb, in0_index, i, j);
                    in0_index++;
                }
                sdo_regs_commit();
                sdo_regs_wait();
                for (uint32_t j = 0; j < dst_tiles; ++j) {
                    sd_pack_tile(j, out_cb);
                }
                sdo_regs_release();
            }
        }
        cb_in1.pop_front(rows);
        cb_in0.pop_front(num_tiles);
        if (pack_accumulate) {
            PACK((sdo_pack_l1_acc(false)));
            cb_out.pop_front(num_tiles);
            cb_out.reserve_back(num_tiles);
            cb_out.push_back(num_tiles);
        } else {
            cb_out.push_back(num_tiles);
        }
    }
}
template <uint32_t rows, uint32_t cols>
void sd_mul_block_bcast_cols_inplace(uint32_t in0_cb, uint32_t in1_cb) {
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

    sdo_reconfig(in0_cb, in1_cb);
    sdh_binop_init<EltwiseBinaryType::ELWMUL, BroadcastType::COL>();
    sdo_pack_reconfig(in0_cb);
    cb_in0.wait_front(num_tiles);
    cb_in1.wait_front(rows);
    for (uint32_t i = 0; i < rows; ++i) {
        for (uint32_t u = 0; u < granularity; ++u) {
            sdo_regs_acquire();
            for (uint32_t j = 0; j < dst_tiles; ++j) {
                sd_mul_tiles_bcast_cols(in0_cb, in1_cb, j, i, j);
            }
            sdo_regs_commit();
            cb_in0.pop_front(dst_tiles);
            cb_in0.reserve_back(dst_tiles);
            sdo_regs_wait();
            for (uint32_t j = 0; j < dst_tiles; ++j) {
                sd_pack_tile(j, in0_cb);
            }
            cb_in0.push_back(dst_tiles);
            sdo_regs_release();
        }
    }
    cb_in1.pop_front(rows);
    sdh_reconfig_srcb(in0_cb);
}
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void sd_mul_tiles(
    std::uint32_t icb0, std::uint32_t icb1, std::uint32_t itile0, std::uint32_t itile1, std::uint32_t idst) {
    UNPACK((sdh_unpack_ab<BroadcastType::NONE>(icb0, icb1, itile0, itile1)));
    MATH((sdh_bin<EltwiseBinaryType::ELWMUL, BroadcastType::NONE, EltwiseBinaryReuseDestType::NONE>(idst)));
}
void sd_mul_block_inplace(uint32_t in0_cb, uint32_t in1_cb, uint32_t num_tiles) {
    CircularBuffer cb_in0(in0_cb);
    CircularBuffer cb_in1(in1_cb);

    sdh_binop_init<EltwiseBinaryType::ELWMUL, BroadcastType::NONE>();
    cb_in0.wait_front(num_tiles);
    cb_in1.wait_front(num_tiles);
    for (uint32_t i = 0; i < num_tiles; i++) {
        invalidate_l1_cache();
        sdo_regs_acquire();
        sd_mul_tiles(in0_cb, in1_cb, 0, i, 0);
        sdo_regs_commit();
        cb_in0.pop_front(1);
        cb_in0.reserve_back(1);
        sdo_regs_wait();
        sd_pack_tile(0, in0_cb);
        sdo_regs_release();
        cb_in0.push_back(1);
    }
}
#ifdef TRISC_MATH
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
void sd_recip_tile_first_column(uint32_t idst) {
    static_assert(APPROX);
    sdv_run(sdv_recip_rows, idst, VectorMode::C);
}
#endif
void sd_recip_block_inplace(uint32_t in_cb, uint32_t num_tiles) {
    CircularBuffer cb_in(in_cb);

    sdo_reconfig_srca(in_cb);
    sdo_copy_init(in_cb);

    MATH((sdo_sfpu_init_unary<true>(sdv_recip_init)));
    sdo_pack_reconfig(in_cb);

    cb_in.wait_front(num_tiles);
    for (uint32_t i = 0; i < num_tiles; ++i) {
        sdo_regs_acquire();
        sdo_copy_tile(in_cb, i, 0);
        MATH((sd_recip_tile_first_column(0)));
        sdo_regs_commit();
        sdo_regs_wait();
        sd_pack_tile(0, in_cb);
        sdo_regs_release();
    }
    cb_in.pop_front(num_tiles);
    cb_in.reserve_back(num_tiles);
    cb_in.push_back(num_tiles);
}
template <PoolType reduce_type, ReduceDim reduce_dim, bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void sd_reduce_init(
    std::uint32_t icb, std::uint32_t icb_scaler, std::uint32_t ocb, std::uint32_t call_line = __builtin_LINE()) {
    static_assert(reduce_dim == ReduceDim::REDUCE_ROW);
    if constexpr (reduce_type == PoolType::SUM) {
        UNPACK((sdh_unpack_rsum_init()));
        MATH((sdh_rsum_init()));
    } else {
        static_assert(reduce_type == PoolType::MAX);
        UNPACK((sdh_unpack_rmaxg_init()));
        MATH((sdh_rmaxg_init()));
    }
    static_assert(reduce_dim == ReduceDim::REDUCE_ROW);
    PACK((sdh_pack_rmask<true>()));
}
template <PoolType reduce_type, ReduceDim reduce_dim, bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void sd_reduce_tile(
    std::uint32_t icb, std::uint32_t icb_scaler, std::uint32_t itile, std::uint32_t itile_scaler, std::uint32_t idst) {
    static_assert(reduce_dim == ReduceDim::REDUCE_ROW);
    if constexpr (reduce_type == PoolType::SUM) {
        MATH((sdh_rsum(idst)));
        UNPACK((sdh_unpack_rsum(icb, icb_scaler, itile, itile_scaler)));
    } else {
        MATH((sdh_rmaxg(idst)));
        UNPACK((sdh_unpack_rmaxg(icb, icb_scaler, itile, itile_scaler)));
    }
}
ALWI void sd_reduce_uninit(std::uint32_t icb = 0) { PACK((sdh_pack_rmask<false>())); }
template <
    PoolType pool_type,
    ReduceDim reduce_dim,
    uint32_t in0_cb,
    uint32_t scale_cb,
    uint32_t rows,
    uint32_t cols,
    VectorMode vector_mode = VectorMode::C>
void sd_reduce_c(uint32_t out_cb, uint32_t prev_cb, bool do_eltwise_max = false) {
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
        sdo_regs_acquire();

        if (do_eltwise_max) {
            cb_prev.wait_front(g * dst_tiles);

            sdo_reconfig_srca(prev_cb);
            sdh_copy_tr_init(prev_cb);
            for (uint32_t i = 0; i < dst_tiles; i++) {
                const uint32_t cur_max_dst_idx = i;
                sdo_copy_tile(prev_cb, (row_start_idx + i), cur_max_dst_idx);
            }
            sdo_reconfig_srca(in0_cb);
        }

        UNPACK((sdh_unpack_rmax_cfg<true>()));
        MATH((sdh_rmax_init()));
        PACK((sdh_pack_rmask<true>()));
        for (uint32_t i = 0; i < dst_tiles; i++) {
            const uint32_t reduce_dst_idx = i;
            UNPACK((sdh_unpack_rmax<cols>(in0_cb, scale_cb, (row_start_idx + i) * cols)));
            MATH((sdh_rmax<cols>(reduce_dst_idx)));
        }
        PACK((sdh_pack_rmask<false>()));
        UNPACK((sdh_unpack_rmax_cfg<false>()));

        sdo_regs_commit();
        sdo_regs_wait();
        sdo_pack_reconfig(out_cb);
        for (uint32_t i = 0; i < dst_tiles; i++) {
            const uint32_t cur_max_dst_idx = i;
            sd_pack_tile<true>(cur_max_dst_idx, out_cb, (row_start_idx + i));
        }
        sdo_regs_release();

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
void sd_reduce_c(uint32_t out_cb, uint32_t prev_cb, uint32_t cols, bool do_eltwise_max = false) {
    CircularBuffer cb_in0(in0_cb);
    CircularBuffer cb_scale(scale_cb);
    CircularBuffer cb_out(out_cb);

    uint32_t num_tiles = rows * cols;
    cb_scale.wait_front(1);
    cb_in0.wait_front(num_tiles);
    cb_out.reserve_back(rows);

    sdo_pack_reconfig(out_cb);

    sd_binary_max_tile_init();
    constexpr uint32_t reduce_dst_idx = 0;
    constexpr uint32_t prev_max_dst_idx = 1;

    for (uint32_t i = 0; i < rows; i++) {
        sdo_reconfig_srca(in0_cb);
        sdo_regs_acquire();
        sd_reduce_init<pool_type, reduce_dim>(in0_cb, scale_cb, out_cb);
        for (uint32_t j = 0; j < cols; j++) {
            sd_reduce_tile<pool_type, reduce_dim>(in0_cb, scale_cb, i * cols + j, 0, reduce_dst_idx);
        }
        sd_reduce_uninit();
        if (do_eltwise_max) {
            sdo_reconfig_srca(prev_cb);
            sdo_copy_init(prev_cb);
            sdo_copy_tile(prev_cb, i, prev_max_dst_idx);
            sd_binary_max_tile(reduce_dst_idx, prev_max_dst_idx, reduce_dst_idx, vector_mode);
        }

        sdo_regs_commit();
        sdo_regs_wait();
        sd_pack_tile(reduce_dst_idx, out_cb);
        sdo_regs_release();
    }

    cb_out.push_back(rows);
}
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void sd_sub_tiles(
    std::uint32_t icb0, std::uint32_t icb1, std::uint32_t itile0, std::uint32_t itile1, std::uint32_t idst) {
    UNPACK((sdh_unpack_ab<BroadcastType::NONE>(icb0, icb1, itile0, itile1)));
    MATH((sdh_bin<EltwiseBinaryType::ELWSUB, BroadcastType::NONE, EltwiseBinaryReuseDestType::NONE>(idst)));
}
#if defined(TRISC_MATH) || defined(TRISC_PACK)
template <bool SDPA_EXP_APPROX_MODE, uint16_t scale_bf16, bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
void sd_exp_tile_first_column(uint32_t idst) {
#ifdef TRISC_MATH
    static_assert(!SDPA_EXP_APPROX_MODE && is_fp32_dest_acc_en);
    sdv_run(sdv_exp_fc<scale_bf16>, idst, VectorMode::C);
#endif
}
#endif
template <uint32_t scale_fp32>
void sd_sub_exp_block(uint32_t in0_cb, uint32_t in1_cb, uint32_t out_cb, uint32_t num_tiles) {
    CircularBuffer cb_in0(in0_cb);
    CircularBuffer cb_in1(in1_cb);
    CircularBuffer cb_out(out_cb);

    sdh_binop_init<EltwiseBinaryType::ELWSUB, BroadcastType::NONE>();
    sd_exp_tile_init<EXP_APPROX_MODE>();
    cb_in0.wait_front(num_tiles);
    cb_in1.wait_front(num_tiles);
    cb_out.reserve_back(num_tiles);

    constexpr uint16_t scale_bf16 = scale_fp32 >> 16;

    for (uint32_t i = 0; i < num_tiles; i++) {
        invalidate_l1_cache();
        sdo_regs_acquire();
        sd_sub_tiles(in0_cb, in1_cb, i, i, 0);
        MATH((sd_exp_tile_first_column<EXP_APPROX_MODE, scale_bf16>(0)));
        sdo_regs_commit();
        sdo_regs_wait();
        sd_pack_tile(0, out_cb);
        sdo_regs_release();
        cb_out.push_back(1);
    }
}
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void sd_sub_tiles_bcast_cols(uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst) {
    MATH((sdh_bin<EltwiseBinaryType::ELWSUB, BroadcastType::COL, EltwiseBinaryReuseDestType::NONE>(idst)));
    UNPACK((sdh_unpack_ab<BroadcastType::COL>(icb0, icb1, itile0, itile1)));
}
template <
    bool approx = false,
    bool scale_en = false,
    bool clamp_negative = true,
    int iterations = 8,
    bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void sd_exp_tile(uint32_t idst, VectorMode vector_mode = VectorMode::RC, uint16_t scale = p_sfpu::kCONST_1_FP16B) {
    static_assert(!approx && is_fp32_dest_acc_en);
    MATH((sdv_exp_vm<scale_en, iterations>(idst, vector_mode, static_cast<std::uint32_t>(scale))));
}
template <uint32_t P>
ALWI void sd_mul_unary_tile(uint32_t idst, VectorMode vector_mode = VectorMode::RC) {
    MATH((sdv_run(sdv_mul_rows<P, 8>, idst, vector_mode)));
}
template <
    uint32_t in0_cb,
    uint32_t rows,
    uint32_t scale_fp32,
    bool write_result_inplace = true,
    bool do_reduce = true,
    VectorMode vector_mode = VectorMode::RC>
void sd_sub_exp_block_bcast_cols_inplace(uint32_t in1_cb, uint32_t reduce_cb, uint32_t cols) {
    CircularBuffer cb_in0(in0_cb);
    CircularBuffer cb_in1(in1_cb);
    CircularBuffer cb_reduce(reduce_cb);

    sdo_reconfig(in0_cb, in1_cb);
    sdh_binop_init<EltwiseBinaryType::ELWSUB, BroadcastType::COL>();

    if constexpr (EXP_APPROX_MODE || vector_mode != VectorMode::RC) {
        sd_exp_tile_init<true, scale_fp32, false>();
    }
    PACK((sdo_pack_relu(true)));

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
            sdo_regs_acquire();
            for (uint32_t j = 0; j < dst_tiles; ++j) {
                sd_sub_tiles_bcast_cols(in0_cb, in1_cb, j, i, j);

                constexpr int iterations = (vector_mode == VectorMode::RC) ? 32 : 8;
                constexpr VectorMode vector_mode_exp = (vector_mode == VectorMode::RC) ? VectorMode::None : vector_mode;
                if constexpr (EXP_APPROX_MODE || vector_mode != VectorMode::RC) {
                    sd_exp_tile<true, false, false, iterations>(j, vector_mode_exp);
                } else {
                    sd_binop_with_scalar_tile_init();
                    sd_mul_unary_tile<scale_fp32>(j);
                    sd_exp_tile_init<false, 0x3F800000, true>();
                    sd_exp_tile<false, false, true, iterations>(j, vector_mode_exp);
                }
            }
            sdo_regs_commit();

            if constexpr (write_result_inplace) {
                cb_in0.pop_front(dst_tiles);
                cb_in0.reserve_back(dst_tiles);
            }

            sdo_regs_wait();

            if constexpr (write_result_inplace) {
                sdo_pack_reconfig(in0_cb);
                for (uint32_t j = 0; j < dst_tiles; ++j) {
                    sd_pack_tile(j, in0_cb);
                }

                cb_in0.push_back(dst_tiles);
            }

            if constexpr (do_reduce) {
                sdo_pack_reconfig(reduce_cb);

                if (u > 0) {
                    PACK((sdo_pack_l1_acc(1)));
                }
                for (uint32_t j = 0; j < dst_tiles; ++j) {
                    sd_pack_tile<true>(j, reduce_cb, i);
                    if (u == 0 && j == 0) {
                        PACK((sdo_pack_l1_acc(1)));
                    }
                }
            }
            sdo_regs_release();
            if constexpr (do_reduce) {
                PACK((sdo_pack_l1_acc(0)));
            }
        }
    }
    if constexpr (do_reduce) {
        cb_reduce.push_back(rows);
    }

    PACK((sdo_pack_relu(false)));
}
}  // namespace ckernel
