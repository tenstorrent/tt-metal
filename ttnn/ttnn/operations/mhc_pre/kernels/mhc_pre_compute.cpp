// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// mhc_pre compute kernel. Per block (block_token_tiles token tile-rows of this rank's K slice):
//
//   project_block       matmul_block helper: mix partial = X_blk @ W_slice   -> cb_partial [mix rows]
//                        (in0 WaitAndRetainOnLastBlock, num_k_blocks = 1: the X block stays resident)
//   sumsq_block         eltwise_chain Mul x*x DEST-accumulated over K        -> cb_sq_acc
//                        + reduce<SUM, REDUCE_ROW, Accurate>                  -> cb_partial [sumsq rows]
//   combine_block       root only: rank-ordered fp32 SFPU fold of cb_gathered -> cb_combined
//   coefficients_block  custom SFPU block op: r, pre, post, logits          -> cb_coef_out (+cb_logits_coef)
//   ymix_block          eltwise_chain Mul x_i * bcast_col(pre_i), DEST-accumulated over i -> cb_y_out
//   sinkhorn_block      custom SFPU block op on the owned rows              -> cb_comb_coef
//
// Raw-LLK deviations (helper considered and rejected; see op_design.md "Helpers considered and rejected"):
//   * combine_block: reduce<AccumulateViaAdd> reads the fp32 partials through the FPU (tf32 truncation);
//     eltwise_chain cannot express a runtime group_cores-deep fold in one DEST window. Realized with the
//     chain's own two primitives, copy_tile (UnpackToDestFp32) + add_binary_tile (SFPU), in a loop.
//   * coefficients_block / sinkhorn_block: the math runs across slots WITHIN one tile (row / column sums
//     over the n x n matrix, per-token softmax); chain SFPU elements are whole-tile elementwise. Realized as
//     custom SFPI functions over one DEST tile that call the non-approximate SFPI exp primitive, with a
//     Newton-refined reciprocal / rsqrt (no bare approximate reciprocal).

#include <stdint.h>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp"

#ifdef TRISC_MATH
#include "sfpi.h"
#include "ckernel_sfpu_exp.h"
#include "sfpu/ckernel_sfpu_converter.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#endif

constexpr uint32_t cb_x_resident = get_compile_time_arg_val(0);
constexpr uint32_t cb_weight = get_compile_time_arg_val(1);
constexpr uint32_t cb_bias_coef = get_compile_time_arg_val(2);
constexpr uint32_t cb_reduce_scaler = get_compile_time_arg_val(3);
constexpr uint32_t cb_sq_acc = get_compile_time_arg_val(4);
constexpr uint32_t cb_partial = get_compile_time_arg_val(5);
constexpr uint32_t cb_gathered = get_compile_time_arg_val(6);
constexpr uint32_t cb_combined = get_compile_time_arg_val(7);
constexpr uint32_t cb_coef_in = get_compile_time_arg_val(8);
constexpr uint32_t cb_coef_out = get_compile_time_arg_val(9);
constexpr uint32_t cb_logits_coef = get_compile_time_arg_val(10);
constexpr uint32_t cb_comb_coef = get_compile_time_arg_val(11);
constexpr uint32_t cb_pre_cols = get_compile_time_arg_val(12);
constexpr uint32_t cb_y_out = get_compile_time_arg_val(13);
constexpr uint32_t n_streams = get_compile_time_arg_val(14);
constexpr uint32_t block_token_tiles = get_compile_time_arg_val(15);
constexpr uint32_t core_k_tiles_max = get_compile_time_arg_val(16);
constexpr uint32_t group_cores = get_compile_time_arg_val(17);

#ifdef TRISC_MATH
namespace mhc_sfpu {

using namespace sfpi;
using ckernel::sfpu::Converter;

constexpr int N = static_cast<int>(n_streams);
constexpr int MIX = N * (N + 2);  // slots 0..MIX-1 hold the mixes, slot MIX holds sum(x^2)
constexpr int LOGIT0 = 2 * N;     // first logit / comb slot
constexpr int SCRATCH0 = 31;      // unused slots (MIX+1 .. 31) serve as per-lane scratch
constexpr int SCRATCH1 = 30;
constexpr int TILE_SLOTS = 32;  // dst_reg stride of one DEST tile

// 1/x for x > 0 finite: hardware seed + two Newton steps (~fp32 accurate).
sfpi_inline vFloat recip_pos(vFloat x) {
    vFloat y = approx_recip(x);
    vFloat t = 2.0f - x * y;
    y = y * t;
    t = 2.0f - x * y;
    y = y * t;
    return y;
}

// 1/sqrt(x) for x > 0 finite: bit-trick seed + four Newton steps.
sfpi_inline vFloat rsqrt_pos(vFloat x) {
    vInt i = as<vInt>(as<vUInt>(x) >> 1);
    vInt magic = 0x5f3759df;
    vFloat y = as<vFloat>(magic - i);
    vFloat half_x = x * 0.5f;
#pragma GCC unroll 0
    for (int it = 0; it < 4; ++it) {
        y = y * (1.5f - half_x * y * y);
    }
    return y;
}

// sigmoid(x) = 1 / (1 + exp(-x)); exp argument clamped so 1 + exp stays finite.
sfpi_inline vFloat sigmoid_acc(vFloat x) {
    vFloat z = -x;
    v_if(z > 80.0f) { z = 80.0f; }
    v_endif;
    vFloat e = ckernel::sfpu::_sfpu_exp_fp32_accurate_<false>(z);
    return recip_pos(e + 1.0f);
}

// Round-to-nearest-even to tf32 (10 explicit mantissa bits): the FPU then reads the value losslessly.
sfpi_inline vFloat rne_tf32(vFloat v) {
    vUInt u = as<vUInt>(v);
    vUInt lsb = (u << 18) >> 31;
    u = u + 0xFFF;
    u = u + lsb;
    u = (u >> 13) << 13;
    return as<vFloat>(u);
}

// coefficients_block on one coefficient-major tile: DEST tile 0 = raw sums, DEST tile 1 = bias.
void coefficients(
    uint32_t a_pre_bits,
    uint32_t a_post_bits,
    uint32_t a_res_bits,
    uint32_t eps_bits,
    uint32_t norm_eps_bits,
    uint32_t inv_nc_bits) {
    {
        vFloat ssq = dst_reg[MIX];
        vFloat inv_nc = Converter::as_float(inv_nc_bits);
        vFloat neps = Converter::as_float(norm_eps_bits);
        vFloat r = rsqrt_pos(ssq * inv_nc + neps);
        dst_reg[MIX] = r;
    }
#pragma GCC unroll 0
    for (int k = 0; k < MIX; ++k) {
        const uint32_t a_bits = k < N ? a_pre_bits : (k < 2 * N ? a_post_bits : a_res_bits);
        vFloat r = dst_reg[MIX];
        vFloat a = Converter::as_float(a_bits);
        vFloat z = dst_reg[k] * r;
        z = z * a + dst_reg[TILE_SLOTS + k];
        if (k < N) {
            vFloat s = sigmoid_acc(z);
            vFloat e = Converter::as_float(eps_bits);
            dst_reg[k] = rne_tf32(s + e);
        } else if (k < 2 * N) {
            vFloat s = sigmoid_acc(z);
            dst_reg[k] = s + s;
        } else {
            dst_reg[k] = z;
        }
    }
}

// Column normalisation: m[i][j] *= 1 / (sum_i m[i][j] + eps), sums in index order.
sfpi_inline void col_norm(uint32_t eps_bits) {
#pragma GCC unroll 0
    for (int j = 0; j < N; ++j) {
        vFloat s = dst_reg[LOGIT0 + j];
        for (int i = 1; i < N; ++i) {
            s = s + dst_reg[LOGIT0 + i * N + j];
        }
        vFloat e = Converter::as_float(eps_bits);
        vFloat rc = recip_pos(s + e);
        for (int i = 0; i < N; ++i) {
            dst_reg[LOGIT0 + i * N + j] = dst_reg[LOGIT0 + i * N + j] * rc;
        }
    }
}

// Row normalisation: m[i][j] *= 1 / (sum_j m[i][j] + eps).
sfpi_inline void row_norm(uint32_t eps_bits) {
#pragma GCC unroll 0
    for (int i = 0; i < N; ++i) {
        vFloat s = dst_reg[LOGIT0 + i * N];
        for (int j = 1; j < N; ++j) {
            s = s + dst_reg[LOGIT0 + i * N + j];
        }
        vFloat e = Converter::as_float(eps_bits);
        vFloat rr = recip_pos(s + e);
        for (int j = 0; j < N; ++j) {
            dst_reg[LOGIT0 + i * N + j] = dst_reg[LOGIT0 + i * N + j] * rr;
        }
    }
}

// sinkhorn_block on one coefficient-major tile (DEST tile 0): logits -> comb, all iterations in DEST.
void sinkhorn(uint32_t eps_bits, uint32_t iters) {
    // m = softmax_j(L) + eps, row max subtracted (overflow-safe).
#pragma GCC unroll 0
    for (int i = 0; i < N; ++i) {
        const int row = LOGIT0 + i * N;
        {
            vFloat mx = dst_reg[row];
            for (int j = 1; j < N; ++j) {
                vFloat t = dst_reg[row + j];
                v_if(t > mx) { mx = t; }
                v_endif;
            }
            dst_reg[SCRATCH0] = mx;
            dst_reg[SCRATCH1] = 0.0f;
        }
        for (int j = 0; j < N; ++j) {
            vFloat lv = dst_reg[row + j];
            vFloat mxv = dst_reg[SCRATCH0];
            vFloat d = lv - mxv;
            vFloat ex = ckernel::sfpu::_sfpu_exp_fp32_accurate_<false>(d);
            dst_reg[row + j] = ex;
            dst_reg[SCRATCH1] = dst_reg[SCRATCH1] + ex;
        }
        {
            vFloat rs = recip_pos(dst_reg[SCRATCH1]);
            vFloat e = Converter::as_float(eps_bits);
            for (int j = 0; j < N; ++j) {
                dst_reg[row + j] = dst_reg[row + j] * rs + e;
            }
        }
    }
    col_norm(eps_bits);
#pragma GCC unroll 0
    for (uint32_t it = 1; it < iters; ++it) {
        row_norm(eps_bits);
        col_norm(eps_bits);
    }
}

}  // namespace mhc_sfpu
#endif

// One-time SFPU init for the custom block ops (config reg + addr mods + counters); no programmable
// constants are used by the custom ops.
ALWI void custom_sfpu_init() { MATH((ckernel::llk_math_eltwise_unary_sfpu_init<SfpuType::unused, DST_ACCUM_MODE>())); }

// combine_block (root): S[j] = sum_rank gathered[rank][j], fixed rank order, fp32 SFPU adds.
ALWI void combine_block() {
    constexpr uint32_t slot_tiles = 2 * block_token_tiles;
    cb_wait_front(cb_gathered, group_cores * slot_tiles);
    cb_reserve_back(cb_combined, slot_tiles);
    reconfig_data_format_srca(cb_gathered);
    pack_reconfig_data_format(cb_combined);
    copy_tile_to_dst_init_short(cb_gathered);
    add_binary_tile_init();
    for (uint32_t j = 0; j < slot_tiles; ++j) {
        tile_regs_acquire();
        copy_tile(cb_gathered, j, 0);
        for (uint32_t r = 1; r < group_cores; ++r) {
            copy_tile(cb_gathered, r * slot_tiles + j, 1);
            add_binary_tile(0, 1, 0);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_combined);
        tile_regs_release();
    }
    cb_push_back(cb_combined, slot_tiles);
    cb_pop_front(cb_gathered, group_cores * slot_tiles);
}

void kernel_main() {
    const uint32_t num_blocks = get_arg_val<uint32_t>(0);
    const uint32_t core_token_tiles = get_arg_val<uint32_t>(1);
    const uint32_t core_c_tiles = get_arg_val<uint32_t>(2);
    const uint32_t rank = get_arg_val<uint32_t>(3);
    const uint32_t a_pre_bits = get_arg_val<uint32_t>(4);
    const uint32_t a_post_bits = get_arg_val<uint32_t>(5);
    const uint32_t a_res_bits = get_arg_val<uint32_t>(6);
    const uint32_t eps_bits = get_arg_val<uint32_t>(7);
    const uint32_t norm_eps_bits = get_arg_val<uint32_t>(8);
    const uint32_t inv_nc_bits = get_arg_val<uint32_t>(9);
    const uint32_t sinkhorn_iters = get_arg_val<uint32_t>(10);

    using namespace compute_kernel_lib;

    const uint32_t core_k_tiles = n_streams * core_c_tiles;
    constexpr uint32_t x_block_pages = block_token_tiles * core_k_tiles_max;  // nominal (matches reader)

    CircularBuffer x_buf(cb_x_resident);
    CircularBuffer w_buf(cb_weight);
    CircularBuffer partial_buf(cb_partial);

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_x_resident, cb_weight, cb_partial);

    // Resident constant: the bias tile is waited once and never popped.
    cb_wait_front(cb_bias_coef, 1);

    for (uint32_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
        const uint32_t row0 = block_idx * block_token_tiles;
        const uint32_t extent =
            (core_token_tiles - row0) < block_token_tiles ? (core_token_tiles - row0) : block_token_tiles;

        // ---- project_block: mix partial = X_blk @ W_slice (out subblock height = largest divisor <= DEST) ----
        uint32_t sb_h = 1;
        for (uint32_t h = DEST_AUTO_LIMIT; h > 1; --h) {
            if (extent % h == 0) {
                sb_h = h;
                break;
            }
        }
        matmul_block<
            false,
            false,
            LastBlockTarget::Out,
            OutputCBLayout::SubblockMajor,
            matmul_config::InitMode::Short,
            InputPolicy::WaitAndRetainOnLastBlock,
            InputPolicy::WaitAndRetainOnLastBlock>(
            x_buf, w_buf, partial_buf, partial_buf, MatmulBlockShape::of(extent / sb_h, 1, sb_h, 1, core_k_tiles, 1));

        // ---- sumsq_block: per row, Q = sum_k x_k*x_k (DEST-accumulated), then fp32 row-collapse ----
        for (uint32_t t = 0; t < extent; ++t) {
            const uint32_t base = t * core_k_tiles;
            eltwise_chain(
                IterationShape::tiles(core_k_tiles),
                BinaryFpu<
                    BinaryFpuOp::Mul,
                    input(
                        cb_x_resident,
                        WaitPolicy::None,
                        PopPolicy::None,
                        InputTileMapping::Block,
                        DataFormatReconfig::Enabled,
                        TileAddressing::Offset),
                    input(
                        cb_x_resident,
                        WaitPolicy::None,
                        PopPolicy::None,
                        InputTileMapping::Block,
                        DataFormatReconfig::Enabled,
                        TileAddressing::Offset),
                    Dst::D0,
                    DestAccumulation::WholeShape>{base, base},
                PackTile<output(
                    cb_sq_acc,
                    ReservePolicy::OneUpfront,
                    PushPolicy::OneAtEnd,
                    DataFormatReconfig::Enabled,
                    TileAddressing::Direct,
                    DestAccumulation::WholeShape)>{});
            reduce<
                PoolType::SUM,
                ReduceDim::REDUCE_ROW,
                cb_sq_acc,
                cb_reduce_scaler,
                cb_partial,
                ReduceInputPolicy::WaitAndPopPerTile,
                ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
                ReduceFp32Mode::Accurate>(ReduceInputBlockShape::single());
        }

        // ---- combine_block (root only) ----
        if (rank == 0) {
            combine_block();
        }

        // ---- coefficients_block ----
        cb_wait_front(cb_coef_in, extent);
        cb_reserve_back(cb_coef_out, extent);
        reconfig_data_format_srca(cb_coef_in);
        pack_reconfig_data_format(cb_coef_out);
        copy_tile_to_dst_init_short(cb_coef_in);
        custom_sfpu_init();
        for (uint32_t t = 0; t < extent; ++t) {
            const bool owned = ((row0 + t) % group_cores) == rank;
            if (owned) {
                cb_reserve_back(cb_logits_coef, 1);
            }
            tile_regs_acquire();
            copy_tile(cb_coef_in, t, 0);
            copy_tile(cb_bias_coef, 0, 1);
            MATH((_llk_math_eltwise_unary_sfpu_params_(
                mhc_sfpu::coefficients,
                0,
                VectorMode::None,
                a_pre_bits,
                a_post_bits,
                a_res_bits,
                eps_bits,
                norm_eps_bits,
                inv_nc_bits)));
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_coef_out);
            if (owned) {
                pack_tile(0, cb_logits_coef);
            }
            tile_regs_release();
            if (owned) {
                cb_push_back(cb_logits_coef, 1);
            }
        }
        cb_push_back(cb_coef_out, extent);
        cb_pop_front(cb_coef_in, extent);

        // ---- ymix_block: y[c] = sum_i x[i][c] * bcast_col(pre_i), n-deep DEST accumulation per output ----
        cb_wait_front(cb_pre_cols, n_streams * extent);
        for (uint32_t t = 0; t < extent; ++t) {
            eltwise_chain(
                IterationShape::grid(core_c_tiles, n_streams),
                BinaryFpu<
                    BinaryFpuOp::Mul,
                    input(
                        cb_x_resident,
                        WaitPolicy::None,
                        PopPolicy::None,
                        InputTileMapping::Block,
                        DataFormatReconfig::Enabled,
                        TileAddressing::Offset),
                    input(
                        input(
                            cb_pre_cols,
                            WaitPolicy::None,
                            PopPolicy::None,
                            InputTileMapping::Row,
                            DataFormatReconfig::Enabled,
                            TileAddressing::Offset),
                        BroadcastDim::Col),
                    Dst::D0,
                    DestAccumulation::PerRow>{t * core_k_tiles, t * n_streams},
                PackTile<output(
                    cb_y_out,
                    ReservePolicy::PerOuter,
                    PushPolicy::PerOuter,
                    DataFormatReconfig::Enabled,
                    TileAddressing::Direct,
                    DestAccumulation::PerRow)>{});
        }
        cb_pop_front(cb_pre_cols, n_streams * extent);
        cb_pop_front(cb_x_resident, x_block_pages);  // X block freed: the reader may load block+2

        // ---- sinkhorn_block (owned rows), after the y-mix (stall-shadow reorder) ----
        bool sinkhorn_initialized = false;
        for (uint32_t t = 0; t < extent; ++t) {
            if (((row0 + t) % group_cores) != rank) {
                continue;
            }
            if (!sinkhorn_initialized) {
                reconfig_data_format_srca(cb_logits_coef);
                pack_reconfig_data_format(cb_comb_coef);
                copy_tile_to_dst_init_short(cb_logits_coef);
                custom_sfpu_init();
                sinkhorn_initialized = true;
            }
            cb_wait_front(cb_logits_coef, 1);
            cb_reserve_back(cb_comb_coef, 1);
            tile_regs_acquire();
            copy_tile(cb_logits_coef, 0, 0);
            MATH((_llk_math_eltwise_unary_sfpu_params_(
                mhc_sfpu::sinkhorn, 0, VectorMode::None, eps_bits, sinkhorn_iters)));
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_comb_coef);
            tile_regs_release();
            cb_push_back(cb_comb_coef, 1);
            cb_pop_front(cb_logits_coef, 1);
        }
    }
}
