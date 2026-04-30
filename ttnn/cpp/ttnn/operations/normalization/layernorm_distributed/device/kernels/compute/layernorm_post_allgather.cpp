
// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// PARTIAL MIGRATION: reduce_init/reduce_tile/reduce_uninit + chain_llk<>
// stages stay raw (multi-tile reduction + custom node-driven chain). Inner
// FPU binary single-DEST stages (E[x]², var, var+eps, rsqrt) migrate via
// V2 helper with per-stage BinaryFpuMacroOp / single-CopyTile chains.

#include <cstdint>

#define BCAST_LLKOP EltwiseBinaryType::ELWMUL
#define BCAST_DIM BroadcastType::COL

#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/reduce.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/layernorm.h"
#include "chain_llk.hpp"

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_math.hpp"

ALWI void ACQ() { acquire_dst(); }
ALWI void REL() { release_dst(); }

constexpr uint32_t cb_inp = tt::CBIndex::c_0;
constexpr uint32_t cb_stats = tt::CBIndex::c_1;

constexpr uint32_t cb_eps = tt::CBIndex::c_4;

constexpr uint32_t cb_out = tt::CBIndex::c_14;

constexpr uint32_t cb_stats_reduced = tt::CBIndex::c_6;
constexpr uint32_t cb_recip_sqrt_var = tt::CBIndex::c_10;
constexpr uint32_t cb_x_normed = tt::CBIndex::c_12;

constexpr uint32_t cb_x_minus_mean = tt::CBIndex::c_11;

constexpr uint32_t cb_norm_x_input = cb_x_minus_mean;
constexpr uint32_t stats_tile_stride = 2;

struct x_minus_mean_node {
    static constexpr LLK_Node node{
        .llk_init = sub_bcast_cols_init_short,
        .llk = FN_compute(sub_tiles_bcast_cols),
        .CB_A = cb_inp,
        .CB_B = cb_stats_reduced,
        .CB_OUT = cb_x_minus_mean,
        .fixed_CB_B_index = 1,
        .fixed_dest_reg = 0xFFFF,
    };
};
constexpr uint32_t do_gamma = get_compile_time_arg_val(3);
constexpr uint32_t do_beta = get_compile_time_arg_val(4);
constexpr uint32_t normed_output_cb = do_gamma || do_beta ? cb_x_normed : cb_out;
struct normed_output_node {
    static constexpr LLK_Node node{
        .llk_init = mul_bcast_cols_init_short,
        .llk = FN_compute(mul_tiles_bcast_cols),
        .CB_A = cb_norm_x_input,
        .CB_B = cb_recip_sqrt_var,
        .CB_OUT = normed_output_cb,
        .fixed_CB_B_index = 0,
        .fixed_dest_reg = 0xFFFF,
    };
};
constexpr uint32_t cb_gamma = tt::CBIndex::c_2;
constexpr uint32_t cb_length = get_compile_time_arg_val(8);
constexpr uint32_t Wt = get_compile_time_arg_val(0);
constexpr uint32_t pop_gamma_beta = Wt == cb_length ? 0xDDDD : 0xFFFF;

constexpr uint32_t cb_times_gamma_out = do_beta ? tt::CBIndex::c_13 : cb_out;
struct gamma_optional_node {
    static constexpr LLK_Node node{
        .llk_init = mul_bcast_rows_init_short,
        .llk = FN_compute(mul_tiles_bcast_rows),
        .CB_A = cb_x_normed,
        .CB_B = cb_gamma,
        .CB_OUT = cb_times_gamma_out,
        .fixed_CB_B_index = pop_gamma_beta,
        .fixed_dest_reg = 0xFFFF,
    };
};
constexpr uint32_t cb_in_beta = do_gamma ? cb_times_gamma_out : normed_output_cb;
constexpr uint32_t cb_beta = tt::CBIndex::c_3;
struct beta_optional_node {
    static constexpr LLK_Node node{
        .llk_init = add_bcast_rows_init_short,
        .llk = FN_compute(add_tiles_bcast_rows),
        .CB_A = cb_in_beta,
        .CB_B = cb_beta,
        .CB_OUT = cb_out,
        .fixed_CB_B_index = pop_gamma_beta,
        .fixed_dest_reg = 0xFFFF,
    };
};

namespace {
template <tt::CBIndex CbA, tt::CBIndex CbB, uint32_t TileA, uint32_t TileB>
struct MulTilesAt : compute_kernel_lib::UnaryOp<MulTilesAt<CbA, CbB, TileA, TileB>, compute_kernel_lib::Dst::D0> {
    static constexpr bool clobbers_sfpu_lut = false;
    static constexpr bool clashes_with_fpu = true;
    ALWI static void init() { mul_tiles_init(CbA, CbB); }
    ALWI static void call(uint32_t dst) { mul_tiles(CbA, CbB, TileA, TileB, dst); }
};
template <tt::CBIndex CbA, tt::CBIndex CbB, uint32_t TileA, uint32_t TileB>
struct SubTilesAt : compute_kernel_lib::UnaryOp<SubTilesAt<CbA, CbB, TileA, TileB>, compute_kernel_lib::Dst::D0> {
    static constexpr bool clobbers_sfpu_lut = false;
    static constexpr bool clashes_with_fpu = true;
    ALWI static void init() { sub_tiles_init(CbA, CbB); }
    ALWI static void call(uint32_t dst) { sub_tiles(CbA, CbB, TileA, TileB, dst); }
};
template <tt::CBIndex CbA, tt::CBIndex CbB, uint32_t TileA, uint32_t TileB>
struct AddTilesAt : compute_kernel_lib::UnaryOp<AddTilesAt<CbA, CbB, TileA, TileB>, compute_kernel_lib::Dst::D0> {
    static constexpr bool clobbers_sfpu_lut = false;
    static constexpr bool clashes_with_fpu = true;
    ALWI static void init() { add_tiles_init(CbA, CbB); }
    ALWI static void call(uint32_t dst) { add_tiles(CbA, CbB, TileA, TileB, dst); }
};

template <bool LegacyRsqrt>
struct RsqrtOp : compute_kernel_lib::UnaryOp<RsqrtOp<LegacyRsqrt>, compute_kernel_lib::Dst::D0> {
    static constexpr bool clobbers_sfpu_lut = true;
    ALWI static void init() { rsqrt_tile_init<LegacyRsqrt>(); }
    ALWI static void call(uint32_t dst) { rsqrt_tile<LegacyRsqrt>(dst); }
};
}  // namespace

void kernel_main() {
    uint32_t NCHt = get_arg_val<uint32_t>(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    constexpr uint32_t blk = get_compile_time_arg_val(1);
    constexpr uint32_t stats_tiles_cols = get_compile_time_arg_val(2);
    constexpr uint32_t do_gamma = get_compile_time_arg_val(3);
    constexpr uint32_t do_beta = get_compile_time_arg_val(4);
    constexpr bool FLOAT32_DTYPE = get_compile_time_arg_val(5) == 1;
    constexpr bool FLOAT32_REDUCTION = get_compile_time_arg_val(6) == 1;
    constexpr bool LEGACY_RSQRT = get_compile_time_arg_val(7) == 1;
    constexpr uint32_t cb_length = get_compile_time_arg_val(8);

    constexpr uint32_t onetile = 1;

    constexpr uint32_t cb_reduce = tt::CBIndex::c_5;
    constexpr uint32_t cb_var_eps = tt::CBIndex::c_9;
    constexpr uint32_t cb_var = tt::CBIndex::c_8;
    constexpr uint32_t cb_mean_squared = tt::CBIndex::c_7;

    binary_op_init_common(cb_inp, cb_inp, cb_stats_reduced);

    cb_wait_front(cb_reduce, 1);
    cb_wait_front(cb_eps, 1);

    using compute_kernel_lib::eltwise_chain;
    using compute_kernel_lib::eltwise_pipeline;

    for (uint32_t ncht = 0; ncht < NCHt; ncht++) {
        constexpr int onetile = 1;
        constexpr int dst0 = 0;

        reconfig_data_format(cb_stats, cb_reduce);
        pack_reconfig_data_format(cb_stats_reduced);

        // ----- Reduce stage stays raw (multi-tile DEST + custom indices) -----
        reduce_init<PoolType::AVG, ReduceDim::REDUCE_ROW, FLOAT32_REDUCTION>(cb_stats, cb_reduce, cb_stats_reduced);
        cb_wait_front(cb_stats, stats_tiles_cols);
        cb_reserve_back(cb_stats_reduced, stats_tile_stride);

        ACQ();
        for (uint32_t i = 0; i < stats_tiles_cols; i += stats_tile_stride) {
            reduce_tile<PoolType::AVG, ReduceDim::REDUCE_ROW, FLOAT32_REDUCTION>(cb_stats, cb_reduce, i, 0, 0);
        }
        pack_tile(0, cb_stats_reduced);
        for (uint32_t i = 1; i < stats_tiles_cols; i += stats_tile_stride) {
            reduce_tile<PoolType::AVG, ReduceDim::REDUCE_ROW, FLOAT32_REDUCTION>(cb_stats, cb_reduce, i, 0, 1);
        }
        pack_tile(1, cb_stats_reduced);
        REL();
        cb_push_back(cb_stats_reduced, stats_tile_stride);
        cb_pop_front(cb_stats, stats_tiles_cols);
        reduce_uninit();

        // ----- E[x]² via V2 helper (mul_tiles at tile index 1) -----
        reconfig_data_format(cb_stats_reduced, cb_stats_reduced);
        pack_reconfig_data_format(cb_mean_squared);
        cb_wait_front(cb_stats_reduced, stats_tile_stride);
        eltwise_pipeline<cb_mean_squared>(
            onetile, eltwise_chain(MulTilesAt<tt::CBIndex::c_6, tt::CBIndex::c_6, 1, 1>{}));

        // ----- E[x²] - E[x]² via V2 helper -----
        reconfig_data_format(cb_stats_reduced, cb_mean_squared);
        pack_reconfig_data_format(cb_var);
        cb_wait_front(cb_mean_squared, 1);
        eltwise_pipeline<cb_var>(onetile, eltwise_chain(SubTilesAt<tt::CBIndex::c_6, tt::CBIndex::c_7, 0, 0>{}));
        cb_pop_front(cb_mean_squared, 1);

        // ----- 1/sqrt(var + eps) via V2 helper (Add + Rsqrt chain) -----
        cb_wait_front(cb_var, 1);
        reconfig_data_format(cb_var, cb_eps);
        pack_reconfig_data_format(cb_recip_sqrt_var);
        eltwise_pipeline<cb_recip_sqrt_var>(
            onetile, eltwise_chain(AddTilesAt<tt::CBIndex::c_8, tt::CBIndex::c_4, 0, 0>{}, RsqrtOp<LEGACY_RSQRT>{}));
        cb_pop_front(cb_var, 1);

        // ----- Final post-rsqrt stages stay on chain_llk<> (custom multi-node) -----
        if constexpr (do_gamma && do_beta) {
            chain_llk<Wt, cb_length, true>(
                x_minus_mean_node{}, normed_output_node{}, gamma_optional_node{}, beta_optional_node{});
        } else if (do_gamma) {
            chain_llk<Wt, cb_length, true>(x_minus_mean_node{}, normed_output_node{}, gamma_optional_node{});
        } else if (do_beta) {
            chain_llk<Wt, cb_length, true>(x_minus_mean_node{}, normed_output_node{}, beta_optional_node{});
        } else {
            chain_llk<Wt, cb_length, true>(x_minus_mean_node{}, normed_output_node{});
        }

        cb_pop_front(cb_stats_reduced, stats_tile_stride);
        cb_pop_front(cb_recip_sqrt_var, 1);
    }
    cb_pop_front(cb_eps, 1);
    cb_pop_front(cb_reduce, 1);
}
