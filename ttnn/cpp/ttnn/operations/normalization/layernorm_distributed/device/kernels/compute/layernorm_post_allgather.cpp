
// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

/*
 * This kernel computes layernorm or rmsnorm, dependent on the RMSNORM define.
 * For layernorm it receives E(x**2) and E(x) and computes the remaining normalization based on gamma, beta and epsilon.
 *   E(x**2) and E(x) are contained in a two tile wide tensor containing E(x**2) and E(x) in the left most columns per
 * tile. For rmsnorm it receives E(x**2) and computes the remaining normalization based on gamma, beta and epsilon.
 *   E(x**2) is contained in a one tile wide tensor containing E(x**2) in the left most column.
 */

#include <cstdint>

#define BCAST_LLKOP EltwiseBinaryType::ELWMUL
#define BCAST_DIM BroadcastType::COL

#include "api/compute/reduce.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/layernorm.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_math.hpp"

namespace ckl = compute_kernel_lib;

constexpr uint32_t cb_inp = tt::CBIndex::c_0;
constexpr uint32_t cb_stats = tt::CBIndex::c_1;
constexpr uint32_t cb_eps = tt::CBIndex::c_4;
constexpr uint32_t cb_out = tt::CBIndex::c_14;
constexpr uint32_t cb_stats_reduced = tt::CBIndex::c_6;    // [E(x**2), E(x)]
constexpr uint32_t cb_recip_sqrt_var = tt::CBIndex::c_10;  // 1/sqrt(var+eps)
constexpr uint32_t cb_x_normed = tt::CBIndex::c_12;        // (x - E(x)) * 1/sqrt(var+eps) or x * 1/sqrt(E(x**2) + eps)
constexpr uint32_t cb_x_minus_mean = tt::CBIndex::c_11;    // x - E(x)
constexpr uint32_t cb_norm_x_input = cb_x_minus_mean;
constexpr uint32_t stats_tile_stride = 2;
constexpr uint32_t do_gamma = get_compile_time_arg_val(3);
constexpr uint32_t do_beta = get_compile_time_arg_val(4);
constexpr uint32_t normed_output_cb =
    do_gamma || do_beta ? cb_x_normed : cb_out;  // (x - E(x)) * 1/sqrt(var+eps) or x * 1/sqrt(E(x**2) + eps)
constexpr uint32_t cb_gamma = tt::CBIndex::c_2;
constexpr uint32_t cb_length = get_compile_time_arg_val(8);
constexpr uint32_t Wt = get_compile_time_arg_val(0);
constexpr uint32_t cb_times_gamma_out = do_beta ? tt::CBIndex::c_13 : cb_out;
constexpr uint32_t cb_beta = tt::CBIndex::c_3;

ALWI void normalize_chunk(const uint32_t num_tiles) {
    const auto shape = ckl::EltwiseShape::tiles(num_tiles, ckl::DEST_AUTO_LIMIT);
    constexpr auto gamma_beta_lifecycle =
        Wt == cb_length ? ckl::InputLifecycle::HeldCumulative : ckl::InputLifecycle::Chunked;

    ckl::eltwise_chain(
        shape,
        ckl::BinaryFpu<
            ckl::input(cb_inp, ckl::InputLifecycle::Chunked, ckl::OperandKind::Block),
            ckl::input(
                cb_stats_reduced,
                ckl::InputLifecycle::HeldBulk,
                ckl::OperandKind::Scalar,
                ckl::DataFormatReconfig::Enabled,
                ckl::TileOffset::Set),
            ckl::BinaryFpuOp::Sub,
            ckl::BroadcastDim::Col>{0u, 1u},
        ckl::PackTile<ckl::output(cb_x_minus_mean, ckl::OutputLifecycle::Chunked)>{});

    ckl::mul<
        ckl::input(cb_norm_x_input, ckl::InputLifecycle::Chunked, ckl::OperandKind::Block),
        ckl::input(cb_recip_sqrt_var, ckl::InputLifecycle::HeldBulk),
        ckl::output(normed_output_cb, ckl::OutputLifecycle::Chunked),
        ckl::BroadcastDim::Col>(shape);

    if constexpr (do_gamma) {
        ckl::mul<
            ckl::input(cb_x_normed, ckl::InputLifecycle::Chunked, ckl::OperandKind::Block),
            ckl::input(cb_gamma, gamma_beta_lifecycle, ckl::OperandKind::Block),
            ckl::output(cb_times_gamma_out, ckl::OutputLifecycle::Chunked),
            ckl::BroadcastDim::Row>(shape);
    }
    if constexpr (do_beta) {
        constexpr uint32_t cb_beta_input = do_gamma ? cb_times_gamma_out : normed_output_cb;
        ckl::add<
            ckl::input(cb_beta_input, ckl::InputLifecycle::Chunked, ckl::OperandKind::Block),
            ckl::input(cb_beta, gamma_beta_lifecycle, ckl::OperandKind::Block),
            ckl::output(cb_out, ckl::OutputLifecycle::Chunked),
            ckl::BroadcastDim::Row>(shape);
    }
}

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

    constexpr uint32_t cb_var_eps = tt::CBIndex::c_9;  // var + epsilon (or E(x**2) + epsilon)

    constexpr uint32_t cb_var = tt::CBIndex::c_8;  // E(x**2) - E(x)**2 or E(x**2)

    // Layernorm-specific CBs
    constexpr uint32_t cb_mean_squared = tt::CBIndex::c_7;  // E(x)**2

    compute_kernel_hw_startup(cb_inp, cb_inp, cb_stats_reduced);

    CircularBuffer cb_reduce_obj(cb_reduce);
    CircularBuffer cb_eps_obj(cb_eps);
    CircularBuffer cb_stats_obj(cb_stats);
    CircularBuffer cb_stats_reduced_obj(cb_stats_reduced);
    CircularBuffer cb_recip_sqrt_var_obj(cb_recip_sqrt_var);

    cb_reduce_obj.wait_front(1);  // comes from the reader
    cb_eps_obj.wait_front(1);     // comes from the reader

    for (uint32_t ncht = 0; ncht < NCHt; ncht++) {
        constexpr int onetile = 1;
        constexpr int dst0 = 0;

        reconfig_data_format(cb_reduce, cb_stats);
        pack_reconfig_data_format(cb_stats_reduced);

        /*
         * Reduce stats input.
         * cb_stats = [sum(x0**2), sum(x0), sum(x1**2), sum(x1), ...]
         * RMSNorm packs mean(x**2) into cb_var. Layernorm just uses cb_stats_reduced.
         */
        reduce_init<PoolType::AVG, ReduceDim::REDUCE_ROW>(cb_stats, cb_reduce, cb_stats_reduced);
        cb_stats_obj.wait_front(stats_tiles_cols);

        tile_regs_acquire();
        // Reduce sum(x**2) first
        for (uint32_t i = 0; i < stats_tiles_cols; i += stats_tile_stride) {
            reduce_tile<PoolType::AVG, ReduceDim::REDUCE_ROW>(cb_stats, cb_reduce, i, 0, 0);
        }
        // Reduce sum(x) next
        for (uint32_t i = 1; i < stats_tiles_cols; i += stats_tile_stride) {
            reduce_tile<PoolType::AVG, ReduceDim::REDUCE_ROW>(cb_stats, cb_reduce, i, 0, 1);
        }
        tile_regs_commit();

        cb_stats_obj.pop_front(stats_tiles_cols);
        cb_stats_reduced_obj.reserve_back(stats_tile_stride);

        tile_regs_wait();
        pack_tile(0, cb_stats_reduced);
        pack_tile(1, cb_stats_reduced);
        tile_regs_release();

        cb_stats_reduced_obj.push_back(stats_tile_stride);

        reduce_uninit();

        cb_stats_reduced_obj.wait_front(stats_tile_stride);

        // E[x]**2
        ckl::eltwise_chain(
            ckl::EltwiseShape::single(),
            ckl::BinaryFpu<
                ckl::input(
                    cb_stats_reduced,
                    ckl::InputLifecycle::HeldBulk,
                    ckl::OperandKind::Scalar,
                    ckl::DataFormatReconfig::Enabled,
                    ckl::TileOffset::Set),
                ckl::input(
                    cb_stats_reduced,
                    ckl::InputLifecycle::HeldBulk,
                    ckl::OperandKind::Scalar,
                    ckl::DataFormatReconfig::Enabled,
                    ckl::TileOffset::Set),
                ckl::BinaryFpuOp::Mul,
                ckl::BroadcastDim::None>{1, 1},
            ckl::PackTile<ckl::output(cb_mean_squared, ckl::OutputLifecycle::Bulk)>{});

        // E[x**2] - E[x]**2
        ckl::sub<
            ckl::input(cb_stats_reduced, ckl::InputLifecycle::HeldBulk),
            ckl::input(cb_mean_squared, ckl::InputLifecycle::Bulk),
            ckl::output(cb_var, ckl::OutputLifecycle::Bulk),
            ckl::BroadcastDim::None>(ckl::EltwiseShape::single());

        // 1/sqrt(var + eps)
        ckl::eltwise_chain(
            ckl::EltwiseShape::single(),
            ckl::BinaryFpu<
                ckl::input(cb_var),
                ckl::input(cb_eps, ckl::InputLifecycle::CallerManaged),
                ckl::BinaryFpuOp::Add,
                ckl::BroadcastDim::None>{},
            ckl::Rsqrt<ckl::Approx::Exact, LEGACY_RSQRT ? ckl::Legacy::On : ckl::Legacy::Off, ckl::Dst::D0>{},
            ckl::PackTile<ckl::output(cb_recip_sqrt_var)>{});

        constexpr uint32_t chunk_iterations = Wt / cb_length;
        constexpr uint32_t leftover_tiles = Wt % cb_length;
        for (uint32_t chunk = 0; chunk < chunk_iterations; ++chunk) {
            normalize_chunk(cb_length);
        }
        if constexpr (leftover_tiles > 0) {
            normalize_chunk(leftover_tiles);
        }

        cb_stats_reduced_obj.pop_front(stats_tile_stride);
        cb_recip_sqrt_var_obj.pop_front(1);
    }
    cb_eps_obj.pop_front(1);
    cb_reduce_obj.pop_front(1);
}
