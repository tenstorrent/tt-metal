// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Compute kernel for groupnorm_sc_N_1_HW_C (single-core GroupNorm, (N,1,HW,C)).
//
// Per (n, g) group — three streaming passes over the Ht x Wg slab:
//   Pass 1 (mean):   reduce<SUM, REDUCE_SCALAR> with scaler 1/sqrt(HW*Cg)
//                    (REDUCE_SCALAR applies the scaler twice -> SUM/N = mean)
//                    -> cb_mean (1 tile, valid at (0,0))
//   Pass 2 (var):    per tile row b: sub<Scalar>(x - mean) -> square ->
//                    accumulate_reduce_block -> cb_var (1 tile after b = Ht-1)
//   rstd:            transform_in_place(cb_var): +eps, rsqrt
//   Pass 3 (norm):   per tile row b: sub<Scalar> -> mul<Scalar rstd> ->
//                    [mul<Row gamma + g*Wg>] -> [add<Row beta + g*Wg>] -> cb_output_tiles
//
// cb_mean / cb_var persist across passes (HeldBulk operands), popped at group
// end. gamma / beta (Wt tiles) persist for the whole kernel, popped at exit.
// All work goes through kernel-lib helpers — no raw tile_regs / pack loops.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/streaming_reduce_helpers.hpp"

namespace {
constexpr uint32_t cb_input_tiles = 0;
constexpr uint32_t cb_gamma_tiles = 1;
constexpr uint32_t cb_beta_tiles = 2;
constexpr uint32_t cb_scaler = 8;
constexpr uint32_t cb_output_tiles = 16;
constexpr uint32_t cb_mean = 24;
constexpr uint32_t cb_var = 25;
constexpr uint32_t cb_centered = 26;
constexpr uint32_t cb_xhat = 27;
constexpr uint32_t cb_scaled = 28;
}  // namespace

namespace ckl = compute_kernel_lib;

void kernel_main() {
    constexpr uint32_t Ht = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);
    constexpr uint32_t Wg = get_compile_time_arg_val(2);
    constexpr uint32_t G = get_compile_time_arg_val(3);
    constexpr uint32_t N = get_compile_time_arg_val(4);
    constexpr bool HAS_GAMMA = get_compile_time_arg_val(5) != 0;
    constexpr bool HAS_BETA = get_compile_time_arg_val(6) != 0;
    constexpr uint32_t EPS_BITS = get_compile_time_arg_val(7);

    // Stage outputs when affine stages are absent / present.
    constexpr uint32_t cb_norm_out = HAS_GAMMA ? cb_xhat : cb_output_tiles;    // 3b output
    constexpr uint32_t cb_gamma_out = HAS_BETA ? cb_scaled : cb_output_tiles;  // 3c output

    compute_kernel_hw_startup(cb_input_tiles, cb_scaler, cb_output_tiles);

    constexpr auto slab_shape = ckl::ReduceInputBlockShape::of(Ht, Wg);
    constexpr auto row_reduce_shape = ckl::ReduceInputBlockShape::of(1, Wg);
    constexpr auto row_shape = ckl::EltwiseShape::grid(1, Wg);

    for (uint32_t n = 0; n < N; ++n) {
        for (uint32_t g = 0; g < G; ++g) {
            // ---- Pass 1: mean over the slab -> cb_mean (1 tile) ----
            ckl::reduce<ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_SCALAR>(
                cb_input_tiles, cb_scaler, cb_mean, slab_shape);

            // ---- Pass 2: variance via E[(x - mean)^2] ----
            for (uint32_t b = 0; b < Ht; ++b) {
                ckl::sub<
                    cb_input_tiles,
                    cb_mean,
                    cb_centered,
                    ckl::BroadcastDim::Scalar,
                    ckl::InputLifecycle::Streaming,
                    ckl::InputLifecycle::HeldBulk>(row_shape);
                ckl::square<cb_centered, cb_centered>(row_shape);
                ckl::accumulate_reduce_block<ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_SCALAR>(
                    cb_centered, cb_scaler, cb_var, row_reduce_shape, b, Ht);
            }

            // ---- var -> rstd = 1/sqrt(var + eps), in place on cb_var ----
            ckl::transform_in_place(cb_var, [](uint32_t dst) {
                binop_with_scalar_tile_init();
                add_unary_tile(dst, EPS_BITS);
                rsqrt_tile_init();
                rsqrt_tile(dst);
            });

            // ---- Pass 3: normalize + optional affine, per tile row ----
            for (uint32_t b = 0; b < Ht; ++b) {
                ckl::sub<
                    cb_input_tiles,
                    cb_mean,
                    cb_centered,
                    ckl::BroadcastDim::Scalar,
                    ckl::InputLifecycle::Streaming,
                    ckl::InputLifecycle::HeldBulk>(row_shape);
                ckl::mul<
                    cb_centered,
                    cb_var,
                    cb_norm_out,
                    ckl::BroadcastDim::Scalar,
                    ckl::InputLifecycle::Streaming,
                    ckl::InputLifecycle::HeldBulk>(row_shape);

                if constexpr (HAS_GAMMA) {
                    // xhat * gamma[g*Wg + c] — Row broadcast from the persistent
                    // gamma row; TileOffset::Set supplies the per-group base.
                    ckl::eltwise_chain(
                        row_shape,
                        ckl::BinaryFpu<
                            cb_xhat,
                            cb_gamma_tiles,
                            ckl::BinaryFpuOp::Mul,
                            ckl::BroadcastDim::Row,
                            ckl::InputLifecycle::Streaming,
                            ckl::InputLifecycle::HeldBulk,
                            ckl::BinaryDataFormatReconfig::Input,
                            ckl::Dst::D0,
                            ckl::OperandKind::Scalar,
                            ckl::OperandKind::Row,
                            ckl::TileOffset::Unset,
                            ckl::TileOffset::Set>{0, g * Wg},
                        ckl::PackTile<cb_gamma_out>{});
                }
                if constexpr (HAS_BETA) {
                    ckl::eltwise_chain(
                        row_shape,
                        ckl::BinaryFpu<
                            cb_scaled,
                            cb_beta_tiles,
                            ckl::BinaryFpuOp::Add,
                            ckl::BroadcastDim::Row,
                            ckl::InputLifecycle::Streaming,
                            ckl::InputLifecycle::HeldBulk,
                            ckl::BinaryDataFormatReconfig::Input,
                            ckl::Dst::D0,
                            ckl::OperandKind::Scalar,
                            ckl::OperandKind::Row,
                            ckl::TileOffset::Unset,
                            ckl::TileOffset::Set>{0, g * Wg},
                        ckl::PackTile<cb_output_tiles>{});
                }
            }

            // ---- group end: release per-group statistics ----
            cb_pop_front(cb_mean, 1);
            cb_pop_front(cb_var, 1);
        }
    }

    // ---- kernel end: drain persistent operands ----
    cb_pop_front(cb_scaler, 1);
    if constexpr (HAS_GAMMA) {
        cb_pop_front(cb_gamma_tiles, Wt);
    }
    if constexpr (HAS_BETA) {
        cb_pop_front(cb_beta_tiles, Wt);
    }
}
