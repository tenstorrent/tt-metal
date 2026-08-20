// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/kernel/compute/moreh_common.hpp"

namespace ckl = compute_kernel_lib;

#if defined(FP32_DEST_ACC_EN)
constexpr auto kDataFormatReconfig = ckl::DataFormatReconfig::Enabled;
#else
constexpr auto kDataFormatReconfig = ckl::DataFormatReconfig::Disabled;
#endif

// Optional masks are absent from the generated DFB accessor header when not bound. These macros
// therefore remove their complete operation (including the accessor name), rather than merely
// making its predicate false at runtime.
#ifdef DO_MASK_H
#define MAYBE_MASK_H(predicate)                \
    ckl::runtime_if(                           \
        predicate,                             \
        ckl::CopyTile<                         \
            ckl::input(                        \
                dfb::mask_h,                   \
                ckl::WaitPolicy::None,         \
                ckl::PopPolicy::None,          \
                ckl::InputTileMapping::Scalar, \
                kDataFormatReconfig,           \
                ckl::TileAddressing::Offset),  \
            ckl::Dst::D1>{0},                  \
        ckl::Mask<>{}),
#else
#define MAYBE_MASK_H(predicate)
#endif

#ifdef DO_MASK_W
#define MAYBE_MASK_W(predicate)                \
    ckl::runtime_if(                           \
        predicate,                             \
        ckl::CopyTile<                         \
            ckl::input(                        \
                dfb::mask_w,                   \
                ckl::WaitPolicy::None,         \
                ckl::PopPolicy::None,          \
                ckl::InputTileMapping::Scalar, \
                kDataFormatReconfig,           \
                ckl::TileAddressing::Offset),  \
            ckl::Dst::D1>{0},                  \
        ckl::Mask<>{}),
#else
#define MAYBE_MASK_W(predicate)
#endif

void kernel_main() {
    constexpr auto num_cols_per_core = get_arg(args::num_cols_per_core);
    constexpr auto origin_H = get_arg(args::origin_H);
    constexpr auto origin_W = get_arg(args::origin_W);
    constexpr auto NCHt = get_arg(args::NCHt);
    constexpr auto Wt = get_arg(args::Wt);
    constexpr bool is_lastdim_layernorm = get_arg(args::is_lastdim_layernorm) == 1;
    constexpr bool is_groupnorm = get_arg(args::is_groupnorm) == 1;

    DataflowBuffer dfb_scaler_obj(dfb::scaler);
    DataflowBuffer dfb_dycopy_obj(dfb::dycopy);
#ifdef DO_MASK_H
    DataflowBuffer dfb_mask_h_obj(dfb::mask_h);
#endif
#ifdef DO_MASK_W
    DataflowBuffer dfb_mask_w_obj(dfb::mask_w);
#endif

    constexpr uint32_t onetile = 1;
    constexpr uint32_t TILE_H = 32;
    constexpr uint32_t TILE_W = 32;
    constexpr uint32_t origin_Ht = (origin_H + TILE_H - 1) / TILE_H;
    constexpr uint32_t origin_Wt = (origin_W + TILE_W - 1) / TILE_W;
    constexpr uint32_t HtWt = origin_Ht * Wt;

#ifdef GAMMA_GRAD_HAS_VALUE
    constexpr auto dfb_out_init = dfb::dgamma;
#else
    constexpr auto dfb_out_init = dfb::dbeta;
#endif
    compute_kernel_hw_startup(dfb::dy, dfb::dy, dfb_out_init);

    dfb_scaler_obj.wait_front(onetile);
#ifdef DO_MASK_H
    dfb_mask_h_obj.wait_front(onetile);
#endif
#ifdef DO_MASK_W
    dfb_mask_w_obj.wait_front(onetile);
#endif

    uint32_t h_idx;
    uint32_t w_idx;
    for (uint32_t outer_idx = 0; outer_idx < num_cols_per_core; ++outer_idx) {
        for (uint32_t inner_idx = 0; inner_idx < NCHt; ++inner_idx) {
            if (is_groupnorm) {
                h_idx = (inner_idx % HtWt) / Wt;
                w_idx = (inner_idx % HtWt) % Wt;
            } else {
                h_idx = inner_idx;
                w_idx = outer_idx;
            }

            ckl::eltwise_chain(
                ckl::IterationShape::one_tile(),
                ckl::CopyTile<ckl::input(
                    dfb::dy, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, kDataFormatReconfig)>{},
                MAYBE_MASK_H((h_idx + 1) % origin_Ht == 0) MAYBE_MASK_W((w_idx + 1) % origin_Wt == 0)
                    ckl::PackTile<ckl::output(
                        dfb::dycopy, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile, kDataFormatReconfig)>{});

#ifdef BETA_GRAD_HAS_VALUE
            if (inner_idx == 0) {
#ifdef GAMMA_GRAD_HAS_VALUE
                copy_tile_to_dfb<dfb::dycopy, dfb::dyadd>(0, 0);
#else
                copy_tile_to_dfb<dfb::dycopy, dfb::dyadd>(0, 1);
#endif
            } else {
#ifdef GAMMA_GRAD_HAS_VALUE
                add_tiles_to_dfb<dfb::dyadd, dfb::dycopy, dfb::dyadd>(0, 0, 1, 0);
#else
                add_tiles_to_dfb<dfb::dyadd, dfb::dycopy, dfb::dyadd>(0, 0, 1, 1);
#endif
            }
#endif

#ifdef GAMMA_GRAD_HAS_VALUE
            ckl::eltwise_chain(
                ckl::IterationShape::one_tile(),
                ckl::BinaryFpu<
                    ckl::BinaryFpuOp::Sub,
                    ckl::input(dfb::x, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, kDataFormatReconfig),
                    ckl::input(
                        dfb::mean,
                        is_lastdim_layernorm ? ckl::BroadcastDim::Col : ckl::BroadcastDim::Scalar,
                        ckl::WaitPolicy::PerTile,
                        ckl::PopPolicy::PerTile,
                        kDataFormatReconfig)>{},
                MAYBE_MASK_H((h_idx + 1) % origin_Ht == 0) MAYBE_MASK_W((w_idx + 1) % origin_Wt == 0)
                    ckl::PackTile<ckl::output(
                        dfb::xmm, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile, kDataFormatReconfig)>{});

            ckl::mul<
                ckl::input(dfb::xmm, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, kDataFormatReconfig),
                ckl::input(
                    dfb::rstd,
                    is_lastdim_layernorm ? ckl::BroadcastDim::Col : ckl::BroadcastDim::Scalar,
                    ckl::WaitPolicy::PerTile,
                    ckl::PopPolicy::PerTile,
                    kDataFormatReconfig),
                ckl::output(dfb::y, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile, kDataFormatReconfig)>(
                ckl::IterationShape::one_tile());

#ifdef BETA_GRAD_HAS_VALUE
            mul_tiles_to_dfb<dfb::y, dfb::dycopy, dfb::ydy>(0, 0, 1, 0);
#else
            mul_tiles_to_dfb<dfb::y, dfb::dycopy, dfb::ydy>(0, 0, 1, 1);
#endif
            if (inner_idx == 0) {
                copy_tile_to_dfb<dfb::ydy, dfb::ydyadd>();
            } else {
                add_tiles_to_dfb<dfb::ydyadd, dfb::ydy, dfb::ydyadd>();
            }
#endif

#if defined(GAMMA_GRAD_HAS_VALUE) && defined(BETA_GRAD_HAS_VALUE)
            dfb_dycopy_obj.pop_front(onetile);
#endif
        }

#ifdef GAMMA_GRAD_HAS_VALUE
        if (is_lastdim_layernorm || is_groupnorm) {
            compute_kernel_lib::reduce<REDUCE_OP, REDUCE_DIM, dfb::ydyadd, dfb::scaler, dfb::dgamma>(
                compute_kernel_lib::ReduceInputBlockShape::single());
        } else {
            copy_tile_to_dfb<dfb::ydyadd, dfb::dgamma>();
        }
#endif
#ifdef BETA_GRAD_HAS_VALUE
        if (is_lastdim_layernorm || is_groupnorm) {
            compute_kernel_lib::reduce<REDUCE_OP, REDUCE_DIM, dfb::dyadd, dfb::scaler, dfb::dbeta>(
                compute_kernel_lib::ReduceInputBlockShape::single());
        } else {
            copy_tile_to_dfb<dfb::dyadd, dfb::dbeta>();
        }
#endif
    }

    dfb_scaler_obj.pop_front(onetile);
#ifdef DO_MASK_H
    dfb_mask_h_obj.pop_front(onetile);
#endif
#ifdef DO_MASK_W
    dfb_mask_w_obj.pop_front(onetile);
#endif
}

#undef MAYBE_MASK_H
#undef MAYBE_MASK_W
