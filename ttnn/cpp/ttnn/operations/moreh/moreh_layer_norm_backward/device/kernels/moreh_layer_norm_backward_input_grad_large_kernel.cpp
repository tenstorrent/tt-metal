// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Shared compute kernel: bound by moreh_layer_norm_backward's and moreh_group_norm_backward's
// input_grad factories, on the large-algorithm path. Both bind the same resource names, so a change
// to this kernel's binding vocabulary or argument schema has to land on both factories together.

#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"  // add/sub/mul
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/kernel/compute/moreh_common.hpp"

namespace ckl = compute_kernel_lib;

#if defined(FP32_DEST_ACC_EN)
constexpr auto kDataFormatReconfig = ckl::DataFormatReconfig::Enabled;
#else
constexpr auto kDataFormatReconfig = ckl::DataFormatReconfig::Disabled;
#endif

#define MOREH_MASK(predicate, mask_tile_offset)                \
    ckl::runtime_if(                                           \
        predicate,                                             \
        ckl::CopyTile<                                         \
            ckl::input(                                        \
                dfb::mask_h_w,                                 \
                ckl::WaitPolicy::None,                         \
                ckl::PopPolicy::None,                          \
                ckl::InputTileMapping::Scalar,                 \
                kDataFormatReconfig,                           \
                ckl::TileAddressing::Offset),                  \
            ckl::Dst::D1>{dfb_mask_h_w_obj, mask_tile_offset}, \
        ckl::Mask<>{}),

#ifdef DO_MASK_H
#define MOREH_MASK_H(wt) MOREH_MASK(need_to_do_mask_h(wt, origin_Ht, origin_Wt), 0)
#else
#define MOREH_MASK_H(wt)
#endif

#ifdef DO_MASK_W
#define MOREH_MASK_W(wt) MOREH_MASK(((wt + 1) % origin_Wt == 0), 1)
#else
#define MOREH_MASK_W(wt)
#endif

#ifdef GAMMA_HAS_VALUE
#define MOREH_DYCOPY_OP                                                                                                \
    ckl::BinaryFpu<                                                                                                    \
        ckl::BinaryFpuOp::Mul,                                                                                         \
        ckl::input(dfb::dy, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, kDataFormatReconfig),                   \
        ckl::input(dfb::gamma, gamma_bcast, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, kDataFormatReconfig)> { \
        dfb_dy_obj, dfb_gamma_obj                                                                                      \
    }
#else
#define MOREH_DYCOPY_OP                                                                                          \
    ckl::CopyTile<ckl::input(dfb::dy, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, kDataFormatReconfig)> { \
        dfb_dy_obj                                                                                               \
    }
#endif

ALWI bool need_to_do_mask_h(uint32_t w_idx, uint32_t origin_num_h_tiles, uint32_t origin_num_w_tiles) {
    return ((w_idx / origin_num_w_tiles) + 1) % origin_num_h_tiles == 0;
}

void kernel_main() {
    constexpr auto num_rows_per_core = get_arg(args::num_rows_per_core);
    constexpr auto origin_H = get_arg(args::origin_H);
    constexpr auto origin_W = get_arg(args::origin_W);
    constexpr auto Wt = get_arg(args::Wt);
    constexpr bool is_lastdim_layernorm = get_arg(args::is_lastdim_layernorm) == 1;
    constexpr bool is_groupnorm = get_arg(args::is_groupnorm) == 1;

    compute_kernel_hw_startup(dfb::x, dfb::mean, dfb::dx);

    DataflowBuffer dfb_mean_obj(dfb::mean);            // mean
    DataflowBuffer dfb_rstd_obj(dfb::rstd);            // rstd
    DataflowBuffer dfb_scaler_obj(dfb::scaler);        // scaler
    DataflowBuffer dfb_n_recip_n_obj(dfb::n_recip_n);  // n_recip_n
#if defined(DO_MASK_H) || defined(DO_MASK_W)
    DataflowBuffer dfb_mask_h_w_obj(dfb::mask_h_w);  // mask_h_w
#endif
    DataflowBuffer dfb_dysum_obj(dfb::dysum);    // Sum[dy]
    DataflowBuffer dfb_ydysum_obj(dfb::ydysum);  // Sum[y * dy]
    DataflowBuffer dfb_x_obj(dfb::x);
    DataflowBuffer dfb_dy_obj(dfb::dy);
#ifdef GAMMA_HAS_VALUE
    DataflowBuffer dfb_gamma_obj(dfb::gamma);
#endif
    DataflowBuffer dfb_dx_obj(dfb::dx);
    DataflowBuffer dfb_y_obj(dfb::y);
    DataflowBuffer dfb_dycopy_obj(dfb::dycopy);
    DataflowBuffer dfb_tmp1_obj(dfb::tmp1);
    DataflowBuffer dfb_tmp2_obj(dfb::tmp2);
    DataflowBuffer dfb_tmp3_obj(dfb::tmp3);

    constexpr uint32_t onetile = 1;

    dfb_scaler_obj.wait_front(onetile);  // comes from the reader
    dfb_n_recip_n_obj.wait_front(2);     // comes from the reader

    constexpr uint32_t TILE_H = 32;
    constexpr uint32_t TILE_W = 32;

    constexpr uint32_t origin_Ht = (origin_H + TILE_H - 1) / TILE_H;

    constexpr uint32_t origin_Wt = (origin_W + TILE_W - 1) / TILE_W;
#ifdef GAMMA_HAS_VALUE
    constexpr auto gamma_bcast = is_groupnorm           ? ckl::BroadcastDim::Scalar
                                 : is_lastdim_layernorm ? ckl::BroadcastDim::Row
                                                        : ckl::BroadcastDim::None;
#endif

#if defined(DO_MASK_H) || defined(DO_MASK_W)
    dfb_mask_h_w_obj.wait_front(2);  // comes from the reader
#endif

    constexpr uint32_t NCHt = num_rows_per_core;

    for (uint32_t ncht = 0; ncht < NCHt; ncht++) {
        dfb_mean_obj.wait_front(onetile);  // comes from the reader
        dfb_rstd_obj.wait_front(onetile);  // comes from the reader

        // Compute y
        // y = (x - mean) * rstd
        constexpr auto dfb_dyadd = dfb::tmp1;
        DataflowBuffer& dfb_dyadd_obj = dfb_tmp1_obj;
        constexpr auto dfb_ydyadd = dfb::tmp2;
        DataflowBuffer& dfb_ydyadd_obj = dfb_tmp2_obj;
        for (uint32_t wt = 0; wt < Wt; wt++) {
            // Compute xmm
            // x - mean
            constexpr auto dfb_xmm = dfb::tmp3;
            DataflowBuffer& dfb_xmm_obj = dfb_tmp3_obj;
            ckl::sub<
                ckl::input(dfb::x, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, kDataFormatReconfig),
                ckl::input(
                    dfb::mean,
                    is_lastdim_layernorm ? ckl::BroadcastDim::Col : ckl::BroadcastDim::Scalar,
                    ckl::WaitPolicy::None,
                    ckl::PopPolicy::None,
                    kDataFormatReconfig),
                ckl::output(dfb_xmm, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile, kDataFormatReconfig)>(
                ckl::IterationShape::one_tile(), dfb_x_obj, dfb_mean_obj, dfb_xmm_obj);

            // Compute y
            // (x - mean) * rstd and mask(optional)
            ckl::eltwise_chain(
                ckl::IterationShape::one_tile(),
                ckl::BinaryFpu<
                    ckl::BinaryFpuOp::Mul,
                    ckl::input(dfb_xmm, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, kDataFormatReconfig),
                    ckl::input(
                        dfb::rstd,
                        is_lastdim_layernorm ? ckl::BroadcastDim::Col : ckl::BroadcastDim::Scalar,
                        ckl::WaitPolicy::None,
                        ckl::PopPolicy::None,
                        kDataFormatReconfig)>{dfb_xmm_obj, dfb_rstd_obj},
                MOREH_MASK_H(wt) MOREH_MASK_W(wt) ckl::PackTile<ckl::output(
                    dfb::y, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile, kDataFormatReconfig)>{dfb_y_obj});

            // Copy dy to dycopy
            // Compute dycopy
            // dycopy = dy * gamma and mask(optional)
            ckl::eltwise_chain(
                ckl::IterationShape::one_tile(),
                MOREH_DYCOPY_OP,
                MOREH_MASK_H(wt) MOREH_MASK_W(wt) ckl::PackTile<ckl::output(
                    dfb::dycopy, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile, kDataFormatReconfig)>{
                    dfb_dycopy_obj});

            // Compute dyadd
            if (wt == 0) {
                ckl::copy<
                    ckl::input(dfb::dycopy, ckl::WaitPolicy::PerTile, ckl::PopPolicy::None, kDataFormatReconfig),
                    ckl::output(dfb_dyadd, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile, kDataFormatReconfig)>(
                    ckl::IterationShape::one_tile(), dfb_dycopy_obj, dfb_dyadd_obj);
            } else {
                ckl::add<
                    ckl::input(dfb_dyadd, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, kDataFormatReconfig),
                    ckl::input(dfb::dycopy, ckl::WaitPolicy::PerTile, ckl::PopPolicy::None, kDataFormatReconfig),
                    ckl::output(dfb_dyadd, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile, kDataFormatReconfig)>(
                    ckl::IterationShape::one_tile(), dfb_dyadd_obj, dfb_dycopy_obj, dfb_dyadd_obj);
            }
            // We don't pop dycopy here.

            // Compute ydy and ydyadd
            constexpr auto dfb_ydy = dfb::tmp3;
            DataflowBuffer& dfb_ydy_obj = dfb_tmp3_obj;
            // Compute ydy
            ckl::mul<
                ckl::input(dfb::y, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, kDataFormatReconfig),
                ckl::input(dfb::dycopy, ckl::WaitPolicy::None, ckl::PopPolicy::PerTile, kDataFormatReconfig),
                ckl::output(dfb_ydy, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile, kDataFormatReconfig)>(
                ckl::IterationShape::one_tile(), dfb_y_obj, dfb_dycopy_obj, dfb_ydy_obj);

            // Compute ydyadd
            if (wt == 0) {
                ckl::copy<
                    ckl::input(dfb_ydy, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, kDataFormatReconfig),
                    ckl::output(
                        dfb_ydyadd, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile, kDataFormatReconfig)>(
                    ckl::IterationShape::one_tile(), dfb_ydy_obj, dfb_ydyadd_obj);
            } else {
                ckl::add<
                    ckl::input(dfb_ydyadd, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, kDataFormatReconfig),
                    ckl::input(dfb_ydy, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, kDataFormatReconfig),
                    ckl::output(
                        dfb_ydyadd, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile, kDataFormatReconfig)>(
                    ckl::IterationShape::one_tile(), dfb_ydyadd_obj, dfb_ydy_obj, dfb_ydyadd_obj);
            }
        }  // Wt loop

        // Compute dysum
        // Sum[dy]
        ckl::reduce<REDUCE_OP, REDUCE_DIM, dfb_dyadd, dfb::scaler, dfb::dysum>(ckl::ReduceInputBlockShape::single());

        // Compute ydysum
        // Sum[y * dy]
        ckl::reduce<REDUCE_OP, REDUCE_DIM, dfb_ydyadd, dfb::scaler, dfb::ydysum>(ckl::ReduceInputBlockShape::single());

        // Compute recip_nrstd
        // rstd / n -> tmp3
        constexpr auto dfb_recip_nrstd = dfb::tmp3;
        DataflowBuffer& dfb_recip_nrstd_obj = dfb_tmp3_obj;
        ckl::eltwise_chain(
            ckl::IterationShape::one_tile(),
            ckl::BinaryFpu<
                ckl::BinaryFpuOp::Mul,
                ckl::input(
                    dfb::n_recip_n,
                    ckl::WaitPolicy::None,
                    ckl::PopPolicy::None,
                    ckl::InputTileMapping::Scalar,
                    kDataFormatReconfig,
                    ckl::TileAddressing::Offset),
                ckl::input(
                    dfb::rstd,
                    is_lastdim_layernorm ? ckl::BroadcastDim::Col : ckl::BroadcastDim::Scalar,
                    ckl::WaitPolicy::None,
                    ckl::PopPolicy::None,
                    kDataFormatReconfig)>{dfb_n_recip_n_obj, dfb_rstd_obj, 1u, 0u},
            ckl::PackTile<ckl::output(
                dfb_recip_nrstd, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile, kDataFormatReconfig)>{
                dfb_recip_nrstd_obj});

        // Compute dx
        // ((n * dy - Sum[dy]) - (y * Sum[y * dy])) * (rstd / n)
        dfb_dysum_obj.wait_front(onetile);
        dfb_ydysum_obj.wait_front(onetile);
        dfb_recip_nrstd_obj.wait_front(onetile);
        for (uint32_t wt = 0; wt < Wt; wt++) {
            // Copy dy to dycopy
            // Compute dycopy
            // dycopy = dy * gamma and mask(optional)
            ckl::eltwise_chain(
                ckl::IterationShape::one_tile(),
                MOREH_DYCOPY_OP,
                MOREH_MASK_H(wt) MOREH_MASK_W(wt) ckl::PackTile<ckl::output(
                    dfb::dycopy, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile, kDataFormatReconfig)>{
                    dfb_dycopy_obj});

            // Compute ndy
            // n * dy
            constexpr auto dfb_ndy = dfb::tmp1;
            DataflowBuffer& dfb_ndy_obj = dfb_tmp1_obj;
            ckl::mul<
                ckl::input(dfb::n_recip_n, ckl::WaitPolicy::None, ckl::PopPolicy::None, kDataFormatReconfig),
                ckl::input(dfb::dycopy, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, kDataFormatReconfig),
                ckl::output(dfb_ndy, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile, kDataFormatReconfig)>(
                ckl::IterationShape::one_tile(), dfb_n_recip_n_obj, dfb_dycopy_obj, dfb_ndy_obj);

            // Compute ndymdysum
            // n * dy - Sum[dy]
            constexpr auto dfb_ndymdysum = dfb::tmp2;
            DataflowBuffer& dfb_ndymdysum_obj = dfb_tmp2_obj;
            ckl::sub<
                ckl::input(dfb_ndy, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, kDataFormatReconfig),
                ckl::input(
                    dfb::dysum,
                    is_lastdim_layernorm ? ckl::BroadcastDim::Col : ckl::BroadcastDim::Scalar,
                    ckl::WaitPolicy::None,
                    ckl::PopPolicy::None,
                    kDataFormatReconfig),
                ckl::output(dfb_ndymdysum, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile, kDataFormatReconfig)>(
                ckl::IterationShape::one_tile(), dfb_ndy_obj, dfb_dysum_obj, dfb_ndymdysum_obj);

            // Compute xmm
            // x - mean and mask(optional)
            constexpr auto dfb_xmm = dfb::tmp1;
            DataflowBuffer& dfb_xmm_obj = dfb_tmp1_obj;
            ckl::eltwise_chain(
                ckl::IterationShape::one_tile(),
                ckl::BinaryFpu<
                    ckl::BinaryFpuOp::Sub,
                    ckl::input(dfb::x, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, kDataFormatReconfig),
                    ckl::input(
                        dfb::mean,
                        is_lastdim_layernorm ? ckl::BroadcastDim::Col : ckl::BroadcastDim::Scalar,
                        ckl::WaitPolicy::None,
                        ckl::PopPolicy::None,
                        kDataFormatReconfig)>{dfb_x_obj, dfb_mean_obj},
                MOREH_MASK_H(wt) MOREH_MASK_W(wt) ckl::PackTile<ckl::output(
                    dfb_xmm, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile, kDataFormatReconfig)>{dfb_xmm_obj});

            // Compute y
            ckl::mul<
                ckl::input(dfb_xmm, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, kDataFormatReconfig),
                ckl::input(
                    dfb::rstd,
                    is_lastdim_layernorm ? ckl::BroadcastDim::Col : ckl::BroadcastDim::Scalar,
                    ckl::WaitPolicy::None,
                    ckl::PopPolicy::None,
                    kDataFormatReconfig),
                ckl::output(dfb::y, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile, kDataFormatReconfig)>(
                ckl::IterationShape::one_tile(), dfb_xmm_obj, dfb_rstd_obj, dfb_y_obj);

            // Compute yydysum
            // y * Sum[y * dy]
            constexpr auto dfb_yydysum = dfb::tmp1;
            DataflowBuffer& dfb_yydysum_obj = dfb_tmp1_obj;
            ckl::mul<
                ckl::input(dfb::y, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, kDataFormatReconfig),
                ckl::input(
                    dfb::ydysum,
                    is_lastdim_layernorm ? ckl::BroadcastDim::Col : ckl::BroadcastDim::Scalar,
                    ckl::WaitPolicy::None,
                    ckl::PopPolicy::None,
                    kDataFormatReconfig),
                ckl::output(dfb_yydysum, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile, kDataFormatReconfig)>(
                ckl::IterationShape::one_tile(), dfb_y_obj, dfb_ydysum_obj, dfb_yydysum_obj);

            // Compute tmp4
            // (n * dy - Sum[dy]) - (y * Sum[y * dy])
            constexpr auto dfb_tmp4 = dfb::y;
            DataflowBuffer& dfb_tmp4_obj = dfb_y_obj;
            ckl::sub<
                ckl::input(dfb_ndymdysum, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, kDataFormatReconfig),
                ckl::input(dfb_yydysum, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, kDataFormatReconfig),
                ckl::output(dfb_tmp4, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile, kDataFormatReconfig)>(
                ckl::IterationShape::one_tile(), dfb_ndymdysum_obj, dfb_yydysum_obj, dfb_tmp4_obj);

            // Compute dx
            ckl::mul<
                ckl::input(dfb_tmp4, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, kDataFormatReconfig),
                ckl::input(dfb_recip_nrstd, ckl::WaitPolicy::None, ckl::PopPolicy::None, kDataFormatReconfig),
                ckl::output(dfb::dx, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile, kDataFormatReconfig)>(
                ckl::IterationShape::one_tile(), dfb_tmp4_obj, dfb_recip_nrstd_obj, dfb_dx_obj);
        }  // Wt loop
        dfb_recip_nrstd_obj.pop_front(onetile);
        dfb_dysum_obj.pop_front(onetile);
        dfb_ydysum_obj.pop_front(onetile);

        dfb_mean_obj.pop_front(onetile);
        dfb_rstd_obj.pop_front(onetile);
    }  // NCHt loop
    dfb_scaler_obj.pop_front(onetile);
    dfb_n_recip_n_obj.pop_front(2);

#if defined(DO_MASK_H) || defined(DO_MASK_W)
    dfb_mask_h_w_obj.pop_front(2);
#endif

#undef MOREH_DYCOPY_OP
#undef MOREH_MASK_W
#undef MOREH_MASK_H
#undef MOREH_MASK
}
