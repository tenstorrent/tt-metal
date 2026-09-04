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

#define MOREH_MASK(predicate, mask_tile_offset) \
    ckl::runtime_if(                            \
        predicate,                              \
        ckl::CopyTile<                          \
            ckl::input(                         \
                dfb::mask_h_w,                  \
                ckl::WaitPolicy::None,          \
                ckl::PopPolicy::None,           \
                ckl::InputTileMapping::Scalar,  \
                kDataFormatReconfig,            \
                ckl::TileAddressing::Offset),   \
            ckl::Dst::D1>{mask_tile_offset},    \
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
    }
#else
#define MOREH_DYCOPY_OP \
    ckl::CopyTile<ckl::input(dfb::dy, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, kDataFormatReconfig)> {}
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

        // Compute dfb::y
        // y = (x - mean) * rstd
        constexpr auto dfb_dyadd_id = dfb::tmp1;
        constexpr auto dfb_ydyadd_id = dfb::tmp2;
        for (uint32_t wt = 0; wt < Wt; wt++) {
            // Compute dfb_xmm_id
            // x - mean
            constexpr auto dfb_xmm_id = dfb::tmp3;
            ckl::sub<
                ckl::input(dfb::x, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, kDataFormatReconfig),
                ckl::input(
                    dfb::mean,
                    is_lastdim_layernorm ? ckl::BroadcastDim::Col : ckl::BroadcastDim::Scalar,
                    ckl::WaitPolicy::None,
                    ckl::PopPolicy::None,
                    kDataFormatReconfig),
                ckl::output(dfb_xmm_id, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile, kDataFormatReconfig)>(
                ckl::IterationShape::tiles(onetile));

            // Compute dfb::y
            // (x - mean) * rstd and mask(optional)
            ckl::eltwise_chain(
                ckl::IterationShape::one_tile(),
                ckl::BinaryFpu<
                    ckl::BinaryFpuOp::Mul,
                    ckl::input(dfb_xmm_id, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, kDataFormatReconfig),
                    ckl::input(
                        dfb::rstd,
                        is_lastdim_layernorm ? ckl::BroadcastDim::Col : ckl::BroadcastDim::Scalar,
                        ckl::WaitPolicy::None,
                        ckl::PopPolicy::None,
                        kDataFormatReconfig)>{},
                MOREH_MASK_H(wt) MOREH_MASK_W(wt) ckl::PackTile<ckl::output(
                    dfb::y, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile, kDataFormatReconfig)>{});

            // Copy dfb::dy to dfb::dycopy
            // Compute dfb::dycopy
            // dycopy = dy * gamma and mask(optional)
            ckl::eltwise_chain(
                ckl::IterationShape::one_tile(),
                MOREH_DYCOPY_OP,
                MOREH_MASK_H(wt) MOREH_MASK_W(wt) ckl::PackTile<ckl::output(
                    dfb::dycopy, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile, kDataFormatReconfig)>{});

            // Compute dfb_dyadd_id
            if (wt == 0) {
                ckl::copy<
                    ckl::input(dfb::dycopy, ckl::WaitPolicy::PerTile, ckl::PopPolicy::None, kDataFormatReconfig),
                    ckl::output(
                        dfb_dyadd_id, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile, kDataFormatReconfig)>(
                    ckl::IterationShape::one_tile());
            } else {
                ckl::add<
                    ckl::input(dfb_dyadd_id, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, kDataFormatReconfig),
                    ckl::input(dfb::dycopy, ckl::WaitPolicy::PerTile, ckl::PopPolicy::None, kDataFormatReconfig),
                    ckl::output(
                        dfb_dyadd_id, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile, kDataFormatReconfig)>(
                    ckl::IterationShape::one_tile());
            }
            // We don't pop dfb::dycopy here.

            // Compute dfb_ydy_id and dfb_ydyadd_id
            constexpr auto dfb_ydy_id = dfb::tmp3;
            // Compute dfb_ydy_id
            ckl::mul<
                ckl::input(dfb::y, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, kDataFormatReconfig),
                ckl::input(dfb::dycopy, ckl::WaitPolicy::None, ckl::PopPolicy::PerTile, kDataFormatReconfig),
                ckl::output(dfb_ydy_id, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile, kDataFormatReconfig)>(
                ckl::IterationShape::tiles(onetile));

            // Compute dfb_ydyadd_id
            if (wt == 0) {
                ckl::copy<
                    ckl::input(dfb_ydy_id, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, kDataFormatReconfig),
                    ckl::output(
                        dfb_ydyadd_id, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile, kDataFormatReconfig)>(
                    ckl::IterationShape::one_tile());
            } else {
                ckl::add<
                    ckl::input(dfb_ydyadd_id, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, kDataFormatReconfig),
                    ckl::input(dfb_ydy_id, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, kDataFormatReconfig),
                    ckl::output(
                        dfb_ydyadd_id, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile, kDataFormatReconfig)>(
                    ckl::IterationShape::one_tile());
            }
        }  // Wt loop

        // Compute dfb::dysum
        // Sum[dy]
        ckl::reduce<REDUCE_OP, REDUCE_DIM, dfb_dyadd_id, dfb::scaler, dfb::dysum>(ckl::ReduceInputBlockShape::single());

        // Compute dfb::ydysum
        // Sum[y * dy]
        ckl::reduce<REDUCE_OP, REDUCE_DIM, dfb_ydyadd_id, dfb::scaler, dfb::ydysum>(
            ckl::ReduceInputBlockShape::single());

        // Compute dfb_recip_nrstd_id
        // rstd / n -> dfb::tmp3
        constexpr auto dfb_recip_nrstd_id = dfb::tmp3;
        DataflowBuffer dfb_recip_nrstd_obj(dfb_recip_nrstd_id);
        ckl::eltwise_chain(
            ckl::IterationShape::tiles(onetile),
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
                    kDataFormatReconfig)>{1u, 0u},
            ckl::PackTile<ckl::output(
                dfb_recip_nrstd_id, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile, kDataFormatReconfig)>{});

        // Compute dfb::dx
        // ((n * dy - Sum[dy]) - (y * Sum[y * dy])) * (rstd / n)
        dfb_dysum_obj.wait_front(onetile);
        dfb_ydysum_obj.wait_front(onetile);
        dfb_recip_nrstd_obj.wait_front(onetile);
        for (uint32_t wt = 0; wt < Wt; wt++) {
            // Copy dfb::dy to dfb::dycopy
            // Compute dfb::dycopy
            // dycopy = dy * gamma and mask(optional)
            ckl::eltwise_chain(
                ckl::IterationShape::one_tile(),
                MOREH_DYCOPY_OP,
                MOREH_MASK_H(wt) MOREH_MASK_W(wt) ckl::PackTile<ckl::output(
                    dfb::dycopy, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile, kDataFormatReconfig)>{});

            // Compute dfb_ndy_id
            // n * dy
            constexpr auto dfb_ndy_id = dfb::tmp1;
            ckl::mul<
                ckl::input(dfb::n_recip_n, ckl::WaitPolicy::None, ckl::PopPolicy::None, kDataFormatReconfig),
                ckl::input(dfb::dycopy, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, kDataFormatReconfig),
                ckl::output(dfb_ndy_id, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile, kDataFormatReconfig)>(
                ckl::IterationShape::tiles(onetile));

            // Compute dfb_ndymdysum_id
            // n * dy - Sum[dy]
            constexpr auto dfb_ndymdysum_id = dfb::tmp2;
            ckl::sub<
                ckl::input(dfb_ndy_id, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, kDataFormatReconfig),
                ckl::input(
                    dfb::dysum,
                    is_lastdim_layernorm ? ckl::BroadcastDim::Col : ckl::BroadcastDim::Scalar,
                    ckl::WaitPolicy::None,
                    ckl::PopPolicy::None,
                    kDataFormatReconfig),
                ckl::output(
                    dfb_ndymdysum_id, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile, kDataFormatReconfig)>(
                ckl::IterationShape::tiles(onetile));

            // Compute dfb_xmm_id
            // x - mean and mask(optional)
            constexpr auto dfb_xmm_id = dfb::tmp1;
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
                        kDataFormatReconfig)>{},
                MOREH_MASK_H(wt) MOREH_MASK_W(wt) ckl::PackTile<ckl::output(
                    dfb_xmm_id, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile, kDataFormatReconfig)>{});

            // Compute dfb::y
            ckl::mul<
                ckl::input(dfb_xmm_id, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, kDataFormatReconfig),
                ckl::input(
                    dfb::rstd,
                    is_lastdim_layernorm ? ckl::BroadcastDim::Col : ckl::BroadcastDim::Scalar,
                    ckl::WaitPolicy::None,
                    ckl::PopPolicy::None,
                    kDataFormatReconfig),
                ckl::output(dfb::y, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile, kDataFormatReconfig)>(
                ckl::IterationShape::tiles(onetile));

            // Compute dfb_yydysum_id
            // y * Sum[y * dy]
            constexpr auto dfb_yydysum_id = dfb::tmp1;
            ckl::mul<
                ckl::input(dfb::y, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, kDataFormatReconfig),
                ckl::input(
                    dfb::ydysum,
                    is_lastdim_layernorm ? ckl::BroadcastDim::Col : ckl::BroadcastDim::Scalar,
                    ckl::WaitPolicy::None,
                    ckl::PopPolicy::None,
                    kDataFormatReconfig),
                ckl::output(
                    dfb_yydysum_id, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile, kDataFormatReconfig)>(
                ckl::IterationShape::tiles(onetile));

            // Compute dfb_tmp4_id
            // (n * dy - Sum[dy]) - (y * Sum[y * dy])
            constexpr auto dfb_tmp4_id = dfb::y;
            ckl::sub<
                ckl::input(dfb_ndymdysum_id, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, kDataFormatReconfig),
                ckl::input(dfb_yydysum_id, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, kDataFormatReconfig),
                ckl::output(dfb_tmp4_id, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile, kDataFormatReconfig)>(
                ckl::IterationShape::tiles(onetile));

            // Compute dfb::dx
            ckl::mul<
                ckl::input(dfb_tmp4_id, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, kDataFormatReconfig),
                ckl::input(dfb_recip_nrstd_id, ckl::WaitPolicy::None, ckl::PopPolicy::None, kDataFormatReconfig),
                ckl::output(dfb::dx, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile, kDataFormatReconfig)>(
                ckl::IterationShape::tiles(onetile));
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
