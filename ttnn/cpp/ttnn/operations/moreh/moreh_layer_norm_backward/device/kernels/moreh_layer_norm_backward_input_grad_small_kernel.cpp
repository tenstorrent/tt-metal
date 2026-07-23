// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/kernel/compute/moreh_common.hpp"
#include "api/dataflow/dataflow_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"  // mul

namespace ckl = compute_kernel_lib;

#if defined(FP32_DEST_ACC_EN)
constexpr auto kDataFormatReconfig = ckl::DataFormatReconfig::Enabled;
#else
constexpr auto kDataFormatReconfig = ckl::DataFormatReconfig::Disabled;
#endif

ALWI bool need_to_do_mask_h(uint32_t w_idx, uint32_t origin_num_h_tiles, uint32_t origin_num_w_tiles) {
    return ((w_idx / origin_num_w_tiles) + 1) % origin_num_h_tiles == 0;
}

void kernel_main() {
    constexpr uint32_t num_rows_per_core = get_compile_time_arg_val(0);
    constexpr uint32_t origin_H = get_compile_time_arg_val(1);
    constexpr uint32_t origin_W = get_compile_time_arg_val(2);
    constexpr uint32_t Wt = get_compile_time_arg_val(3);
    constexpr bool gamma_has_value = get_compile_time_arg_val(4) == 1;
    constexpr bool is_lastdim_layernorm = get_compile_time_arg_val(5) == 1;
    constexpr bool is_groupnorm = get_compile_time_arg_val(6) == 1;

    binary_op_init_common(tt::CBIndex::c_1, tt::CBIndex::c_2, tt::CBIndex::c_16);

    constexpr auto cb_dy = tt::CBIndex::c_0;
    constexpr auto cb_x = tt::CBIndex::c_1;
    constexpr auto cb_mean = tt::CBIndex::c_2;
    DataflowBuffer cb_mean_obj(cb_mean);  // mean
    constexpr auto cb_rstd = tt::CBIndex::c_3;
    DataflowBuffer cb_rstd_obj(cb_rstd);  // rstd
    constexpr auto cb_scaler = tt::CBIndex::c_4;
    DataflowBuffer cb_scaler_obj(cb_scaler);  // scaler
    constexpr auto cb_n_recip_n = tt::CBIndex::c_5;
    DataflowBuffer cb_n_recip_n_obj(cb_n_recip_n);  // n_recip_n
    constexpr auto cb_gamma = tt::CBIndex::c_6;
    constexpr auto cb_mask_h_w = tt::CBIndex::c_7;
    DataflowBuffer cb_mask_h_w_obj(cb_mask_h_w);  // mask_h_w

    // dx = ((n * dy - Sum[dy]) - (y * Sum[y * dy])) * (rstd / n)
    constexpr auto cb_dx = tt::CBIndex::c_16;

    // y = (x - mean) * rstd
    constexpr auto cb_dycopy = tt::CBIndex::c_24;
    DataflowBuffer cb_dycopy_obj(cb_dycopy);  // copy output_grad(==dycopy)
    constexpr auto cb_y = tt::CBIndex::c_25;
    DataflowBuffer cb_y_obj(cb_y);  // output(==y)
    constexpr auto cb_dysum = tt::CBIndex::c_26;
    DataflowBuffer cb_dysum_obj(cb_dysum);  // Sum[dy]
    constexpr auto cb_ydysum = tt::CBIndex::c_27;
    DataflowBuffer cb_ydysum_obj(cb_ydysum);  // Sum[y * dy]
    constexpr auto cb_recip_nrstd = tt::CBIndex::c_28;
    DataflowBuffer cb_recip_nrstd_obj(cb_recip_nrstd);  // rstd / n

    constexpr auto cb_tmp1 = tt::CBIndex::c_29;
    constexpr auto cb_tmp2 = tt::CBIndex::c_30;  // tmp2
    constexpr auto cb_tmp3 = tt::CBIndex::c_31;  // tmp3

    constexpr uint32_t onetile = 1;

    cb_scaler_obj.wait_front(onetile);  // comes from the reader
    cb_n_recip_n_obj.wait_front(2);     // comes from the reader

    constexpr uint32_t TILE_H = 32;
    constexpr uint32_t TILE_W = 32;

    constexpr bool do_mask_h = (origin_H % TILE_H) != 0 && !is_lastdim_layernorm;
    constexpr uint32_t origin_Ht = (origin_H + TILE_H - 1) / TILE_H;

    constexpr bool do_mask_w = (origin_W % TILE_W) != 0;
    constexpr uint32_t origin_Wt = (origin_W + TILE_W - 1) / TILE_W;

    if constexpr (do_mask_h || do_mask_w) {
        cb_mask_h_w_obj.wait_front(2);  // comes from the reader
    }

    constexpr uint32_t NCHt = num_rows_per_core;

    for (uint32_t ncht = 0; ncht < NCHt; ncht++) {
        cb_mean_obj.wait_front(onetile);  // comes from the reader
        cb_rstd_obj.wait_front(onetile);  // comes from the reader

        // Compute cb_recip_nrstd
        // rstd / n
        ckl::eltwise_chain(
            ckl::EltwiseShape::tiles(onetile),
            ckl::BinaryFpu<
                ckl::input(
                    cb_n_recip_n,
                    ckl::InputLifecycle::CallerManaged,
                    ckl::OperandKind::Scalar,
                    kDataFormatReconfig,
                    ckl::TileOffset::Set),
                ckl::input(cb_rstd, ckl::InputLifecycle::CallerManaged, kDataFormatReconfig),
                ckl::BinaryFpuOp::Mul,
                is_lastdim_layernorm ? ckl::BroadcastDim::Col : ckl::BroadcastDim::Scalar>{1u, 0u},
            ckl::PackTile<ckl::output(cb_recip_nrstd, ckl::OutputLifecycle::Streaming, kDataFormatReconfig)>{});

        // y = (x - mean) * rstd
        for (uint32_t wt = 0; wt < Wt; wt++) {
            // Compute cb_xmm
            // x - mean and mask(optional)
            constexpr auto cb_xmm = cb_tmp2;
            ckl::eltwise_chain(
                ckl::EltwiseShape::single(),
                ckl::BinaryFpu<
                    ckl::input(cb_x, ckl::InputLifecycle::Streaming, kDataFormatReconfig),
                    ckl::input(cb_mean, ckl::InputLifecycle::CallerManaged, kDataFormatReconfig),
                    ckl::BinaryFpuOp::Sub,
                    is_lastdim_layernorm ? ckl::BroadcastDim::Col : ckl::BroadcastDim::Scalar>{},
                ckl::runtime_if(
                    do_mask_h && need_to_do_mask_h(wt, origin_Ht, origin_Wt),
                    ckl::CopyTile<
                        ckl::input(
                            cb_mask_h_w,
                            ckl::InputLifecycle::CallerManaged,
                            ckl::OperandKind::Scalar,
                            kDataFormatReconfig,
                            ckl::TileOffset::Set),
                        ckl::Dst::D1>{0},
                    ckl::Mask<>{}),
                ckl::runtime_if(
                    do_mask_w && ((wt + 1) % origin_Wt == 0),
                    ckl::CopyTile<
                        ckl::input(
                            cb_mask_h_w,
                            ckl::InputLifecycle::CallerManaged,
                            ckl::OperandKind::Scalar,
                            kDataFormatReconfig,
                            ckl::TileOffset::Set),
                        ckl::Dst::D1>{1},
                    ckl::Mask<>{}),
                ckl::PackTile<ckl::output(cb_xmm, ckl::OutputLifecycle::Streaming, kDataFormatReconfig)>{});

            // Compute cb_y
            // (x - mean) * rstd
            ckl::mul<
                ckl::input(cb_xmm, ckl::InputLifecycle::Streaming, kDataFormatReconfig),
                ckl::input(cb_rstd, ckl::InputLifecycle::CallerManaged, kDataFormatReconfig),
                ckl::output(cb_y, ckl::OutputLifecycle::Streaming, kDataFormatReconfig),
                is_lastdim_layernorm ? ckl::BroadcastDim::Col : ckl::BroadcastDim::Scalar>(ckl::EltwiseShape::single());
        }  // Wt loop

        // Copy cb_dy to cb_dycopy
        constexpr auto gamma_bcast = is_groupnorm           ? ckl::BroadcastDim::Scalar
                                     : is_lastdim_layernorm ? ckl::BroadcastDim::Row
                                                            : ckl::BroadcastDim::None;
        for (uint32_t wt = 0; wt < Wt; wt++) {
            ckl::eltwise_chain(
                ckl::EltwiseShape::single(),
                ckl::OptionalChainElement<
                    gamma_has_value,
                    ckl::BinaryFpu<
                        ckl::input(cb_dy, ckl::InputLifecycle::Streaming, kDataFormatReconfig),
                        ckl::input(cb_gamma, ckl::InputLifecycle::Streaming, kDataFormatReconfig),
                        ckl::BinaryFpuOp::Mul,
                        gamma_bcast>>{},
                ckl::OptionalChainElement<
                    !gamma_has_value,
                    ckl::CopyTile<ckl::input(cb_dy, ckl::InputLifecycle::Streaming, kDataFormatReconfig)>>{},
                ckl::runtime_if(
                    do_mask_h && need_to_do_mask_h(wt, origin_Ht, origin_Wt),
                    ckl::CopyTile<
                        ckl::input(
                            cb_mask_h_w,
                            ckl::InputLifecycle::CallerManaged,
                            ckl::OperandKind::Scalar,
                            kDataFormatReconfig,
                            ckl::TileOffset::Set),
                        ckl::Dst::D1>{0},
                    ckl::Mask<>{}),
                ckl::runtime_if(
                    do_mask_w && ((wt + 1) % origin_Wt == 0),
                    ckl::CopyTile<
                        ckl::input(
                            cb_mask_h_w,
                            ckl::InputLifecycle::CallerManaged,
                            ckl::OperandKind::Scalar,
                            kDataFormatReconfig,
                            ckl::TileOffset::Set),
                        ckl::Dst::D1>{1},
                    ckl::Mask<>{}),
                ckl::PackTile<ckl::output(cb_dycopy, ckl::OutputLifecycle::Streaming, kDataFormatReconfig)>{});
        }  // Wt loop

        // Compute cb_dyadd
        constexpr auto cb_dyadd = cb_tmp1;
        cb_dycopy_obj.wait_front(Wt);
        for (uint32_t wt = 0; wt < Wt; wt++) {
            if (wt == 0) {
                ckl::copy<
                    ckl::input(
                        cb_dycopy,
                        ckl::InputLifecycle::CallerManaged,
                        ckl::OperandKind::Scalar,
                        kDataFormatReconfig,
                        ckl::TileOffset::Set),
                    ckl::output(cb_dyadd, ckl::OutputLifecycle::Streaming, kDataFormatReconfig)>(
                    ckl::EltwiseShape::single());
            } else {
                ckl::eltwise_chain(
                    ckl::EltwiseShape::single(),
                    ckl::BinaryFpu<
                        ckl::input(cb_dyadd, ckl::InputLifecycle::Streaming, kDataFormatReconfig),
                        ckl::input(
                            cb_dycopy,
                            ckl::InputLifecycle::CallerManaged,
                            ckl::OperandKind::Scalar,
                            kDataFormatReconfig,
                            ckl::TileOffset::Set),
                        ckl::BinaryFpuOp::Add,
                        ckl::BroadcastDim::None>{0, wt},
                    ckl::PackTile<ckl::output(cb_dyadd, ckl::OutputLifecycle::Streaming, kDataFormatReconfig)>{});
            }
        }  // Wt loop
        // We don't pop cb_dycopy here.

        // Compute cb_dysum
        // Sum[dy]
        ckl::reduce<REDUCE_OP, REDUCE_DIM, cb_dyadd, cb_scaler, cb_dysum>(ckl::ReduceInputBlockShape::single());

        // Compute cb_ydy and cb_ydyadd
        constexpr auto cb_ydy = cb_tmp2;
        constexpr auto cb_ydyadd = cb_tmp3;
        cb_y_obj.wait_front(Wt);
        for (uint32_t wt = 0; wt < Wt; wt++) {
            ckl::eltwise_chain(
                ckl::EltwiseShape::tiles(onetile),
                ckl::BinaryFpu<
                    ckl::input(
                        cb_y,
                        ckl::InputLifecycle::CallerManaged,
                        ckl::OperandKind::Scalar,
                        kDataFormatReconfig,
                        ckl::TileOffset::Set),
                    ckl::input(
                        cb_dycopy,
                        ckl::InputLifecycle::CallerManaged,
                        ckl::OperandKind::Scalar,
                        kDataFormatReconfig,
                        ckl::TileOffset::Set),
                    ckl::BinaryFpuOp::Mul,
                    ckl::BroadcastDim::None>{wt, wt},
                ckl::PackTile<ckl::output(cb_ydy, ckl::OutputLifecycle::Streaming, kDataFormatReconfig)>{});

            // Compute cb_ydyadd
            if (wt == 0) {
                ckl::copy<
                    ckl::input(cb_ydy, ckl::InputLifecycle::Streaming, kDataFormatReconfig),
                    ckl::output(cb_ydyadd, ckl::OutputLifecycle::Streaming, kDataFormatReconfig)>(
                    ckl::EltwiseShape::single());
            } else {
                ckl::add<
                    ckl::input(cb_ydyadd, ckl::InputLifecycle::Streaming, kDataFormatReconfig),
                    ckl::input(cb_ydy, ckl::InputLifecycle::Streaming, kDataFormatReconfig),
                    ckl::output(cb_ydyadd, ckl::OutputLifecycle::Streaming, kDataFormatReconfig)>(
                    ckl::EltwiseShape::single());
            }
        }  // Wt loop
        // We don't pop cb_y here.

        // Compute cb_ydysum
        // Sum[y * dy]
        ckl::reduce<REDUCE_OP, REDUCE_DIM, cb_ydyadd, cb_scaler, cb_ydysum>(ckl::ReduceInputBlockShape::single());

        // Compute cb_dx
        // ((n * dy - Sum[dy]) - (y * Sum[y * dy])) * (rstd / n)
        cb_dysum_obj.wait_front(onetile);
        cb_ydysum_obj.wait_front(onetile);
        cb_recip_nrstd_obj.wait_front(onetile);
        for (uint32_t wt = 0; wt < Wt; wt++) {
            // Compute cb_ndy
            // n * dy
            constexpr auto cb_ndy = cb_tmp1;
            ckl::eltwise_chain(
                ckl::EltwiseShape::tiles(onetile),
                ckl::BinaryFpu<
                    ckl::input(cb_n_recip_n, ckl::InputLifecycle::CallerManaged, kDataFormatReconfig),
                    ckl::input(
                        cb_dycopy,
                        ckl::InputLifecycle::CallerManaged,
                        ckl::OperandKind::Scalar,
                        kDataFormatReconfig,
                        ckl::TileOffset::Set),
                    ckl::BinaryFpuOp::Mul,
                    ckl::BroadcastDim::None>{0u, wt},
                ckl::PackTile<ckl::output(cb_ndy, ckl::OutputLifecycle::Streaming, kDataFormatReconfig)>{});

            // cb_ndymdysum
            // n * dy - Sum[dy]
            constexpr auto cb_ndymdysum = cb_tmp2;
            ckl::sub<
                ckl::input(cb_ndy, ckl::InputLifecycle::Streaming, kDataFormatReconfig),
                ckl::input(cb_dysum, ckl::InputLifecycle::CallerManaged, kDataFormatReconfig),
                ckl::output(cb_ndymdysum, ckl::OutputLifecycle::Streaming, kDataFormatReconfig),
                is_lastdim_layernorm ? ckl::BroadcastDim::Col : ckl::BroadcastDim::Scalar>(
                ckl::EltwiseShape::tiles(onetile));

            // Compute cb_yydysum
            // y * Sum[y * dy]
            constexpr auto cb_yydysum = cb_tmp3;
            ckl::eltwise_chain(
                ckl::EltwiseShape::tiles(onetile),
                ckl::BinaryFpu<
                    ckl::input(
                        cb_y,
                        ckl::InputLifecycle::CallerManaged,
                        ckl::OperandKind::Scalar,
                        kDataFormatReconfig,
                        ckl::TileOffset::Set),
                    ckl::input(cb_ydysum, ckl::InputLifecycle::CallerManaged, kDataFormatReconfig),
                    ckl::BinaryFpuOp::Mul,
                    is_lastdim_layernorm ? ckl::BroadcastDim::Col : ckl::BroadcastDim::Scalar>{wt, 0u},
                ckl::PackTile<ckl::output(cb_yydysum, ckl::OutputLifecycle::Streaming, kDataFormatReconfig)>{});

            // Compute cb_tmp1
            // (n * dy - Sum[dy]) - (y * Sum[y * dy])
            ckl::sub<
                ckl::input(cb_ndymdysum, ckl::InputLifecycle::Streaming, kDataFormatReconfig),
                ckl::input(cb_yydysum, ckl::InputLifecycle::Streaming, kDataFormatReconfig),
                ckl::output(cb_tmp1, ckl::OutputLifecycle::Streaming, kDataFormatReconfig)>(
                ckl::EltwiseShape::tiles(onetile));

            // Compute cb_dx
            // ((n * dy - Sum[dy]) - (y * Sum[y * dy])) * (rstd / n)
            ckl::mul<
                ckl::input(cb_tmp1, ckl::InputLifecycle::Streaming, kDataFormatReconfig),
                ckl::input(cb_recip_nrstd, ckl::InputLifecycle::CallerManaged, kDataFormatReconfig),
                ckl::output(cb_dx, ckl::OutputLifecycle::Streaming, kDataFormatReconfig),
                ckl::BroadcastDim::None>(ckl::EltwiseShape::tiles(onetile));
        }  // Wt loop
        cb_dycopy_obj.pop_front(Wt);
        cb_y_obj.pop_front(Wt);

        cb_recip_nrstd_obj.pop_front(onetile);
        cb_dysum_obj.pop_front(onetile);
        cb_ydysum_obj.pop_front(onetile);

        cb_mean_obj.pop_front(onetile);
        cb_rstd_obj.pop_front(onetile);
    }  // NCHt loop
    cb_scaler_obj.pop_front(onetile);
    cb_n_recip_n_obj.pop_front(2);

    if constexpr (do_mask_h || do_mask_w) {
        cb_mask_h_w_obj.wait_front(2);
    }
}
