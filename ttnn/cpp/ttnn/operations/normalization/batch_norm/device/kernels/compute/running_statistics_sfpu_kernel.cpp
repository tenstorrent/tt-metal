// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_binary_sfpu.hpp"  // SubBinary, MulBinary, AddBinary
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_misc.hpp"         // Typecast
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_optional.hpp"     // OptionalChainElement
#include "api/dataflow/circular_buffer.h"

// running_statistics: updated_stat = (1 - momentum) * old_stat + momentum * batch_stat
// for mean and/or var (compile-time gated). Faithful port of the original: each of
// the original's per-tile tile_regs windows becomes its own eltwise_chain(1), and the
// intermediate CBs (cb_tmp1/2/3) are PRESERVED so the rounding/numerics match the
// original exactly (a DEST-collapse would change precision). Intermediate CBs are
// chain-owned (Streaming produce/consume); the held scalars (momentum, one) and the
// per-tile batch_mean + shared cb_out0 stay manually managed (CallerManaged in the
// chains), exactly mirroring the original's cb_wait/reserve/pop/push placement.
namespace ckl = compute_kernel_lib;
using D = ckl::Dst;
namespace IL = ckl;  // for InputLifecycle / OutputLifecycle qualified below

// One binary window: D0 = op(CbA, CbB) -> CbOut. CbA/CbB copied to D0/D1, SFPU binary
// writes D0, packed to CbOut. Lifecycles per operand/output match the original.
template <
    uint32_t CbA,
    ckl::InputLifecycle PolA,
    uint32_t CbB,
    ckl::InputLifecycle PolB,
    class BinElem,
    uint32_t CbOut,
    ckl::OutputLifecycle OutPol>
ALWI void bin_window() {
    ckl::eltwise_chain(
        1,
        ckl::CopyTile<CbA, D::D0, PolA, ckl::OperandKind::Scalar, ckl::CopyTileReconfig::Input>{},
        ckl::CopyTile<CbB, D::D1, PolB, ckl::OperandKind::Scalar, ckl::CopyTileReconfig::Input>{},
        BinElem{},
        ckl::PackTile<CbOut, D::D0, OutPol, ckl::PackTileReconfig::Output>{});
}

template <bool NeedsTypecast, uint32_t TcInFmt, uint32_t TcOutFmt, uint32_t SrcCb, uint32_t DstCb>
ALWI void maybe_typecast_stat() {
    if constexpr (NeedsTypecast) {
        // src (the updated stat just written) -> typecast -> writer-facing CB.
        ckl::eltwise_chain(
            1,
            ckl::CopyTile<
                SrcCb,
                D::D0,
                ckl::InputLifecycle::Streaming,
                ckl::OperandKind::Scalar,
                ckl::CopyTileReconfig::Input>{},
            ckl::Typecast<TcInFmt, TcOutFmt, D::D0>{},
            ckl::PackTile<DstCb, D::D0, ckl::OutputLifecycle::Streaming, ckl::PackTileReconfig::Output>{});
    }
}

void kernel_main() {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);
    constexpr uint32_t old_running_mean_has_value = get_compile_time_arg_val(0) == 1;
    constexpr uint32_t old_running_var_has_value = get_compile_time_arg_val(1) == 1;

    constexpr auto cb_batch_mean = get_compile_time_arg_val(2);
    constexpr auto cb_batch_var = get_compile_time_arg_val(3);
    constexpr auto cb_out0 = get_compile_time_arg_val(4);
    constexpr auto cb_old_running_mean = get_compile_time_arg_val(5);
    constexpr auto cb_old_running_var = get_compile_time_arg_val(6);
    constexpr auto cb_updated_running_mean = get_compile_time_arg_val(7);
    constexpr auto cb_updated_running_var = get_compile_time_arg_val(8);
    constexpr auto cb_momentum = get_compile_time_arg_val(9);
    constexpr auto cb_one = get_compile_time_arg_val(10);
    constexpr auto cb_tmp1 = get_compile_time_arg_val(11);
    constexpr auto cb_tmp2 = get_compile_time_arg_val(12);
    constexpr auto cb_tmp3 = get_compile_time_arg_val(13);
    constexpr auto cb_writer_updated_mean = get_compile_time_arg_val(14);
    constexpr auto cb_writer_updated_var = get_compile_time_arg_val(15);
    constexpr bool stat_needs_typecast = get_compile_time_arg_val(16) == 1;
    constexpr uint32_t tc_in_fmt = get_compile_time_arg_val(17);
    constexpr uint32_t tc_out_fmt = get_compile_time_arg_val(18);
    constexpr bool needs_mean_typecast = old_running_mean_has_value && stat_needs_typecast;
    constexpr bool needs_var_typecast = old_running_var_has_value && stat_needs_typecast;

    constexpr auto CM = ckl::InputLifecycle::CallerManaged;
    constexpr auto STREAM = ckl::InputLifecycle::Streaming;
    constexpr auto OUT_STREAM = ckl::OutputLifecycle::Streaming;
    constexpr auto OUT_CM = ckl::OutputLifecycle::CallerManaged;

    unary_op_init_common(cb_batch_mean, cb_out0);
    constexpr uint32_t onetile = 1;

    cb_wait_front(cb_momentum, 1);  // held for the whole kernel
    cb_wait_front(cb_one, 1);

    for (uint32_t tile_id = 0; tile_id < num_tiles; ++tile_id) {
        cb_wait_front(cb_batch_mean, onetile);  // held across the mean branch (and drained even if mean absent)
        cb_reserve_back(cb_out0, onetile);

        if constexpr (old_running_mean_has_value) {
            using ckl::AddBinary;
            using ckl::MulBinary;
            using ckl::SubBinary;
            // cb_tmp1 = one - momentum
            bin_window<cb_one, CM, cb_momentum, CM, SubBinary<D::D0, D::D1, D::D0>, cb_tmp1, OUT_STREAM>();
            // cb_tmp2 = batch_mean * momentum
            bin_window<cb_batch_mean, CM, cb_momentum, CM, MulBinary<D::D0, D::D1, D::D0>, cb_tmp2, OUT_STREAM>();
            // cb_tmp3 = old_running_mean * cb_tmp1  == (1 - momentum) * old
            bin_window<
                cb_old_running_mean,
                STREAM,
                cb_tmp1,
                STREAM,
                MulBinary<D::D0, D::D1, D::D0>,
                cb_tmp3,
                OUT_STREAM>();
            // cb_updated_running_mean = cb_tmp3 + cb_tmp2 ; also -> cb_out0 when var absent
            cb_reserve_back(cb_updated_running_mean, onetile);
            ckl::eltwise_chain(
                1,
                ckl::CopyTile<cb_tmp3, D::D0, STREAM, ckl::OperandKind::Scalar, ckl::CopyTileReconfig::Input>{},
                ckl::CopyTile<cb_tmp2, D::D1, STREAM, ckl::OperandKind::Scalar, ckl::CopyTileReconfig::Input>{},
                AddBinary<D::D0, D::D1, D::D0>{},
                ckl::PackTile<cb_updated_running_mean, D::D0, OUT_CM, ckl::PackTileReconfig::Output>{},
                ckl::OptionalChainElement<
                    !old_running_var_has_value,
                    ckl::PackTile<cb_out0, D::D0, OUT_CM, ckl::PackTileReconfig::Output>>{});
            cb_push_back(cb_updated_running_mean, onetile);

            maybe_typecast_stat<
                needs_mean_typecast,
                tc_in_fmt,
                tc_out_fmt,
                cb_updated_running_mean,
                cb_writer_updated_mean>();
        }

        cb_pop_front(cb_batch_mean, onetile);

        if constexpr (old_running_var_has_value) {
            using ckl::AddBinary;
            using ckl::MulBinary;
            using ckl::SubBinary;
            // cb_tmp1 = one - momentum
            bin_window<cb_one, CM, cb_momentum, CM, SubBinary<D::D0, D::D1, D::D0>, cb_tmp1, OUT_STREAM>();
            // cb_tmp2 = batch_var * momentum  (batch_var streamed/drained here, as in the original)
            bin_window<cb_batch_var, STREAM, cb_momentum, CM, MulBinary<D::D0, D::D1, D::D0>, cb_tmp2, OUT_STREAM>();
            // cb_tmp3 = old_running_var * cb_tmp1
            bin_window<
                cb_old_running_var,
                STREAM,
                cb_tmp1,
                STREAM,
                MulBinary<D::D0, D::D1, D::D0>,
                cb_tmp3,
                OUT_STREAM>();
            // cb_updated_running_var = cb_tmp3 + cb_tmp2 ; always also -> cb_out0
            cb_reserve_back(cb_updated_running_var, onetile);
            ckl::eltwise_chain(
                1,
                ckl::CopyTile<cb_tmp3, D::D0, STREAM, ckl::OperandKind::Scalar, ckl::CopyTileReconfig::Input>{},
                ckl::CopyTile<cb_tmp2, D::D1, STREAM, ckl::OperandKind::Scalar, ckl::CopyTileReconfig::Input>{},
                AddBinary<D::D0, D::D1, D::D0>{},
                ckl::PackTile<cb_updated_running_var, D::D0, OUT_CM, ckl::PackTileReconfig::Output>{},
                ckl::PackTile<cb_out0, D::D0, OUT_CM, ckl::PackTileReconfig::Output>{});
            cb_push_back(cb_updated_running_var, onetile);

            maybe_typecast_stat<
                needs_var_typecast,
                tc_in_fmt,
                tc_out_fmt,
                cb_updated_running_var,
                cb_writer_updated_var>();
        }

        cb_push_back(cb_out0, onetile);
    }

    cb_pop_front(cb_momentum, 1);
    cb_pop_front(cb_one, 1);
}
