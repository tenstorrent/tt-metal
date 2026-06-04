// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"  // unary
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_binary_sfpu.hpp"  // SubBinary, MulBinary, AddBinary
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_misc.hpp"         // Typecast
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_optional.hpp"     // OptionalChainElement
#include "api/dataflow/circular_buffer.h"

// running_statistics (SFPU / fp32 path): updated_stat = (1 - momentum) * old_stat + momentum * batch_stat
// for mean and/or var (compile-time gated).
//
// One chain per stat, no intermediate CBs (the original staged through cb_tmp1/2/3). The CBs are
// unpacked to DEST as fp32 (UnpackToDestFp32) and the arithmetic runs on DEST via SFPU binaries, so
// the running result is threaded through DEST slots:
//   D0 = one ; D1 = momentum ; D0 = D0 - D1            (1 - momentum)
//   D1 = old_stat ; D0 = D0 * D1                        (1 - momentum) * old
//   D1 = momentum ; D2 = batch_stat ; D1 = D1 * D2      momentum * batch
//   D0 = D0 + D1                                        sum
// (3 DEST slots — within the fp32 limit of 4/8). cb_out0 receives a copy of the last computed stat
// (var if present, else mean). When the output dtype differs from the fp32 compute format the result
// is typecast into a writer-facing CB by maybe_typecast_stat (unchanged).
namespace ckl = compute_kernel_lib;
using D = ckl::Dst;

template <
    uint32_t cb_batch,
    uint32_t cb_old,
    uint32_t cb_updated,
    bool AlsoOut0,
    uint32_t cb_one,
    uint32_t cb_momentum,
    uint32_t cb_out0>
ALWI void update_running_stat() {
    using ckl::AddBinary;
    using ckl::MulBinary;
    using ckl::SubBinary;
    constexpr auto CM = ckl::InputLifecycle::CallerManaged;
    constexpr auto STREAM = ckl::InputLifecycle::Streaming;
    constexpr auto OUT_CM = ckl::OutputLifecycle::CallerManaged;
    constexpr auto SCALAR = ckl::OperandKind::Scalar;
    constexpr auto CP_IN = ckl::CopyTileReconfig::Input;
    constexpr auto PK_OUT = ckl::PackTileReconfig::Output;

    // one/momentum held (CallerManaged); old_stat and batch_stat streamed (wait 1 / pop 1, as the
    // original popped them). cb_updated is caller-reserved/pushed; cb_out0 is handled by kernel_main.
    cb_reserve_back(cb_updated, 1);
    ckl::eltwise_chain(
        1,
        ckl::CopyTile<cb_one, D::D0, CM, SCALAR, CP_IN>{},
        ckl::CopyTile<cb_momentum, D::D1, CM, SCALAR, CP_IN>{},
        SubBinary<D::D0, D::D1, D::D0>{},  // D0 = 1 - momentum
        ckl::CopyTile<cb_old, D::D1, STREAM, SCALAR, CP_IN>{},
        MulBinary<D::D0, D::D1, D::D0>{},  // D0 = (1 - momentum) * old_stat
        ckl::CopyTile<cb_momentum, D::D1, CM, SCALAR, CP_IN>{},
        ckl::CopyTile<cb_batch, D::D2, STREAM, SCALAR, CP_IN>{},
        MulBinary<D::D1, D::D2, D::D1>{},  // D1 = momentum * batch_stat
        AddBinary<D::D0, D::D1, D::D0>{},  // D0 = (1 - momentum) * old + momentum * batch
        ckl::PackTile<cb_updated, D::D0, OUT_CM, PK_OUT>{},
        ckl::OptionalChainElement<AlsoOut0, ckl::PackTile<cb_out0, D::D0, OUT_CM, PK_OUT>>{});
    cb_push_back(cb_updated, 1);
}

template <bool NeedsTypecast, uint32_t TcInFmt, uint32_t TcOutFmt, uint32_t SrcCb, uint32_t DstCb>
ALWI void maybe_typecast_stat() {
    if constexpr (NeedsTypecast) {
        // src (the updated stat just written) -> typecast -> writer-facing CB.
        ckl::unary<
            ckl::Typecast<TcInFmt, TcOutFmt, D::D0>,
            SrcCb,
            DstCb,
            ckl::CopyTileReconfig::Input,
            ckl::OperandKind::Scalar,
            ckl::InputLifecycle::Streaming,
            ckl::OutputLifecycle::Streaming,
            ckl::PackTileReconfig::Output>(1);
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
    // The former cb_tmp1/2/3 CT-args are gone — the chain threads the result through DEST.
    constexpr auto cb_writer_updated_mean = get_compile_time_arg_val(11);
    constexpr auto cb_writer_updated_var = get_compile_time_arg_val(12);
    constexpr bool stat_needs_typecast = get_compile_time_arg_val(13) == 1;
    constexpr uint32_t tc_in_fmt = get_compile_time_arg_val(14);
    constexpr uint32_t tc_out_fmt = get_compile_time_arg_val(15);
    constexpr bool needs_mean_typecast = old_running_mean_has_value && stat_needs_typecast;
    constexpr bool needs_var_typecast = old_running_var_has_value && stat_needs_typecast;

    unary_op_init_common(cb_batch_mean, cb_out0);
    constexpr uint32_t onetile = 1;

    cb_wait_front(cb_momentum, 1);  // held for the whole kernel
    cb_wait_front(cb_one, 1);

    for (uint32_t tile_id = 0; tile_id < num_tiles; ++tile_id) {
        cb_reserve_back(cb_out0, onetile);

        if constexpr (old_running_mean_has_value) {
            update_running_stat<
                cb_batch_mean,
                cb_old_running_mean,
                cb_updated_running_mean,
                /*AlsoOut0=*/!old_running_var_has_value,
                cb_one,
                cb_momentum,
                cb_out0>();
            maybe_typecast_stat<
                needs_mean_typecast,
                tc_in_fmt,
                tc_out_fmt,
                cb_updated_running_mean,
                cb_writer_updated_mean>();
        }

        if constexpr (old_running_var_has_value) {
            update_running_stat<
                cb_batch_var,
                cb_old_running_var,
                cb_updated_running_var,
                /*AlsoOut0=*/true,
                cb_one,
                cb_momentum,
                cb_out0>();
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
