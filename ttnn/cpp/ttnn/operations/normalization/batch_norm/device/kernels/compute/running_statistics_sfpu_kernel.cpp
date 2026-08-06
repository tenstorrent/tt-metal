// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"  // unary
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_binary_sfpu_basic.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_misc.hpp"      // Typecast
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_optional.hpp"  // OptionalChainElement
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

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
    constexpr auto SCALAR = ckl::OperandKind::Scalar;

    ckl::eltwise_chain(
        ckl::EltwiseShape::single(),
        ckl::CopyTile<ckl::input(cb_one, ckl::WaitPolicy::None, ckl::PopPolicy::None, SCALAR), D::D0>{},
        ckl::CopyTile<ckl::input(cb_momentum, ckl::WaitPolicy::None, ckl::PopPolicy::None, SCALAR), D::D1>{},
        SubBinary<D::D0, D::D1, D::D0>{},  // D0 = 1 - momentum
        ckl::CopyTile<ckl::input(cb_old, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, SCALAR), D::D1>{},
        MulBinary<D::D0, D::D1, D::D0>{},  // D0 = (1 - momentum) * old_stat
        ckl::CopyTile<ckl::input(cb_momentum, ckl::WaitPolicy::None, ckl::PopPolicy::None, SCALAR), D::D1>{},
        ckl::CopyTile<ckl::input(cb_batch, ckl::WaitPolicy::None, ckl::PopPolicy::None, SCALAR), D::D2>{},
        MulBinary<D::D1, D::D2, D::D1>{},  // D1 = momentum * batch_stat
        AddBinary<D::D0, D::D1, D::D0>{},  // D0 = (1 - momentum) * old + momentum * batch
        ckl::PackTile<ckl::output(cb_updated, ckl::ReservePolicy::Upfront, ckl::PushPolicy::AtEnd)>{},
        ckl::OptionalChainElement<
            AlsoOut0,
            ckl::PackTile<ckl::output(cb_out0, ckl::ReservePolicy::None, ckl::PushPolicy::None)>>{});
}

template <bool NeedsTypecast, uint32_t TcInFmt, uint32_t TcOutFmt, uint32_t SrcCb, uint32_t DstCb>
ALWI void maybe_typecast_stat() {
    if constexpr (NeedsTypecast) {
        ckl::unary<ckl::Typecast<TcInFmt, TcOutFmt, D::D0>, ckl::input(SrcCb), ckl::output(DstCb)>(
            ckl::EltwiseShape::single());
    }
}

// A writer-facing stat DFB is only bound when the accumulation format is wider than the stat dtype;
// on the other path the writer drains the compute output directly, so the same kernel-side handle
// has to name a different DFB. The aliases are gated at the preprocessor stage because
// dfb::writer_updated_* simply does not exist on the untypecast build. The host computes each flag
// (stat present AND the stat format needs a typecast) so one define gates one alias; the two stats
// are keyed independently, and either may typecast without the other.
#ifdef NEEDS_MEAN_TYPECAST
constexpr bool needs_mean_typecast = true;
constexpr auto dfb_writer_updated_mean = dfb::writer_updated_mean;
#else
constexpr bool needs_mean_typecast = false;
constexpr auto dfb_writer_updated_mean = dfb::updated_mean;
#endif

#ifdef NEEDS_VAR_TYPECAST
constexpr bool needs_var_typecast = true;
constexpr auto dfb_writer_updated_var = dfb::writer_updated_var;
#else
constexpr bool needs_var_typecast = false;
constexpr auto dfb_writer_updated_var = dfb::updated_var;
#endif

void kernel_main() {
    uint32_t num_tiles = get_arg(args::num_tiles);
    constexpr uint32_t old_running_mean_has_value = get_arg(args::old_running_mean_has_value) == 1;
    constexpr uint32_t old_running_var_has_value = get_arg(args::old_running_var_has_value) == 1;
    static_assert(
        old_running_mean_has_value || old_running_var_has_value,
        "running_statistics requires at least one of running_mean / running_var");

    constexpr auto cb_batch_mean = dfb::batch_mean;
    constexpr auto cb_batch_var = dfb::batch_var;
    constexpr auto cb_out0 = dfb::out;
    constexpr auto cb_old_running_mean = dfb::old_running_mean;
    constexpr auto cb_old_running_var = dfb::old_running_var;
    constexpr auto cb_updated_running_mean = dfb::updated_mean;
    constexpr auto cb_updated_running_var = dfb::updated_var;
    constexpr auto cb_momentum = dfb::momentum;
    constexpr auto cb_one = dfb::one;
    constexpr auto cb_writer_updated_mean = dfb_writer_updated_mean;
    constexpr auto cb_writer_updated_var = dfb_writer_updated_var;
    constexpr uint32_t tc_in_fmt = get_arg(args::tc_in_fmt);
    constexpr uint32_t tc_out_fmt = get_arg(args::tc_out_fmt);

    DataflowBuffer cb_batch_mean_obj(cb_batch_mean);
    DataflowBuffer cb_batch_var_obj(cb_batch_var);

    compute_kernel_hw_startup(cb_batch_mean, cb_out0);
    constexpr uint32_t onetile = 1;

    DataflowBuffer(cb_momentum).wait_front(1);
    DataflowBuffer(cb_one).wait_front(1);

    for (uint32_t tile_id = 0; tile_id < num_tiles; ++tile_id) {
        // The reader produces both batch-stat streams for every tile, even when only one
        // running statistic is requested. Consume both streams unconditionally to avoid
        // filling either two-entry buffer and stalling its producer.
        cb_batch_mean_obj.wait_front(onetile);
        cb_batch_var_obj.wait_front(onetile);
        DataflowBuffer(cb_out0).reserve_back(onetile);

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

        DataflowBuffer(cb_out0).push_back(onetile);
        cb_batch_mean_obj.pop_front(onetile);
        cb_batch_var_obj.pop_front(onetile);
    }

    DataflowBuffer(cb_momentum).pop_front(1);
    DataflowBuffer(cb_one).pop_front(1);
}
