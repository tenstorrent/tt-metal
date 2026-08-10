// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/core/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"  // unary
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/binary/sfpu/basic.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/misc.hpp"     // Typecast
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/core/optional.hpp"  // Optional
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

namespace ckl = compute_kernel_lib;
using D = ckl::Dst;

template <
    uint32_t dfb_batch_id,
    uint32_t dfb_old_id,
    uint32_t dfb_updated_id,
    bool AlsoOut0,
    uint32_t dfb_one_id,
    uint32_t dfb_momentum_id,
    uint32_t dfb_out0_id>
ALWI void update_running_stat() {
    using ckl::AddBinary;
    using ckl::MulBinary;
    using ckl::SubBinary;
    constexpr auto SCALAR = ckl::OperandKind::Scalar;

    ckl::eltwise_chain(
        ckl::IterationShape::one_tile(),
        ckl::CopyTile<ckl::input(dfb_one_id, ckl::WaitPolicy::None, ckl::PopPolicy::None, SCALAR), D::D0>{},
        ckl::CopyTile<ckl::input(dfb_momentum_id, ckl::WaitPolicy::None, ckl::PopPolicy::None, SCALAR), D::D1>{},
        SubBinary<D::D0, D::D1, D::D0>{},  // D0 = 1 - momentum
        ckl::CopyTile<ckl::input(dfb_old_id, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, SCALAR), D::D1>{},
        MulBinary<D::D0, D::D1, D::D0>{},  // D0 = (1 - momentum) * old_stat
        ckl::CopyTile<ckl::input(dfb_momentum_id, ckl::WaitPolicy::None, ckl::PopPolicy::None, SCALAR), D::D1>{},
        ckl::CopyTile<ckl::input(dfb_batch_id, ckl::WaitPolicy::None, ckl::PopPolicy::None, SCALAR), D::D2>{},
        MulBinary<D::D1, D::D2, D::D1>{},  // D1 = momentum * batch_stat
        AddBinary<D::D0, D::D1, D::D0>{},  // D0 = (1 - momentum) * old + momentum * batch
        ckl::PackTile<ckl::output(dfb_updated_id, ckl::ReservePolicy::Upfront, ckl::PushPolicy::AtEnd)>{},
        ckl::Optional<
            AlsoOut0,
            ckl::PackTile<ckl::output(dfb_out0_id, ckl::ReservePolicy::None, ckl::PushPolicy::None)>>{});
}

template <bool NeedsTypecast, uint32_t TcInFmt, uint32_t TcOutFmt, uint32_t SrcDfb, uint32_t DstDfb>
ALWI void maybe_typecast_stat() {
    if constexpr (NeedsTypecast) {
        ckl::unary<ckl::Typecast<TcInFmt, TcOutFmt, D::D0>, ckl::input(SrcDfb), ckl::output(DstDfb)>(
            ckl::IterationShape::one_tile());
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
constexpr auto dfb_writer_updated_mean_binding = dfb::writer_updated_mean;
#else
constexpr bool needs_mean_typecast = false;
constexpr auto dfb_writer_updated_mean_binding = dfb::updated_mean;
#endif

#ifdef NEEDS_VAR_TYPECAST
constexpr bool needs_var_typecast = true;
constexpr auto dfb_writer_updated_var_binding = dfb::writer_updated_var;
#else
constexpr bool needs_var_typecast = false;
constexpr auto dfb_writer_updated_var_binding = dfb::updated_var;
#endif

void kernel_main() {
    uint32_t num_tiles = get_arg(args::num_tiles);
    constexpr uint32_t old_running_mean_has_value = get_arg(args::old_running_mean_has_value) == 1;
    constexpr uint32_t old_running_var_has_value = get_arg(args::old_running_var_has_value) == 1;
    static_assert(
        old_running_mean_has_value || old_running_var_has_value,
        "running_statistics requires at least one of running_mean / running_var");

    constexpr auto dfb_batch_mean_id = dfb::batch_mean;
    constexpr auto dfb_batch_var_id = dfb::batch_var;
    constexpr auto dfb_out0_id = dfb::out;
    constexpr auto dfb_old_running_mean_id = dfb::old_running_mean;
    constexpr auto dfb_old_running_var_id = dfb::old_running_var;
    constexpr auto dfb_updated_running_mean_id = dfb::updated_mean;
    constexpr auto dfb_updated_running_var_id = dfb::updated_var;
    constexpr auto dfb_momentum_id = dfb::momentum;
    constexpr auto dfb_one_id = dfb::one;
    constexpr auto dfb_writer_updated_mean_id = dfb_writer_updated_mean_binding;
    constexpr auto dfb_writer_updated_var_id = dfb_writer_updated_var_binding;
    constexpr uint32_t tc_in_fmt = get_arg(args::tc_in_fmt);
    constexpr uint32_t tc_out_fmt = get_arg(args::tc_out_fmt);

    DataflowBuffer dfb_batch_mean_obj(dfb_batch_mean_id);
    DataflowBuffer dfb_batch_var_obj(dfb_batch_var_id);

    compute_kernel_hw_startup(dfb_batch_mean_id, dfb_out0_id);
    constexpr uint32_t onetile = 1;

    DataflowBuffer(dfb_momentum_id).wait_front(1);
    DataflowBuffer(dfb_one_id).wait_front(1);

    for (uint32_t tile_id = 0; tile_id < num_tiles; ++tile_id) {
        // The reader produces both batch-stat streams for every tile, even when only one
        // running statistic is requested. Consume both streams unconditionally to avoid
        // filling either two-entry buffer and stalling its producer.
        dfb_batch_mean_obj.wait_front(onetile);
        dfb_batch_var_obj.wait_front(onetile);
        DataflowBuffer(dfb_out0_id).reserve_back(onetile);

        if constexpr (old_running_mean_has_value) {
            update_running_stat<
                dfb_batch_mean_id,
                dfb_old_running_mean_id,
                dfb_updated_running_mean_id,
                /*AlsoOut0=*/!old_running_var_has_value,
                dfb_one_id,
                dfb_momentum_id,
                dfb_out0_id>();
            maybe_typecast_stat<
                needs_mean_typecast,
                tc_in_fmt,
                tc_out_fmt,
                dfb_updated_running_mean_id,
                dfb_writer_updated_mean_id>();
        }

        if constexpr (old_running_var_has_value) {
            update_running_stat<
                dfb_batch_var_id,
                dfb_old_running_var_id,
                dfb_updated_running_var_id,
                /*AlsoOut0=*/true,
                dfb_one_id,
                dfb_momentum_id,
                dfb_out0_id>();
            maybe_typecast_stat<
                needs_var_typecast,
                tc_in_fmt,
                tc_out_fmt,
                dfb_updated_running_var_id,
                dfb_writer_updated_var_id>();
        }

        DataflowBuffer(dfb_out0_id).push_back(onetile);
        dfb_batch_mean_obj.pop_front(onetile);
        dfb_batch_var_obj.pop_front(onetile);
    }

    DataflowBuffer(dfb_momentum_id).pop_front(1);
    DataflowBuffer(dfb_one_id).pop_front(1);
}
