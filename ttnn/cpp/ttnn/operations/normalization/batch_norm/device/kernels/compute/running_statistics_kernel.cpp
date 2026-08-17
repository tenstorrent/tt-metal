// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/core/chain.hpp"  // BinaryFpu, DestReuseBinary, PackTile
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/binary/sfpu/basic.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/core/optional.hpp"  // Optional
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

namespace ckl = compute_kernel_lib;

template <uint32_t dfb_batch_id, uint32_t dfb_old_id, uint32_t dfb_updated_id, bool AlsoOut0>
ALWI void update_running_stat() {
    using D = ckl::Dst;
    using ckl::BinaryFpuOp;

    ckl::eltwise_chain(
        ckl::IterationShape::one_tile(),
        ckl::BinaryFpu<
            BinaryFpuOp::Sub,
            ckl::input(dfb::one, ckl::WaitPolicy::None, ckl::PopPolicy::None),
            ckl::input(dfb::momentum, ckl::WaitPolicy::None, ckl::PopPolicy::None)>{},  // D0 = 1 - momentum
        ckl::
            DestReuseBinary<ckl::input(dfb_old_id), BinaryFpuOp::Mul, ckl::DestReuseType::DEST_TO_SRCA>{},  // D0 = (1
                                                                                                            // -
                                                                                                            // momentum)
                                                                                                            // *
                                                                                                            // old_stat
        ckl::BinaryFpu<
            BinaryFpuOp::Mul,
            ckl::input(dfb::momentum, ckl::WaitPolicy::None, ckl::PopPolicy::None),
            ckl::input(dfb_batch_id, ckl::WaitPolicy::None, ckl::PopPolicy::None),
            D::D1>{},                           // D1 = momentum * batch_stat
        ckl::AddBinary<D::D0, D::D1, D::D0>{},  // D0 = D0 + D1
        ckl::PackTile<ckl::output(dfb_updated_id, ckl::ReservePolicy::Upfront, ckl::PushPolicy::AtEnd)>{},
        ckl::Optional<
            AlsoOut0,
            ckl::PackTile<ckl::output(dfb::out, ckl::ReservePolicy::None, ckl::PushPolicy::None)>>{});
}

void kernel_main() {
    uint32_t num_tiles = get_arg(args::num_tiles);
    constexpr uint32_t old_running_mean_has_value = get_arg(args::old_running_mean_has_value) == 1;
    constexpr uint32_t old_running_var_has_value = get_arg(args::old_running_var_has_value) == 1;
    static_assert(
        old_running_mean_has_value || old_running_var_has_value,
        "running_statistics requires at least one of running_mean / running_var");

    DataflowBuffer dfb_batch_mean_obj(dfb::batch_mean);
    DataflowBuffer dfb_batch_var_obj(dfb::batch_var);

    compute_kernel_hw_startup(dfb::batch_mean, dfb::batch_var, dfb::out);
    constexpr uint32_t onetile = 1;

    DataflowBuffer(dfb::one).wait_front(1);
    DataflowBuffer(dfb::momentum).wait_front(1);

    for (uint32_t tile_id = 0; tile_id < num_tiles; ++tile_id) {
        // The reader produces both batch-stat streams for every tile, even when only one
        // running statistic is requested. Consume both streams unconditionally to avoid
        // filling either two-entry buffer and stalling its producer.
        dfb_batch_mean_obj.wait_front(onetile);
        dfb_batch_var_obj.wait_front(onetile);
        DataflowBuffer(dfb::out).reserve_back(onetile);

        if constexpr (old_running_mean_has_value) {
            update_running_stat<
                dfb::batch_mean,
                dfb::old_running_mean,
                dfb::updated_mean,
                /*AlsoOut0=*/!old_running_var_has_value>();
        }

        if constexpr (old_running_var_has_value) {
            update_running_stat<
                dfb::batch_var,
                dfb::old_running_var,
                dfb::updated_var,
                /*AlsoOut0=*/true>();
        }

        DataflowBuffer(dfb::out).push_back(onetile);
        dfb_batch_mean_obj.pop_front(onetile);
        dfb_batch_var_obj.pop_front(onetile);
    }

    DataflowBuffer(dfb::one).pop_front(1);
    DataflowBuffer(dfb::momentum).pop_front(1);
}
