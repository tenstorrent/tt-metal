// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"  // BinaryFpu, DestReuseBinary, PackTile
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_binary_sfpu_basic.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_optional.hpp"  // OptionalChainElement
#include "api/dataflow/dataflow_buffer.h"

namespace ckl = compute_kernel_lib;

template <
    uint32_t cb_batch,
    uint32_t cb_old,
    uint32_t cb_updated,
    bool AlsoOut0,
    uint32_t cb_one,
    uint32_t cb_momentum,
    uint32_t cb_out0>
ALWI void update_running_stat() {
    using D = ckl::Dst;
    using ckl::BinaryFpuOp;

    ckl::eltwise_chain(
        ckl::EltwiseShape::single(),
        ckl::BinaryFpu<
            ckl::input(cb_one, ckl::WaitPolicy::None, ckl::PopPolicy::None),
            ckl::input(cb_momentum, ckl::WaitPolicy::None, ckl::PopPolicy::None),
            BinaryFpuOp::Sub,
            ckl::BroadcastDim::None>{},  // D0 = 1 - momentum
        ckl::DestReuseBinary<ckl::input(cb_old), BinaryFpuOp::Mul, ckl::DestReuseType::DEST_TO_SRCA>{},  // D0 = (1 -
                                                                                                         // momentum) *
                                                                                                         // old_stat
        ckl::BinaryFpu<
            ckl::input(cb_momentum, ckl::WaitPolicy::None, ckl::PopPolicy::None),
            ckl::input(cb_batch, ckl::WaitPolicy::None, ckl::PopPolicy::None),
            BinaryFpuOp::Mul,
            ckl::BroadcastDim::None,
            D::D1>{},                           // D1 = momentum * batch_stat
        ckl::AddBinary<D::D0, D::D1, D::D0>{},  // D0 = D0 + D1
        ckl::PackTile<ckl::output(cb_updated, ckl::ReservePolicy::Upfront, ckl::PushPolicy::AtEnd)>{},
        ckl::OptionalChainElement<
            AlsoOut0,
            ckl::PackTile<ckl::output(cb_out0, ckl::ReservePolicy::None, ckl::PushPolicy::None)>>{});
}

void kernel_main() {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);
    constexpr uint32_t old_running_mean_has_value = get_compile_time_arg_val(0) == 1;
    constexpr uint32_t old_running_var_has_value = get_compile_time_arg_val(1) == 1;
    static_assert(
        old_running_mean_has_value || old_running_var_has_value,
        "running_statistics requires at least one of running_mean / running_var");

    constexpr auto cb_batch_mean = get_compile_time_arg_val(2);  // batch mean
    constexpr auto cb_batch_var = get_compile_time_arg_val(3);   // batch var
    constexpr auto cb_out0 = get_compile_time_arg_val(4);
    constexpr auto cb_old_running_mean = get_compile_time_arg_val(5);      // old running mean tensor
    constexpr auto cb_old_running_var = get_compile_time_arg_val(6);       // old running var tensor
    constexpr auto cb_updated_running_mean = get_compile_time_arg_val(7);  // updated running mean tensor
    constexpr auto cb_updated_running_var = get_compile_time_arg_val(8);   // updated running var tensor
    constexpr auto cb_momentum = get_compile_time_arg_val(9);              // momentum
    constexpr auto cb_one = get_compile_time_arg_val(10);                  // stores 1

    DataflowBuffer cb_batch_mean_obj(cb_batch_mean);
    DataflowBuffer cb_batch_var_obj(cb_batch_var);

    compute_kernel_hw_startup(cb_batch_mean, cb_batch_var, cb_out0);
    constexpr uint32_t onetile = 1;

    DataflowBuffer(cb_one).wait_front(1);
    DataflowBuffer(cb_momentum).wait_front(1);

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
        }

        DataflowBuffer(cb_out0).push_back(onetile);
        cb_batch_mean_obj.pop_front(onetile);
        cb_batch_var_obj.pop_front(onetile);
    }

    DataflowBuffer(cb_one).pop_front(1);
    DataflowBuffer(cb_momentum).pop_front(1);
}
