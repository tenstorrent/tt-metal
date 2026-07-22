// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/eltwise_binary.h"                // binary_op_init_common (BIG init)
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"  // BinaryFpu, DestReuseBinary, PackTile
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_binary_sfpu_basic.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_optional.hpp"  // OptionalChainElement
#include "api/dataflow/circular_buffer.h"

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
            ckl::input(cb_one, ckl::InputLifecycle::CallerManaged),
            ckl::input(cb_momentum, ckl::InputLifecycle::CallerManaged),
            BinaryFpuOp::Sub,
            ckl::BroadcastDim::None>{},  // D0 = 1 - momentum
        ckl::DestReuseBinary<ckl::input(cb_old), BinaryFpuOp::Mul, ckl::DestReuseType::DEST_TO_SRCA>{},  // D0 = (1 -
                                                                                                         // momentum) *
                                                                                                         // old_stat
        ckl::BinaryFpu<
            ckl::input(cb_momentum, ckl::InputLifecycle::CallerManaged),
            ckl::input(cb_batch),
            BinaryFpuOp::Mul,
            ckl::BroadcastDim::None,
            D::D1>{},                           // D1 = momentum * batch_stat
        ckl::AddBinary<D::D0, D::D1, D::D0>{},  // D0 = D0 + D1
        ckl::PackTile<ckl::output(cb_updated, ckl::OutputLifecycle::Bulk)>{},
        ckl::
            OptionalChainElement<AlsoOut0, ckl::PackTile<ckl::output(cb_out0, ckl::OutputLifecycle::CallerManaged)>>{});
}

void kernel_main() {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);
    constexpr uint32_t old_running_mean_has_value = get_compile_time_arg_val(0) == 1;
    constexpr uint32_t old_running_var_has_value = get_compile_time_arg_val(1) == 1;

    constexpr auto cb_batch_mean = get_compile_time_arg_val(2);  // batch mean
    constexpr auto cb_batch_var = get_compile_time_arg_val(3);   // batch var
    constexpr auto cb_out0 = get_compile_time_arg_val(4);
    constexpr auto cb_old_running_mean = get_compile_time_arg_val(5);      // old running mean tensor
    constexpr auto cb_old_running_var = get_compile_time_arg_val(6);       // old running var tensor
    constexpr auto cb_updated_running_mean = get_compile_time_arg_val(7);  // updated running mean tensor
    constexpr auto cb_updated_running_var = get_compile_time_arg_val(8);   // updated running var tensor
    constexpr auto cb_momentum = get_compile_time_arg_val(9);              // momentum
    constexpr auto cb_one = get_compile_time_arg_val(10);                  // stores 1

    binary_op_init_common(cb_batch_mean, cb_batch_var, cb_out0);
    constexpr uint32_t onetile = 1;

    cb_wait_front(cb_one, 1);
    cb_wait_front(cb_momentum, 1);

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

        cb_push_back(cb_out0, onetile);
    }

    cb_pop_front(cb_one, 1);
    cb_pop_front(cb_momentum, 1);
}
