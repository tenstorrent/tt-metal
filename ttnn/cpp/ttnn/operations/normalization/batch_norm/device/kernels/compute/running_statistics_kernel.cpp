// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/eltwise_binary.h"                      // binary_op_init_common (BIG init)
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"        // BinaryFpu, DestReuseBinary, PackTile
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_binary_sfpu.hpp"  // AddBinary (DEST + DEST)
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_optional.hpp"     // OptionalChainElement
#include "api/dataflow/circular_buffer.h"

// running_statistics (FPU path): updated_stat = (1 - momentum) * old_stat + momentum * batch_stat
// for mean and/or var (compile-time gated).
//
// One chain per stat, no intermediate CBs (the original staged through cb_tmp1/2/3). The two
// products live in separate DEST slots and are summed in DEST by an SFPU add:
//
//   D0 = (1 - momentum) ;  D0 *= old_stat        BinaryFpu Sub(one, momentum) -> DestReuse Mul(.old)
//   D1 = momentum * batch_stat                    BinaryFpu Mul(momentum, batch) -> D1
//   D0 = D0 + D1                                  AddBinary (DEST + DEST)
//
// cb_out0 receives a copy of the last computed stat (var if present, else mean), reproducing the
// original's trailing pack_tile(0, cb_out0).
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

    // one/momentum held (CallerManaged); old_stat and batch_stat streamed (wait 1 / pop 1, as the
    // original popped them). cb_updated is caller-reserved/pushed (CallerManaged output); cb_out0 is
    // reserved/pushed once per loop iteration by kernel_main.
    cb_reserve_back(cb_updated, 1);
    ckl::eltwise_chain(
        1,
        ckl::BinaryFpu<
            cb_one,
            cb_momentum,
            BinaryFpuOp::Sub,
            ckl::BroadcastDim::None,
            ckl::InputLifecycle::CallerManaged,
            ckl::InputLifecycle::CallerManaged>{},                                           // D0 = 1 - momentum
        ckl::DestReuseBinary<cb_old, BinaryFpuOp::Mul, ckl::DestReuseType::DEST_TO_SRCA>{},  // D0 = (1 - momentum) *
                                                                                             // old_stat
        ckl::BinaryFpu<
            cb_momentum,
            cb_batch,
            BinaryFpuOp::Mul,
            ckl::BroadcastDim::None,
            ckl::InputLifecycle::CallerManaged,
            ckl::InputLifecycle::Streaming,
            ckl::BinaryDataFormatReconfig::Input,
            D::D1>{},                           // D1 = momentum * batch_stat
        ckl::AddBinary<D::D0, D::D1, D::D0>{},  // D0 = D0 + D1
        ckl::PackTile<cb_updated, ckl::OutputLifecycle::CallerManaged>{},
        ckl::OptionalChainElement<AlsoOut0, ckl::PackTile<cb_out0, ckl::OutputLifecycle::CallerManaged>>{});
    cb_push_back(cb_updated, 1);
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

    cb_wait_front(cb_one, 1);  // held for the whole kernel
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
