// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "ttnn/kernel/compute/moreh_common.hpp"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_optional.hpp"

namespace ckl = compute_kernel_lib;

void kernel_main() {
    // compile-time args
    constexpr uint32_t num_output_tiles = get_compile_time_arg_val(0);
    constexpr bool wt_need_bcast = (get_compile_time_arg_val(1) == 1);
    constexpr bool ht_need_bcast = (get_compile_time_arg_val(2) == 1);

    constexpr auto cb_in0 = tt::CBIndex::c_0;
    CircularBuffer cb_in0_obj(cb_in0);  // input
    constexpr auto cb_in1 = tt::CBIndex::c_1;
    CircularBuffer cb_in1_obj(cb_in1);  // zero tile
    constexpr auto cb_scalar = tt::CBIndex::c_2;
    constexpr auto cb_out0 = tt::CBIndex::c_16;
    CircularBuffer cb_out0_obj(cb_out0);
    constexpr auto cb_intermed0 = tt::CBIndex::c_24;
    CircularBuffer cb_intermed0_obj(cb_intermed0);
    constexpr uint32_t onetile = 1;
    constexpr uint32_t dst0 = 0;

    binary_op_init_common(tt::CBIndex::c_0, tt::CBIndex::c_1, tt::CBIndex::c_16);
    cb_in1_obj.wait_front(onetile);

    constexpr bool has_bcast = ht_need_bcast || wt_need_bcast;
    constexpr auto bcast_dim = (ht_need_bcast && wt_need_bcast) ? ckl::BroadcastDim::Scalar
                               : ht_need_bcast                  ? ckl::BroadcastDim::Row
                               : wt_need_bcast                  ? ckl::BroadcastDim::Col
                                                                : ckl::BroadcastDim::None;

    for (uint32_t i = 0; i < num_output_tiles; i++) {
        ckl::eltwise_chain(
            ckl::EltwiseShape::tiles(onetile),
            ckl::OptionalChainElement<
                has_bcast,
                ckl::BinaryFpu<
                    cb_in1,
                    cb_in0,
                    ckl::BinaryFpuOp::Add,
                    bcast_dim,
                    ckl::input(ckl::InputLifecycle::CallerManaged)>>{},
            ckl::OptionalChainElement<!has_bcast, ckl::CopyTile<cb_in0>>{},
            ckl::PackTile<cb_intermed0>{});

        ckl::mul<
            cb_intermed0,
            cb_scalar,
            cb_out0,
            ckl::BroadcastDim::Scalar,
            ckl::input(),
            ckl::input(ckl::InputLifecycle::CallerManaged)>(ckl::EltwiseShape::tiles(onetile));
    }
    cb_in1_obj.pop_front(onetile);
}
