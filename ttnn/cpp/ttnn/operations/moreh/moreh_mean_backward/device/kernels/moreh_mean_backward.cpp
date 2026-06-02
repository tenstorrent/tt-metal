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
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_optional.hpp"

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

    // Stage A: cb_intermed0 = add_bcast<dim>(cb_in1, cb_in0) (or plain copy if no bcast).
    //   bcast dim chosen at compile time from (ht_need_bcast, wt_need_bcast).
    //   cb_in1 InputLifecycle::CallerManaged + Scalar (held outside the loop).
    //   cb_in0 InputLifecycle::Streaming + Scalar (chain owns wait+pop).
    //   cb_intermed0 OutputLifecycle::Streaming + Scalar.
    // Reconfig: *_init_short_with_dt + pack_tile_with_dt -> Input + Output.
    //
    // Four (ht_need_bcast × wt_need_bcast) cases collapse into one chain via two
    // mutually-exclusive OptionalChainElement gates: BinaryFpu<Add, bcast_dim> runs
    // when any bcast is needed (Scalar / Row / Col); CopyTile runs when neither is
    // needed (faster than add(zero, x) for the no-bcast path).
    constexpr bool has_bcast = ht_need_bcast || wt_need_bcast;
    constexpr auto bcast_dim = (ht_need_bcast && wt_need_bcast) ? compute_kernel_lib::BroadcastDim::Scalar
                               : ht_need_bcast                  ? compute_kernel_lib::BroadcastDim::Row
                               : wt_need_bcast                  ? compute_kernel_lib::BroadcastDim::Col
                                                                : compute_kernel_lib::BroadcastDim::None;

    for (uint32_t i = 0; i < num_output_tiles; i++) {
        compute_kernel_lib::eltwise_chain(
            onetile,
            compute_kernel_lib::OptionalChainElement<
                has_bcast,
                compute_kernel_lib::BinaryFpu<
                    cb_in1,
                    cb_in0,
                    compute_kernel_lib::BinaryFpuOp::Add,
                    bcast_dim,
                    compute_kernel_lib::BinaryDataFormatReconfig::Input,
                    compute_kernel_lib::InputLifecycle::CallerManaged,
                    compute_kernel_lib::InputLifecycle::Streaming,
                    compute_kernel_lib::OperandKind::Scalar,
                    compute_kernel_lib::Dst::D0,
                    compute_kernel_lib::OperandKind::Scalar>>{},
            compute_kernel_lib::OptionalChainElement<
                !has_bcast,
                compute_kernel_lib::CopyTile<
                    cb_in0,
                    compute_kernel_lib::Dst::D0,
                    compute_kernel_lib::InputLifecycle::Streaming,
                    compute_kernel_lib::OperandKind::Scalar,
                    compute_kernel_lib::CopyTileReconfig::Input>>{},
            compute_kernel_lib::PackTile<
                cb_intermed0,
                compute_kernel_lib::OutputLifecycle::Streaming,
                compute_kernel_lib::PackTileReconfig::Output>{});

        // Stage B: cb_out0 = cb_intermed0 * cb_scalar (SCALAR bcast).
        compute_kernel_lib::eltwise_chain(
            onetile,
            compute_kernel_lib::BinaryFpu<
                cb_intermed0,
                cb_scalar,
                compute_kernel_lib::BinaryFpuOp::Mul,
                compute_kernel_lib::BroadcastDim::Scalar,
                compute_kernel_lib::BinaryDataFormatReconfig::Input,
                compute_kernel_lib::InputLifecycle::Streaming,
                compute_kernel_lib::InputLifecycle::CallerManaged,
                compute_kernel_lib::OperandKind::Scalar,
                compute_kernel_lib::Dst::D0,
                compute_kernel_lib::OperandKind::Scalar>{},
            compute_kernel_lib::PackTile<
                cb_out0,
                compute_kernel_lib::OutputLifecycle::Streaming,
                compute_kernel_lib::PackTileReconfig::Output>{});
    }
    cb_in1_obj.pop_front(onetile);
}
