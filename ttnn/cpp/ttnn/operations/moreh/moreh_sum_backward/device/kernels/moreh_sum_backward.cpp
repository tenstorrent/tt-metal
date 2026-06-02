// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

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
    constexpr auto cb_out0 = tt::CBIndex::c_16;
    CircularBuffer cb_out0_obj(cb_out0);
    constexpr uint32_t onetile = 1;
    constexpr uint32_t dst0 = 0;

    binary_op_init_common(cb_in1, cb_in0, cb_out0);
    cb_in1_obj.wait_front(onetile);

    // cb_out0 = add_bcast<dim>(cb_in1, cb_in0)  (or plain copy if no bcast).
    // Reconfig: original uses *_init_short (NOT _with_dt) and plain pack_tile,
    // relying on startup binary_op_init_common formats -> None + None.
    // cb_in1 InputLifecycle::CallerManaged + Scalar (held outside loop). cb_in0 InputLifecycle::Streaming + Scalar.
    // cb_out0 OutputLifecycle::Streaming + Scalar.
    //
    // The four (ht_need_bcast × wt_need_bcast) cases collapse into one chain via two
    // mutually-exclusive OptionalChainElement gates: the BinaryFpu<Add, bcast_dim>
    // runs when any bcast is needed (Scalar / Row / Col); the CopyTile runs when
    // neither is needed (faster than add(zero, x) for the no-bcast path).
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
                    compute_kernel_lib::BinaryDataFormatReconfig::None,
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
                    compute_kernel_lib::CopyTileReconfig::None>>{},
            compute_kernel_lib::PackTile<
                cb_out0,
                compute_kernel_lib::OutputLifecycle::Streaming,
                compute_kernel_lib::PackTileReconfig::None>{});
    }
    cb_in1_obj.pop_front(onetile);
}
