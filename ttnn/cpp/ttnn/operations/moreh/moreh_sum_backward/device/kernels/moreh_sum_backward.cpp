// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/kernel/compute/moreh_common.hpp"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
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
    constexpr auto cb_out0 = tt::CBIndex::c_16;
    CircularBuffer cb_out0_obj(cb_out0);
    constexpr uint32_t onetile = 1;
    constexpr uint32_t dst0 = 0;

    binary_op_init_common(cb_in1, cb_in0, cb_out0);
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
                    ckl::input(cb_in1, ckl::InputLifecycle::CallerManaged, ckl::DataFormatReconfig::Disabled),
                    ckl::input(cb_in0, ckl::InputLifecycle::Streaming, ckl::DataFormatReconfig::Disabled),
                    ckl::BinaryFpuOp::Add,
                    bcast_dim>>{},
            ckl::OptionalChainElement<
                !has_bcast,
                ckl::CopyTile<
                    ckl::input(cb_in0, ckl::InputLifecycle::Streaming, ckl::DataFormatReconfig::Disabled),
                    ckl::Dst::D0>>{},
            ckl::PackTile<ckl::output(cb_out0, ckl::OutputLifecycle::Streaming, ckl::DataFormatReconfig::Disabled)>{});
    }
    cb_in1_obj.pop_front(onetile);
}
