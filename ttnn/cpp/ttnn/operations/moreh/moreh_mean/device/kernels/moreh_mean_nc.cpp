// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "ttnn/kernel/compute/moreh_common.hpp"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"

void kernel_main() {
    const auto num_input_tiles = get_arg_val<uint32_t>(0);
    const auto num_output_tiles = get_arg_val<uint32_t>(1);

    constexpr auto cb_in0 = tt::CBIndex::c_0;
    CircularBuffer cb_in0_obj(cb_in0);
    constexpr auto cb_in1 = tt::CBIndex::c_1;
    CircularBuffer cb_in1_obj(cb_in1);
    constexpr auto cb_scalar = tt::CBIndex::c_2;
    CircularBuffer cb_scalar_obj(cb_scalar);
    constexpr auto cb_out0 = tt::CBIndex::c_16;
    CircularBuffer cb_out0_obj(cb_out0);
    constexpr auto cb_intermed0 = tt::CBIndex::c_24;
    CircularBuffer cb_intermed0_obj(cb_intermed0);
    constexpr uint32_t onetile = 1;
    constexpr uint32_t dst0 = 0;
    constexpr uint32_t dst1 = 1;
    constexpr uint32_t first_tile = 0;

    binary_op_init_common(tt::CBIndex::c_0, tt::CBIndex::c_1, tt::CBIndex::c_16);

    cb_in1_obj.wait_front(onetile);
    cb_scalar_obj.wait_front(1);  // scalar tile from the reader

    for (uint32_t i = 0; i < num_output_tiles; i++) {
        bool enable_reload = false;
        for (uint32_t j = 0; j < num_input_tiles; ++j) {
            // Accumulator add: cb_intermed0 = cb_in0 + (reload? cb_intermed0 : cb_in1)
            // First iter (!reload): B=cb_in1 (InputLifecycle::CallerManaged — pre-waited, never popped).
            // Subsequent iters (reload): B=cb_intermed0 (InputLifecycle::Streaming, same-CB in/out).
            // cb_in0 InputLifecycle::Streaming on A. cb_intermed0 OutputLifecycle::Streaming.
            // Reconfig: add_tiles_init_with_dt + pack_tile_with_dt -> Input + Output.
            if (enable_reload) {
                compute_kernel_lib::eltwise_chain(
                    onetile,
                    compute_kernel_lib::BinaryFpu<
                        cb_in0,
                        cb_intermed0,
                        compute_kernel_lib::BinaryFpuOp::Add,
                        compute_kernel_lib::BroadcastDim::None,
                        compute_kernel_lib::BinaryDataFormatReconfig::Input,
                        compute_kernel_lib::InputLifecycle::Streaming,
                        compute_kernel_lib::InputLifecycle::Streaming,
                        compute_kernel_lib::OperandKind::Scalar,
                        compute_kernel_lib::Dst::D0,
                        compute_kernel_lib::OperandKind::Scalar>{},
                    compute_kernel_lib::PackTile<
                        cb_intermed0,
                        compute_kernel_lib::OutputLifecycle::Streaming,
                        compute_kernel_lib::PackTileReconfig::Output>{});
            } else {
                compute_kernel_lib::eltwise_chain(
                    onetile,
                    compute_kernel_lib::BinaryFpu<
                        cb_in0,
                        cb_in1,
                        compute_kernel_lib::BinaryFpuOp::Add,
                        compute_kernel_lib::BroadcastDim::None,
                        compute_kernel_lib::BinaryDataFormatReconfig::Input,
                        compute_kernel_lib::InputLifecycle::Streaming,
                        compute_kernel_lib::InputLifecycle::CallerManaged,
                        compute_kernel_lib::OperandKind::Scalar,
                        compute_kernel_lib::Dst::D0,
                        compute_kernel_lib::OperandKind::Scalar>{},
                    compute_kernel_lib::PackTile<
                        cb_intermed0,
                        compute_kernel_lib::OutputLifecycle::Streaming,
                        compute_kernel_lib::PackTileReconfig::Output>{});
            }

            enable_reload = true;
        }

        // output * (1 / number_of_elements)  — SCALAR bcast Mul.
        // cb_intermed0 InputLifecycle::Streaming (last iter's push, popped here);
        // cb_scalar InputLifecycle::CallerManaged + Scalar (pre-waited at top of kernel);
        // cb_out0 OutputLifecycle::Streaming.
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
}
