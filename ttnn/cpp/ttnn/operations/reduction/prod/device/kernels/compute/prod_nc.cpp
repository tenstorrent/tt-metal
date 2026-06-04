// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    const auto num_input_tiles = get_arg_val<uint32_t>(0);
    const auto num_output_tiles = get_arg_val<uint32_t>(1);

    constexpr auto cb_in0 = tt::CBIndex::c_0;
    constexpr auto cb_in1 = tt::CBIndex::c_1;
    constexpr auto cb_out0 = tt::CBIndex::c_3;
    constexpr auto cb_intermed0 = tt::CBIndex::c_2;
    constexpr uint32_t onetile = 1;

    CircularBuffer cb_in1_obj(cb_in1);

    binary_op_init_common(cb_in0, cb_in1, cb_out0);
    cb_in1_obj.wait_front(onetile);

    // Reduction by product across num_input_tiles per output tile.
    // First iteration multiplies cb_in0 by cb_in1 (ones scaler held outside);
    // subsequent iterations multiply cb_in0 by the running cb_intermed0
    // partial. Final iteration packs to cb_out0 instead of cb_intermed0.
    //
    // Reconfig: original used `mul_tiles_init(cb_in0, cb_add)` (NOT _with_dt)
    // each iteration and plain `pack_tile` — no reconfigs. BinaryFpu emits
    // Input reconfig per chain call which is a no-op when the data format
    // doesn't change (cb_in0/cb_in1/cb_intermed0 share intermed_data_format
    // by program-factory layout).
    for (uint32_t i = 0; i < num_output_tiles; ++i) {
        if (num_input_tiles == 1) {
            // Single input tile: cb_in0 * cb_in1 -> cb_out0.
            compute_kernel_lib::mul<
                cb_in0,
                cb_in1,
                cb_out0,
                compute_kernel_lib::BroadcastDim::None,
                compute_kernel_lib::BinaryDataFormatReconfig::Input,
                compute_kernel_lib::OperandKind::Scalar,
                compute_kernel_lib::InputLifecycle::Streaming,
                compute_kernel_lib::InputLifecycle::HeldStream>(onetile);
        } else {
            // Seed: cb_in0 * cb_in1 -> cb_intermed0 (cb_in1 = scaler, held).
            compute_kernel_lib::mul<
                cb_in0,
                cb_in1,
                cb_intermed0,
                compute_kernel_lib::BroadcastDim::None,
                compute_kernel_lib::BinaryDataFormatReconfig::Input,
                compute_kernel_lib::OperandKind::Scalar,
                compute_kernel_lib::InputLifecycle::Streaming,
                compute_kernel_lib::InputLifecycle::HeldStream>(onetile);
            // Middle: cb_in0 * cb_intermed0 -> cb_intermed0 (n_input - 2 iters).
            if (num_input_tiles > 2) {
                compute_kernel_lib::mul<cb_in0, cb_intermed0, cb_intermed0>(num_input_tiles - 2u);
            }
            // Final: cb_in0 * cb_intermed0 -> cb_out0.
            compute_kernel_lib::mul<cb_in0, cb_intermed0, cb_out0>(onetile);
        }
    }
}
