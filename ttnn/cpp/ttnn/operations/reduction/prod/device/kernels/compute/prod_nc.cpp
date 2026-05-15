// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"

void kernel_main() {
    const auto num_input_tiles = get_arg_val<uint32_t>(0);
    const auto num_output_tiles = get_arg_val<uint32_t>(1);

    constexpr auto cb_in0 = tt::CBIndex::c_0;
    constexpr auto cb_in1 = tt::CBIndex::c_1;
    constexpr auto cb_out0 = tt::CBIndex::c_3;
    constexpr auto cb_intermed0 = tt::CBIndex::c_2;

    // Caller-init: boot engine. The union CB triple covers all stages — cb_in0
    // is the persistent srcA across every stage; B-side cycles between cb_in1
    // (seed) and cb_intermed0 (feedback); output cycles between cb_intermed0
    // (middle iters) and cb_out0 (final iter). One boot is sufficient — each
    // chain's reconfig fold emits per-element reconfigs.
    compute_kernel_hw_startup(cb_in0, cb_intermed0, cb_out0);

    // The reader pre-waits cb_in1 (scalar/seed tile). The chain treats it as
    // caller-managed via NoWaitNoPop on the B-side.
    cb_wait_front(cb_in1, 1);

    for (uint32_t i = 0; i < num_output_tiles; i++) {
        if (num_input_tiles == 1) {
            // Single-input case: cb_out0 = cb_in0 * cb_in1.
            compute_kernel_lib::eltwise_chain(
                1,
                compute_kernel_lib::BinaryFpu<
                    cb_in0,
                    cb_in1,
                    compute_kernel_lib::BinaryFpuOp::Mul,
                    compute_kernel_lib::BroadcastDim::None,
                    compute_kernel_lib::BinaryDataFormatReconfig::Input,
                    compute_kernel_lib::CopyTilePolicy::WaitAndPop,
                    compute_kernel_lib::CopyTilePolicy::NoWaitNoPop,
                    compute_kernel_lib::CbIndexMode::FirstTile,
                    compute_kernel_lib::Dst::D0>{},
                compute_kernel_lib::PackTile<
                    cb_out0,
                    compute_kernel_lib::Dst::D0,
                    compute_kernel_lib::PackTilePolicy::PerTileReserveAndPush,
                    compute_kernel_lib::PackTileIndexMode::FirstTile,
                    compute_kernel_lib::PackTileReconfig::Output>{});
        } else {
            // Seed: cb_intermed0 = cb_in0 * cb_in1 (cb_in1 caller-managed).
            compute_kernel_lib::eltwise_chain(
                1,
                compute_kernel_lib::BinaryFpu<
                    cb_in0,
                    cb_in1,
                    compute_kernel_lib::BinaryFpuOp::Mul,
                    compute_kernel_lib::BroadcastDim::None,
                    compute_kernel_lib::BinaryDataFormatReconfig::Input,
                    compute_kernel_lib::CopyTilePolicy::WaitAndPop,
                    compute_kernel_lib::CopyTilePolicy::NoWaitNoPop,
                    compute_kernel_lib::CbIndexMode::FirstTile,
                    compute_kernel_lib::Dst::D0>{},
                compute_kernel_lib::PackTile<
                    cb_intermed0,
                    compute_kernel_lib::Dst::D0,
                    compute_kernel_lib::PackTilePolicy::PerTileReserveAndPush,
                    compute_kernel_lib::PackTileIndexMode::FirstTile,
                    compute_kernel_lib::PackTileReconfig::Output>{});

            // Middle iters: cb_intermed0 = cb_in0 * cb_intermed0 (in-place).
            for (uint32_t j = 1; j < num_input_tiles - 1; ++j) {
                compute_kernel_lib::eltwise_chain(
                    1,
                    compute_kernel_lib::BinaryFpu<
                        cb_in0,
                        cb_intermed0,
                        compute_kernel_lib::BinaryFpuOp::Mul,
                        compute_kernel_lib::BroadcastDim::None,
                        compute_kernel_lib::BinaryDataFormatReconfig::Input,
                        compute_kernel_lib::CopyTilePolicy::WaitAndPop,
                        compute_kernel_lib::CopyTilePolicy::WaitAndPop,
                        compute_kernel_lib::CbIndexMode::FirstTile,
                        compute_kernel_lib::Dst::D0>{},
                    compute_kernel_lib::PackTile<
                        cb_intermed0,
                        compute_kernel_lib::Dst::D0,
                        compute_kernel_lib::PackTilePolicy::PerTileReserveAndPush,
                        compute_kernel_lib::PackTileIndexMode::FirstTile,
                        compute_kernel_lib::PackTileReconfig::Output>{});
            }

            // Final iter: cb_out0 = cb_in0 * cb_intermed0.
            compute_kernel_lib::eltwise_chain(
                1,
                compute_kernel_lib::BinaryFpu<
                    cb_in0,
                    cb_intermed0,
                    compute_kernel_lib::BinaryFpuOp::Mul,
                    compute_kernel_lib::BroadcastDim::None,
                    compute_kernel_lib::BinaryDataFormatReconfig::Input,
                    compute_kernel_lib::CopyTilePolicy::WaitAndPop,
                    compute_kernel_lib::CopyTilePolicy::WaitAndPop,
                    compute_kernel_lib::CbIndexMode::FirstTile,
                    compute_kernel_lib::Dst::D0>{},
                compute_kernel_lib::PackTile<
                    cb_out0,
                    compute_kernel_lib::Dst::D0,
                    compute_kernel_lib::PackTilePolicy::PerTileReserveAndPush,
                    compute_kernel_lib::PackTileIndexMode::FirstTile,
                    compute_kernel_lib::PackTileReconfig::Output>{});
        }
    }
}
