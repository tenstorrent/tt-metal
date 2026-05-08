// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/common.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"

// Map host-injected BCAST_LLKOP to chain's BinaryFpuOp.
#if BCAST_LLKOP == EltwiseBinaryType::ELWADD
constexpr auto FPU_OP = compute_kernel_lib::BinaryFpuOp::Add;
#elif BCAST_LLKOP == EltwiseBinaryType::ELWSUB
constexpr auto FPU_OP = compute_kernel_lib::BinaryFpuOp::Sub;
#elif BCAST_LLKOP == EltwiseBinaryType::ELWMUL
constexpr auto FPU_OP = compute_kernel_lib::BinaryFpuOp::Mul;
#else
#error "BCAST_LLKOP must be one of ELWADD / ELWSUB / ELWMUL"
#endif

void kernel_main() {
    using namespace compute_kernel_lib;

    constexpr uint32_t cb_a   = tt::CBIndex::c_0;
    constexpr uint32_t cb_b   = tt::CBIndex::c_1;
    constexpr uint32_t cb_out = tt::CBIndex::c_2;

    uint32_t B = get_arg_val<uint32_t>(0);
    uint32_t Ht = get_arg_val<uint32_t>(1);
    uint32_t Wt = get_arg_val<uint32_t>(2);
    const uint32_t num_tiles = B * Ht * Wt;

    // D5/D8: caller-side BIG init at the top of MAIN().
    compute_kernel_hw_startup(cb_a, cb_b, cb_out);

    using BinElt = BinaryFpu<
        cb_a,
        cb_b,
        FPU_OP,
        BroadcastDim::Row,
        BinaryFpuOutputPolicy::PerTile,
        BinaryDataFormatReconfig::None,
        CopyTilePolicy::WaitAndPop,
        CopyTilePolicy::WaitAndPop,
        CbIndexMode::FirstTile,
        CbIndexMode::FirstTile,
        Dst::D0,
        0,
        0,
        0,
        cb_out>;

    eltwise_chain(
        num_tiles,
        BinElt{},
        PackTile<cb_out, Dst::D0, PackTilePolicy::PerTileReserveAndPush>{}
    );
}
