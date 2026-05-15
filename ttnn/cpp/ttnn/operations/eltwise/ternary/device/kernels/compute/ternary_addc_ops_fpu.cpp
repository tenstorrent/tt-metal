// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/common.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_scalar.hpp"  // MulUnary

namespace cklib = compute_kernel_lib;

template <bool ScalarIsNotOne>
inline void run_addc_fpu(uint32_t num_tiles, uint32_t scalar_arg) {
    constexpr auto cb_in0 = tt::CBIndex::c_0;  // input_a
    constexpr auto cb_in1 = tt::CBIndex::c_1;  // input_b
    constexpr auto cb_in2 = tt::CBIndex::c_2;  // input_c
    constexpr auto cb_out = tt::CBIndex::c_3;

    if constexpr (ScalarIsNotOne) {
        // out = input_a + scalar * input_b * input_c
        //   D0 = input_b * input_c (FPU binary, streaming WaitAndPop on both inputs)
        //   D0 = D0 * scalar       (MulUnary)
        //   D0 = input_a + D0      (DestReuseBinary, ELWADD, DEST_TO_SRCA — CB→srcb, DEST→srca)
        cklib::eltwise_chain(
            num_tiles,
            cklib::BinaryFpu<cb_in1, cb_in2, cklib::BinaryFpuOp::Mul>{},
            cklib::MulUnary<cklib::Dst::D0>{scalar_arg},
            cklib::DestReuseBinary<
                cb_in0,
                cklib::BinaryFpuOp::Add,
                cklib::DestReuseType::DEST_TO_SRCA,
                cklib::Dst::D0,
                cklib::Dst::D0,
                cklib::DestReuseReconfig::None,
                cklib::CopyTilePolicy::WaitAndPop,
                cklib::CbIndexMode::FirstTile>{},
            cklib::PackTile<
                cb_out,
                cklib::Dst::D0,
                cklib::PackTilePolicy::PerTileReserveAndPush,
                PackTileIndexMode::FirstTile,
                PackTileReconfig::None>{});
    } else {
        (void)scalar_arg;
        // out = input_a + input_b * input_c (no scalar)
        cklib::eltwise_chain(
            num_tiles,
            cklib::BinaryFpu<cb_in1, cb_in2, cklib::BinaryFpuOp::Mul>{},
            cklib::DestReuseBinary<
                cb_in0,
                cklib::BinaryFpuOp::Add,
                cklib::DestReuseType::DEST_TO_SRCA,
                cklib::Dst::D0,
                cklib::Dst::D0,
                cklib::DestReuseReconfig::None,
                cklib::CopyTilePolicy::WaitAndPop,
                cklib::CbIndexMode::FirstTile>{},
            cklib::PackTile<
                cb_out,
                cklib::Dst::D0,
                cklib::PackTilePolicy::PerTileReserveAndPush,
                PackTileIndexMode::FirstTile,
                PackTileReconfig::None>{});
    }
}

void kernel_main() {
    constexpr auto cb_in1 = tt::CBIndex::c_1;  // input_b — primary FPU srcA
    constexpr auto cb_in2 = tt::CBIndex::c_2;  // input_c — primary FPU srcB
    constexpr auto cb_out = tt::CBIndex::c_3;
    // D5/D8: caller-side BIG init at the top of MAIN(). The chain's first FPU
    // op is `BinaryFpu<cb_in1, cb_in2, Mul>` — boot for that triple.
    // (cb_in0 enters via DestReuseBinary's per-element srcb reconfig later.)
    compute_kernel_hw_startup(cb_in1, cb_in2, cb_out);

    uint32_t num_tiles = get_arg_val<uint32_t>(0);
    uint32_t scalar_arg = get_arg_val<uint32_t>(3);
    constexpr uint32_t num_tiles_per_cycle = get_compile_time_arg_val(0);  // set to 1
    (void)num_tiles_per_cycle;

    if (scalar_arg != 1u) {
        run_addc_fpu<true>(num_tiles, scalar_arg);
    } else {
        run_addc_fpu<false>(num_tiles, scalar_arg);
    }
}
