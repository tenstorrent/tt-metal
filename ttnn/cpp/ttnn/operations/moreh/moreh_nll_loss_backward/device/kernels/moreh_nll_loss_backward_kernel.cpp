// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_math.hpp"  // Recip
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_misc.hpp"  // Negative
#include "ttnn/kernel/compute/moreh_common.hpp"
#include "api/compute/eltwise_unary/eltwise_unary.h"

void kernel_main() {
    constexpr uint32_t per_core_tile_cnt = get_compile_time_arg_val(0);

    const uint32_t tile_offset = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_divisor = tt::CBIndex::c_3;
    constexpr uint32_t cb_output_grad = tt::CBIndex::c_0;
    constexpr uint32_t cb_tmp_weight = tt::CBIndex::c_24;
    constexpr uint32_t cb_tmp1 = tt::CBIndex::c_25;
    constexpr uint32_t cb_tmp2 = tt::CBIndex::c_26;
    constexpr uint32_t cb_input_grad = tt::CBIndex::c_16;

    constexpr uint32_t onetile = 1;

    init_sfpu(cb_output_grad, tt::CBIndex::c_16);

    using namespace compute_kernel_lib;

#if defined(DIVISOR)
    // cb_tmp1 = 1 / cb_divisor (already migrated, kept).
    cb_wait_front(cb_divisor, onetile);
    eltwise_chain(
        onetile,
        CopyTile<cb_divisor, Dst::D0, CopyTilePolicy::NoWaitNoPop, CbIndexMode::FirstTile, CopyTileReconfig::Input>{},
        Recip<Dst::D0>{},
        PackTile<
            cb_tmp1,
            Dst::D0,
            PackTilePolicy::PerTileReserveAndPush,
            PackTileIndexMode::FirstTile,
            PackTileReconfig::Output,
            /*EnableFp32DestAcc=*/DST_ACCUM_MODE>{});
#endif

    cb_wait_front(cb_output_grad, onetile);

    for (uint32_t b = 0; b < per_core_tile_cnt; ++b) {
#if defined(DIVISOR)
        // T1.25: cb_tmp2 = -(cb_tmp_weight * cb_output_grad)  (bcast scalar, B=output_grad held)
        eltwise_chain(
            onetile,
            BinaryFpu<
                cb_tmp_weight,
                cb_output_grad,
                cb_tmp2,
                BinaryFpuOp::Mul,
                BroadcastDim::Scalar,
                BinaryDataFormatReconfig::InputAndOutput,
                CopyTilePolicy::WaitAndPop,
                CopyTilePolicy::NoWaitNoPop,
                CbIndexMode::FirstTile,
                Dst::D0,
                /*EnableFp32DestAcc=*/DST_ACCUM_MODE>{},
            Negative<Dst::D0>{},
            PackTile<
                cb_tmp2,
                Dst::D0,
                PackTilePolicy::PerTileReserveAndPush,
                PackTileIndexMode::FirstTile,
                PackTileReconfig::Output,
                /*EnableFp32DestAcc=*/DST_ACCUM_MODE>{});

        // T1.26: cb_input_grad = cb_tmp2 * cb_tmp1  (bcast scalar, B=tmp1 held)
        eltwise_chain(
            onetile,
            BinaryFpu<
                cb_tmp2,
                cb_tmp1,
                cb_input_grad,
                BinaryFpuOp::Mul,
                BroadcastDim::Scalar,
                BinaryDataFormatReconfig::InputAndOutput,
                CopyTilePolicy::WaitAndPop,
                CopyTilePolicy::WaitNoPop,
                CbIndexMode::FirstTile,
                Dst::D0,
                /*EnableFp32DestAcc=*/DST_ACCUM_MODE>{},
            PackTile<
                cb_input_grad,
                Dst::D0,
                PackTilePolicy::PerTileReserveAndPush,
                PackTileIndexMode::FirstTile,
                PackTileReconfig::Output,
                /*EnableFp32DestAcc=*/DST_ACCUM_MODE>{});
#else
        // T1.27 (no-DIVISOR branch): cb_input_grad = -(cb_tmp_weight * cb_output_grad)
        eltwise_chain(
            onetile,
            BinaryFpu<
                cb_tmp_weight,
                cb_output_grad,
                cb_input_grad,
                BinaryFpuOp::Mul,
                BroadcastDim::Scalar,
                BinaryDataFormatReconfig::InputAndOutput,
                CopyTilePolicy::WaitAndPop,
                CopyTilePolicy::NoWaitNoPop,
                CbIndexMode::FirstTile,
                Dst::D0,
                /*EnableFp32DestAcc=*/DST_ACCUM_MODE>{},
            Negative<Dst::D0>{},
            PackTile<
                cb_input_grad,
                Dst::D0,
                PackTilePolicy::PerTileReserveAndPush,
                PackTileIndexMode::FirstTile,
                PackTileReconfig::Output,
                /*EnableFp32DestAcc=*/DST_ACCUM_MODE>{});
#endif
    }

#if defined(DIVISOR)
    cb_pop_front(cb_divisor, onetile);
#endif
}
