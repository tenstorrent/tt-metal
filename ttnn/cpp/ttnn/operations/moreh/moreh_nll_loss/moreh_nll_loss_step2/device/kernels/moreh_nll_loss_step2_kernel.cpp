// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_math.hpp"  // Recip
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_misc.hpp"  // Negative
#include "ttnn/kernel/compute/moreh_common.hpp"

void kernel_main() {
    constexpr uint32_t per_core_tile_cnt = get_compile_time_arg_val(0);

    constexpr uint32_t cb_weight = tt::CBIndex::c_2;
    constexpr uint32_t cb_divisor = tt::CBIndex::c_3;

    constexpr uint32_t cb_tmp_weight = tt::CBIndex::c_24;
    constexpr uint32_t cb_tmp_input = tt::CBIndex::c_25;
    constexpr uint32_t cb_tmp1 = tt::CBIndex::c_26;
    constexpr uint32_t cb_divisor_recip = tt::CBIndex::c_27;  // 1/divisor
    constexpr uint32_t cb_tmp3 = tt::CBIndex::c_28;

    constexpr uint32_t cb_output = tt::CBIndex::c_16;

    constexpr uint32_t onetile = 1;

    binary_op_init_common(cb_tmp_weight, cb_tmp_input, cb_output);

    using namespace compute_kernel_lib;

#if defined(DIVISOR)
    // cb_divisor_recip = 1 / cb_divisor (already migrated, kept).
    eltwise_chain(
        onetile,
        CopyTile<cb_divisor, Dst::D0, CopyTilePolicy::WaitAndPop, CbIndexMode::FirstTile, CopyTileReconfig::Input>{},
        Recip<Dst::D0>{},
        PackTile<
            cb_divisor_recip,
            Dst::D0,
            PackTilePolicy::PerTileReserveAndPush,
            PackTileIndexMode::FirstTile,
            PackTileReconfig::Output,
            /*EnableFp32DestAcc=*/DST_ACCUM_MODE>{});
#endif

    for (uint32_t b = 0; b < per_core_tile_cnt; ++b) {
#if defined(WEIGHT)
        // T1.28: -cb_tmp_input -> cb_tmp1
        eltwise_chain(
            onetile,
            CopyTile<
                cb_tmp_input,
                Dst::D0,
                CopyTilePolicy::WaitAndPop,
                CbIndexMode::FirstTile,
                CopyTileReconfig::Input>{},
            Negative<Dst::D0>{},
            PackTile<
                cb_tmp1,
                Dst::D0,
                PackTilePolicy::PerTileReserveAndPush,
                PackTileIndexMode::FirstTile,
                PackTileReconfig::Output,
                /*EnableFp32DestAcc=*/DST_ACCUM_MODE>{});

#if defined(DIVISOR)
        // T1.29: cb_tmp1 * cb_tmp_weight -> cb_tmp3
        eltwise_chain(
            onetile,
            BinaryFpu<
                cb_tmp1,
                cb_tmp_weight,
                cb_tmp3,
                BinaryFpuOp::Mul,
                BroadcastDim::None,
                BinaryDataFormatReconfig::InputAndOutput,
                CopyTilePolicy::WaitAndPop,
                CopyTilePolicy::WaitAndPop,
                CbIndexMode::FirstTile,
                Dst::D0,
                /*EnableFp32DestAcc=*/DST_ACCUM_MODE>{},
            PackTile<
                cb_tmp3,
                Dst::D0,
                PackTilePolicy::PerTileReserveAndPush,
                PackTileIndexMode::FirstTile,
                PackTileReconfig::Output,
                /*EnableFp32DestAcc=*/DST_ACCUM_MODE>{});

        // T1.30: cb_tmp3 * cb_divisor_recip (bcast scalar, B held) -> cb_output
        eltwise_chain(
            onetile,
            BinaryFpu<
                cb_tmp3,
                cb_divisor_recip,
                cb_output,
                BinaryFpuOp::Mul,
                BroadcastDim::Scalar,
                BinaryDataFormatReconfig::InputAndOutput,
                CopyTilePolicy::WaitAndPop,
                CopyTilePolicy::WaitNoPop,
                CbIndexMode::FirstTile,
                Dst::D0,
                /*EnableFp32DestAcc=*/DST_ACCUM_MODE>{},
            PackTile<
                cb_output,
                Dst::D0,
                PackTilePolicy::PerTileReserveAndPush,
                PackTileIndexMode::FirstTile,
                PackTileReconfig::Output,
                /*EnableFp32DestAcc=*/DST_ACCUM_MODE>{});
#else
        // WEIGHT && !DIVISOR: cb_tmp1 * cb_tmp_weight -> cb_output
        eltwise_chain(
            onetile,
            BinaryFpu<
                cb_tmp1,
                cb_tmp_weight,
                cb_output,
                BinaryFpuOp::Mul,
                BroadcastDim::None,
                BinaryDataFormatReconfig::InputAndOutput,
                CopyTilePolicy::WaitAndPop,
                CopyTilePolicy::WaitAndPop,
                CbIndexMode::FirstTile,
                Dst::D0,
                /*EnableFp32DestAcc=*/DST_ACCUM_MODE>{},
            PackTile<
                cb_output,
                Dst::D0,
                PackTilePolicy::PerTileReserveAndPush,
                PackTileIndexMode::FirstTile,
                PackTileReconfig::Output,
                /*EnableFp32DestAcc=*/DST_ACCUM_MODE>{});
#endif
#else
#if defined(DIVISOR)
        // T1.31: !WEIGHT && DIVISOR — pipeline:
        //   step 1: -cb_tmp_input -> cb_tmp1
        //   step 2: cb_tmp1 * cb_divisor_recip (bcast scalar, B held) -> cb_output
        eltwise_chain(
            onetile,
            CopyTile<
                cb_tmp_input,
                Dst::D0,
                CopyTilePolicy::WaitAndPop,
                CbIndexMode::FirstTile,
                CopyTileReconfig::Input>{},
            Negative<Dst::D0>{},
            PackTile<
                cb_tmp1,
                Dst::D0,
                PackTilePolicy::PerTileReserveAndPush,
                PackTileIndexMode::FirstTile,
                PackTileReconfig::Output,
                /*EnableFp32DestAcc=*/DST_ACCUM_MODE>{});

        eltwise_chain(
            onetile,
            BinaryFpu<
                cb_tmp1,
                cb_divisor_recip,
                cb_output,
                BinaryFpuOp::Mul,
                BroadcastDim::Scalar,
                BinaryDataFormatReconfig::InputAndOutput,
                CopyTilePolicy::WaitAndPop,
                CopyTilePolicy::WaitNoPop,
                CbIndexMode::FirstTile,
                Dst::D0,
                /*EnableFp32DestAcc=*/DST_ACCUM_MODE>{},
            PackTile<
                cb_output,
                Dst::D0,
                PackTilePolicy::PerTileReserveAndPush,
                PackTileIndexMode::FirstTile,
                PackTileReconfig::Output,
                /*EnableFp32DestAcc=*/DST_ACCUM_MODE>{});
#else
        // !WEIGHT && !DIVISOR: -cb_tmp_input -> cb_output
        eltwise_chain(
            onetile,
            CopyTile<
                cb_tmp_input,
                Dst::D0,
                CopyTilePolicy::WaitAndPop,
                CbIndexMode::FirstTile,
                CopyTileReconfig::Input>{},
            Negative<Dst::D0>{},
            PackTile<
                cb_output,
                Dst::D0,
                PackTilePolicy::PerTileReserveAndPush,
                PackTileIndexMode::FirstTile,
                PackTileReconfig::Output,
                /*EnableFp32DestAcc=*/DST_ACCUM_MODE>{});
#endif
#endif
    }

#if defined(DIVISOR)
    cb_pop_front(cb_divisor_recip, onetile);
#endif
}
