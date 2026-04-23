// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_utils_common.hpp"
#include "ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_utils_sfpu.hpp"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/where.h"
#include "api/compute/eltwise_unary/lerp.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_helpers.hpp"

void kernel_main() {
    const uint32_t num_tiles = get_arg_val<uint32_t>(0);
    constexpr uint32_t num_tiles_per_cycle = get_compile_time_arg_val(0);

    constexpr auto cb_pre_in1 = tt::CBIndex::c_0;
    constexpr auto cb_pre_in2 = tt::CBIndex::c_1;
    constexpr auto cb_pre_in3 = tt::CBIndex::c_2;
    constexpr auto cb_out = tt::CBIndex::c_3;

    using namespace compute_kernel_lib;

    unary_op_init_common(cb_pre_in1, cb_out);

    // Explicit #if/elif chains replace TERNARY_SFPU_OP_INIT/FUNC macros.
    // Host sets TERNARY_OP_WHERE_FP32 etc. alongside the legacy macro defines.
#if defined(TERNARY_OP_WHERE_FP32)
    {
        auto chain = sfpu_chain(
            Load<cb_pre_in1, Dst::D0>{},
            Load<cb_pre_in2, Dst::D1>{},
            Load<cb_pre_in3, Dst::D2>{},
            Where<DataFormat::Float32, Dst::D0, Dst::D1, Dst::D2, Dst::D0>{});
        eltwise_op<cb_out>(chain, EltwiseTileShape::flat(num_tiles));
    }
#elif defined(TERNARY_OP_WHERE_FP16B)
    {
        auto chain = sfpu_chain(
            Load<cb_pre_in1, Dst::D0>{},
            Load<cb_pre_in2, Dst::D1>{},
            Load<cb_pre_in3, Dst::D2>{},
            Where<DataFormat::Float16_b, Dst::D0, Dst::D1, Dst::D2, Dst::D0>{});
        eltwise_op<cb_out>(chain, EltwiseTileShape::flat(num_tiles));
    }
#elif defined(TERNARY_OP_WHERE_INT32)
    {
        auto chain = sfpu_chain(
            Load<cb_pre_in1, Dst::D0>{},
            Load<cb_pre_in2, Dst::D1>{},
            Load<cb_pre_in3, Dst::D2>{},
            Where<DataFormat::Int32, Dst::D0, Dst::D1, Dst::D2, Dst::D0>{});
        eltwise_op<cb_out>(chain, EltwiseTileShape::flat(num_tiles));
    }
#elif defined(TERNARY_OP_LERP_FP32)
    {
        auto chain = sfpu_chain(
            Load<cb_pre_in1, Dst::D0>{},
            Load<cb_pre_in2, Dst::D1>{},
            Load<cb_pre_in3, Dst::D2>{},
            Lerp<DataFormat::Float32, Dst::D0, Dst::D1, Dst::D2, Dst::D0>{});
        eltwise_op<cb_out>(chain, EltwiseTileShape::flat(num_tiles));
    }
#elif defined(TERNARY_OP_LERP_FP16B)
    {
        auto chain = sfpu_chain(
            Load<cb_pre_in1, Dst::D0>{},
            Load<cb_pre_in2, Dst::D1>{},
            Load<cb_pre_in3, Dst::D2>{},
            Lerp<DataFormat::Float16_b, Dst::D0, Dst::D1, Dst::D2, Dst::D0>{});
        eltwise_op<cb_out>(chain, EltwiseTileShape::flat(num_tiles));
    }
#else
    // Legacy macro path
    for (uint32_t tile_id = 0; tile_id < num_tiles; ++tile_id) {
        cb_wait_front(cb_pre_in1, num_tiles_per_cycle);
        cb_wait_front(cb_pre_in2, num_tiles_per_cycle);
        cb_wait_front(cb_pre_in3, num_tiles_per_cycle);
        cb_reserve_back(cb_out, num_tiles_per_cycle);
        tile_regs_acquire();
        copy_tile_to_dst_init_short(cb_pre_in1);
        copy_tile(cb_pre_in1, 0, 0);
        copy_tile_to_dst_init_short(cb_pre_in2);
        copy_tile(cb_pre_in2, 0, 1);
        copy_tile_to_dst_init_short(cb_pre_in3);
        copy_tile(cb_pre_in3, 0, 2);
        TERNARY_SFPU_OP_INIT();
        TERNARY_SFPU_OP_FUNC(0, 1, 2, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();
        cb_push_back(cb_out, num_tiles_per_cycle);
        cb_pop_front(cb_pre_in1, num_tiles_per_cycle);
        cb_pop_front(cb_pre_in2, num_tiles_per_cycle);
        cb_pop_front(cb_pre_in3, num_tiles_per_cycle);
    }
#endif
}
