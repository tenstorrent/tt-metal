// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/addcmul.h"
#include "api/compute/eltwise_unary/addcdiv.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_helpers.hpp"

void kernel_main() {
    const uint32_t num_tiles = get_arg_val<uint32_t>(0);
    const uint32_t scalar_arg = get_arg_val<uint32_t>(3);
    constexpr uint32_t num_tiles_per_cycle = get_compile_time_arg_val(0);

    constexpr auto cb_in0 = tt::CBIndex::c_0;
    constexpr auto cb_in1 = tt::CBIndex::c_1;
    constexpr auto cb_in2 = tt::CBIndex::c_2;
    constexpr auto cb_out = tt::CBIndex::c_3;

    using namespace compute_kernel_lib;

    unary_op_init_common(cb_in0, cb_out);

    // output = a + scalar * b * c  (addcmul) or a + scalar * b / c  (addcdiv)
    // Explicit #if/elif chains replace TERNARY_SFPU_OP_INIT/FUNC macros.
#if defined(TERNARY_OP_ADDCMUL_FP32)
    {
        auto chain = sfpu_chain(
            Load<cb_in0, Dst::D0>{},
            Load<cb_in1, Dst::D1>{},
            Load<cb_in2, Dst::D2>{},
            Addcmul<DataFormat::Float32, Dst::D0, Dst::D1, Dst::D2, Dst::D0>{scalar_arg});
        eltwise_op<cb_out>(chain, EltwiseTileShape::flat(num_tiles));
    }
#elif defined(TERNARY_OP_ADDCMUL_FP16B)
    {
        auto chain = sfpu_chain(
            Load<cb_in0, Dst::D0>{},
            Load<cb_in1, Dst::D1>{},
            Load<cb_in2, Dst::D2>{},
            Addcmul<DataFormat::Float16_b, Dst::D0, Dst::D1, Dst::D2, Dst::D0>{scalar_arg});
        eltwise_op<cb_out>(chain, EltwiseTileShape::flat(num_tiles));
    }
#elif defined(TERNARY_OP_ADDCDIV_FP32)
    {
        auto chain = sfpu_chain(
            Load<cb_in0, Dst::D0>{},
            Load<cb_in1, Dst::D1>{},
            Load<cb_in2, Dst::D2>{},
            Addcdiv<DataFormat::Float32, Dst::D0, Dst::D1, Dst::D2, Dst::D0>{scalar_arg});
        eltwise_op<cb_out>(chain, EltwiseTileShape::flat(num_tiles));
    }
#elif defined(TERNARY_OP_ADDCDIV_FP16B)
    {
        auto chain = sfpu_chain(
            Load<cb_in0, Dst::D0>{},
            Load<cb_in1, Dst::D1>{},
            Load<cb_in2, Dst::D2>{},
            Addcdiv<DataFormat::Float16_b, Dst::D0, Dst::D1, Dst::D2, Dst::D0>{scalar_arg});
        eltwise_op<cb_out>(chain, EltwiseTileShape::flat(num_tiles));
    }
#else
    // Legacy macro path
    for (uint32_t tile_id = 0; tile_id < num_tiles; ++tile_id) {
        cb_wait_front(cb_in0, num_tiles_per_cycle);
        cb_wait_front(cb_in1, num_tiles_per_cycle);
        cb_wait_front(cb_in2, num_tiles_per_cycle);
        cb_reserve_back(cb_out, num_tiles_per_cycle);
        tile_regs_acquire();
        copy_tile_init(cb_in0);
        copy_tile(cb_in0, 0, 0);
        copy_tile_init(cb_in1);
        copy_tile(cb_in1, 0, 1);
        copy_tile_init(cb_in2);
        copy_tile(cb_in2, 0, 2);
        TERNARY_SFPU_OP_INIT();
        TERNARY_SFPU_OP_FUNC(0, 1, 2, 0, scalar_arg);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();
        cb_push_back(cb_out, num_tiles_per_cycle);
        cb_pop_front(cb_in0, num_tiles_per_cycle);
        cb_pop_front(cb_in1, num_tiles_per_cycle);
        cb_pop_front(cb_in2, num_tiles_per_cycle);
    }
#endif
}
