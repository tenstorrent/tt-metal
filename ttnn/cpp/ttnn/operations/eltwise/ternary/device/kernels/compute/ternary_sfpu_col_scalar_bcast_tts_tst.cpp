// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/where.h"
#include "api/compute/eltwise_unary/lerp.h"
#include "api/compute/eltwise_unary/fill.h"
#include "ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_utils_common.hpp"
#include "ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_utils_sfpu.hpp"

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"

namespace {
template <compute_kernel_lib::Dst Slot>
struct FillMacroOp : compute_kernel_lib::UnaryOp<FillMacroOp<Slot>, Slot> {
    static constexpr bool clobbers_sfpu_lut = false;
    uint32_t value;

    ALWI static void init() { ckernel::fill_tile_init(); }
    ALWI void call(uint32_t dst) const {
#ifdef FILL_WITH_VALUE_FLOAT
        const auto fval = reinterpret_cast<const float*>(&value);
        FILL_LLK(dst, *fval);
#endif
#ifdef FILL_WITH_VALUE_INT
        FILL_LLK(dst, value);
#endif
    }
};

struct TernaryMacroOp : compute_kernel_lib::TernaryOp<
                            TernaryMacroOp,
                            compute_kernel_lib::Dst::D0,
                            compute_kernel_lib::Dst::D1,
                            compute_kernel_lib::Dst::D2,
                            compute_kernel_lib::Dst::D0> {
    static constexpr bool clobbers_sfpu_lut = true;

    ALWI static void init() { TERNARY_SFPU_OP_INIT(); }
    ALWI static void call(uint32_t i0, uint32_t i1, uint32_t i2, uint32_t out_idx) {
        TERNARY_SFPU_OP_FUNC(i0, i1, i2, out_idx);
    }
};
}  // namespace

template <tt::CBIndex predicate_cb, tt::CBIndex tensor_cb, tt::CBIndex cb_out, bool scalar_is_true>
ALWI void process_tile(uint32_t freq, uint32_t tile_start, uint32_t num_tiles_per_cycle, uint32_t scalar) {
    using namespace ckernel;
    using compute_kernel_lib::CopyTile;
    using compute_kernel_lib::CopyTilePolicy;
    using compute_kernel_lib::Dst;
    using compute_kernel_lib::eltwise_chain;
    using compute_kernel_lib::eltwise_pipeline;

#if BCAST_A
    cb_wait_front(predicate_cb, num_tiles_per_cycle);
#endif
#if BCAST_B && !BCAST_C
    cb_wait_front(tensor_cb, num_tiles_per_cycle);
#endif
#if BCAST_C && !BCAST_B
    cb_wait_front(tensor_cb, num_tiles_per_cycle);
#endif

    for (uint32_t j = tile_start; j < freq; ++j) {
#if !BCAST_A
        cb_wait_front(predicate_cb, num_tiles_per_cycle);
#endif
#if !(BCAST_B && !BCAST_C) && !(BCAST_C && !BCAST_B)
        cb_wait_front(tensor_cb, num_tiles_per_cycle);
#endif

        if constexpr (!scalar_is_true) {
            // TTS: tensor at D1 (true), scalar fill at D2 (false)
            eltwise_pipeline<static_cast<uint32_t>(cb_out)>(
                num_tiles_per_cycle,
                eltwise_chain(
                    CopyTile<static_cast<uint32_t>(predicate_cb), Dst::D0, CopyTilePolicy::NoWaitNoPop>{},
                    CopyTile<static_cast<uint32_t>(tensor_cb), Dst::D1, CopyTilePolicy::NoWaitNoPop>{},
                    FillMacroOp<Dst::D2>{{}, scalar},
                    TernaryMacroOp{}));
        } else {
            // TST: scalar fill at D1 (true), tensor at D2 (false)
            eltwise_pipeline<static_cast<uint32_t>(cb_out)>(
                num_tiles_per_cycle,
                eltwise_chain(
                    CopyTile<static_cast<uint32_t>(predicate_cb), Dst::D0, CopyTilePolicy::NoWaitNoPop>{},
                    FillMacroOp<Dst::D1>{{}, scalar},
                    CopyTile<static_cast<uint32_t>(tensor_cb), Dst::D2, CopyTilePolicy::NoWaitNoPop>{},
                    TernaryMacroOp{}));
        }

#if !BCAST_A
        cb_pop_front(predicate_cb, num_tiles_per_cycle);
#endif
#if !(BCAST_B && !BCAST_C) && !(BCAST_C && !BCAST_B)
        cb_pop_front(tensor_cb, num_tiles_per_cycle);
#endif
    }

#if BCAST_A
    cb_pop_front(predicate_cb, num_tiles_per_cycle);
#endif
#if BCAST_B && !BCAST_C
    cb_pop_front(tensor_cb, num_tiles_per_cycle);
#endif
#if BCAST_C && !BCAST_B
    cb_pop_front(tensor_cb, num_tiles_per_cycle);
#endif
}

void kernel_main() {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);
    uint32_t tile_freq = get_arg_val<uint32_t>(1);
    uint32_t tile_start = get_arg_val<uint32_t>(2);
    const uint32_t scalar_value = get_arg_val<uint32_t>(3);

    constexpr uint32_t num_tiles_per_cycle = get_compile_time_arg_val(0);
    static_assert(num_tiles_per_cycle == 1, "TTS/TST col_scalar_bcast path runs one tile per chain invocation");
    constexpr bool scalar_is_true = get_compile_time_arg_val(1);

    if (num_tiles == 0) {
        return;
    }

    constexpr auto predicate_cb = tt::CBIndex::c_0;
    constexpr auto tensor_cb = tt::CBIndex::c_1;
    constexpr auto cb_out = tt::CBIndex::c_3;

    ckernel::compute_kernel_hw_startup(predicate_cb, cb_out);
    ckernel::init_sfpu(predicate_cb, cb_out);

    uint32_t complete_iterations = (num_tiles + tile_start) / tile_freq;
    uint32_t remaining_iterations = (num_tiles + tile_start) % tile_freq;

    for (uint32_t i = 0; i < complete_iterations; ++i, tile_start = 0) {
        process_tile<predicate_cb, tensor_cb, cb_out, scalar_is_true>(
            tile_freq, tile_start, num_tiles_per_cycle, scalar_value);
    }

    if (remaining_iterations > 0) {
        process_tile<predicate_cb, tensor_cb, cb_out, scalar_is_true>(
            remaining_iterations, tile_start, num_tiles_per_cycle, scalar_value);
    }
}
