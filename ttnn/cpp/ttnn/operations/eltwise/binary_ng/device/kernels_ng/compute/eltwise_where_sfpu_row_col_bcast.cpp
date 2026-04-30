// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// PARTIAL MIGRATION: unary_bcast pre-process stays raw LLK. Where SFPU stage
// migrated via WhereMacroOp + FillMacroOp.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/where.h"
#include "api/compute/eltwise_unary/fill.h"
#include "ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_utils_common.hpp"
#include "ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_utils_sfpu.hpp"
#include "api/compute/bcast.h"

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

struct WhereMacroOp : compute_kernel_lib::TernaryOp<
                          WhereMacroOp,
                          compute_kernel_lib::Dst::D0,
                          compute_kernel_lib::Dst::D1,
                          compute_kernel_lib::Dst::D2,
                          compute_kernel_lib::Dst::D0> {
    static constexpr bool clobbers_sfpu_lut = true;
    ALWI static void init() {}
    ALWI static void call(uint32_t i0, uint32_t i1, uint32_t i2, uint32_t out_idx) {
        BINARY_SFPU_OP(i0, i1, i2, out_idx);
    }
};
}  // namespace

template <tt::CBIndex cb_in0, tt::CBIndex cb_in1, tt::CBIndex cb_out>
ALWI void process_tile(uint32_t freq, uint32_t tile_start, uint32_t num_tiles_per_cycle, uint32_t scalar_value) {
    using namespace ckernel;
    using compute_kernel_lib::CopyTile;
    using compute_kernel_lib::Dst;
    using compute_kernel_lib::eltwise_chain;
    using compute_kernel_lib::eltwise_pipeline;

#if BCAST_INPUT
    constexpr auto CB_BCAST = cb_in1;
    constexpr auto CB_OTHER = cb_in0;
    constexpr auto cb_llk_post = tt::CBIndex::c_5;
    constexpr auto cb_left = tt::CBIndex::c_5;
    constexpr auto cb_right = cb_in1;
#else
    constexpr auto CB_BCAST = cb_in0;
    constexpr auto CB_OTHER = cb_in1;
    constexpr auto cb_llk_post = tt::CBIndex::c_6;
    constexpr auto cb_left = cb_in0;
    constexpr auto cb_right = tt::CBIndex::c_6;
#endif

    cb_wait_front(CB_BCAST, num_tiles_per_cycle);

    for (uint32_t j = tile_start; j < freq; ++j) {
        cb_wait_front(CB_OTHER, num_tiles_per_cycle);

        // ---- Stage 1: unary_bcast pre-process (raw LLK) ----
        cb_reserve_back(cb_llk_post, num_tiles_per_cycle);
        unary_bcast_init<BroadcastType::ROW>(CB_OTHER, cb_llk_post);
        tile_regs_acquire();
        unary_bcast<BroadcastType::ROW>(CB_OTHER, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_llk_post);
        cb_push_back(cb_llk_post, num_tiles_per_cycle);
        tile_regs_release();

        cb_pop_front(CB_OTHER, num_tiles_per_cycle);
        pack_reconfig_data_format(cb_llk_post, cb_out);
#ifdef ARCH_BLACKHOLE
        PACK((llk_pack_hw_configure<DST_ACCUM_MODE>(cb_out)));
#endif

        // ---- Stage 2: where SFPU on cb_left/cb_right (V2 helper) ----
#if WHERE_TTS
        eltwise_pipeline<static_cast<uint32_t>(cb_out)>(
            num_tiles_per_cycle,
            eltwise_chain(
                CopyTile<static_cast<uint32_t>(cb_left), Dst::D0>{},
                CopyTile<static_cast<uint32_t>(cb_right), Dst::D1>{},
                FillMacroOp<Dst::D2>{{}, scalar_value},
                WhereMacroOp{}));
#endif
#if WHERE_TST
        eltwise_pipeline<static_cast<uint32_t>(cb_out)>(
            num_tiles_per_cycle,
            eltwise_chain(
                CopyTile<static_cast<uint32_t>(cb_left), Dst::D0>{},
                FillMacroOp<Dst::D1>{{}, scalar_value},
                CopyTile<static_cast<uint32_t>(cb_right), Dst::D2>{},
                WhereMacroOp{}));
#endif
    }
    cb_pop_front(CB_BCAST, num_tiles_per_cycle);
}

void kernel_main() {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);
    uint32_t tile_freq = get_arg_val<uint32_t>(1);
    uint32_t tile_start = get_arg_val<uint32_t>(2);
    const uint32_t scalar_value = get_arg_val<uint32_t>(3);

    constexpr uint32_t num_tiles_per_cycle = get_compile_time_arg_val(0);
    static_assert(num_tiles_per_cycle == 1, "where_sfpu_row_col_bcast path runs one tile per chain invocation");

    if (num_tiles == 0) {
        return;
    }

    constexpr auto cb_in0 = tt::CBIndex::c_0;
    constexpr auto cb_in1 = tt::CBIndex::c_1;
    constexpr auto cb_out = tt::CBIndex::c_2;

    ckernel::compute_kernel_hw_startup(cb_in0, cb_out);
    ckernel::init_sfpu(cb_in0, cb_out);
    BINARY_SFPU_INIT

    uint32_t complete_iterations = (num_tiles + tile_start) / tile_freq;
    uint32_t remaining_iterations = (num_tiles + tile_start) % tile_freq;

    for (uint32_t i = 0; i < complete_iterations; ++i, tile_start = 0) {
        process_tile<cb_in0, cb_in1, cb_out>(tile_freq, tile_start, num_tiles_per_cycle, scalar_value);
    }

    if (remaining_iterations > 0) {
        process_tile<cb_in0, cb_in1, cb_out>(remaining_iterations, tile_start, num_tiles_per_cycle, scalar_value);
    }
}
