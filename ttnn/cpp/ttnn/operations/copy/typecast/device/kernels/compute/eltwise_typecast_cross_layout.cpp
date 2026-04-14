// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Cross-layout typecast compute kernel.
//
// Handles two modes via compile-time defines:
//
//   TILIZE_INPUT  (RM→TILE): tilize input_cb → intermediate_cb, then typecast intermediate_cb → output_cb
//   UNTILIZE_OUTPUT (TILE→RM): typecast input_cb → intermediate_cb, then untilize intermediate_cb → output_cb
//
// Uses two-pass approach with an intermediate CB to leverage well-tested LLK tilize/untilize helpers.

#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/typecast.h"

#ifdef TILIZE_INPUT
#include "api/compute/tilize.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#endif

#ifdef UNTILIZE_OUTPUT
#include "api/compute/pack_untilize.h"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#endif

void kernel_main() {
    constexpr uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
    constexpr uint32_t per_core_block_dim = get_compile_time_arg_val(1);
    constexpr uint32_t input_cb = get_compile_time_arg_val(2);
    constexpr uint32_t output_cb = get_compile_time_arg_val(3);
    constexpr uint32_t intermediate_cb = get_compile_time_arg_val(4);
#ifdef TILIZE_INPUT
    constexpr uint32_t total_input_pages = get_compile_time_arg_val(5);
#endif

#if defined(TILIZE_INPUT)
    // RM→TILE: Phase 1 — tilize input_cb → intermediate_cb
    //          Phase 2 — typecast intermediate_cb → output_cb

    // Phase 1: tilize
    compute_kernel_hw_startup(input_cb, intermediate_cb);

    constexpr auto fp32_mode = compute_kernel_lib::is_fp32_input_format<input_cb>()
                                   ? compute_kernel_lib::tilize_config::Fp32Mode::Lossless
                                   : compute_kernel_lib::tilize_config::Fp32Mode::Fast;

    compute_kernel_lib::tilize<
        per_core_block_dim,
        input_cb,
        intermediate_cb,
        compute_kernel_lib::tilize_config::InitUninitMode::InitAndUninit,
        compute_kernel_lib::tilize_config::WaitMode::WaitBlock,
        compute_kernel_lib::tilize_config::ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure,
        fp32_mode>(per_core_block_cnt, total_input_pages);

    // Phase 2: typecast from intermediate_cb → output_cb
    init_sfpu(intermediate_cb, output_cb);
    for (uint32_t block_index = 0; block_index < per_core_block_cnt; block_index++) {
        cb_reserve_back(output_cb, per_core_block_dim);
        for (uint32_t tile_index = 0; tile_index < per_core_block_dim; ++tile_index) {
            tile_regs_acquire();
            cb_wait_front(intermediate_cb, 1);
            copy_tile(intermediate_cb, 0, 0);
            TYPECAST_LLK_INIT();
            TYPECAST_LLK(0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, output_cb);
            cb_pop_front(intermediate_cb, 1);
            tile_regs_release();
        }
        cb_push_back(output_cb, per_core_block_dim);
    }

#elif defined(UNTILIZE_OUTPUT)
    // TILE→RM: Phase 1 — typecast input_cb → intermediate_cb
    //          Phase 2 — untilize intermediate_cb → output_cb

    // Phase 1: typecast
    init_sfpu(input_cb, intermediate_cb);
    for (uint32_t block_index = 0; block_index < per_core_block_cnt; block_index++) {
        cb_reserve_back(intermediate_cb, per_core_block_dim);
        for (uint32_t tile_index = 0; tile_index < per_core_block_dim; ++tile_index) {
            tile_regs_acquire();
            cb_wait_front(input_cb, 1);
            copy_tile(input_cb, 0, 0);
            TYPECAST_LLK_INIT();
            TYPECAST_LLK(0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, intermediate_cb);
            cb_pop_front(input_cb, 1);
            tile_regs_release();
        }
        cb_push_back(intermediate_cb, per_core_block_dim);
    }

    // Phase 2: untilize from intermediate_cb → output_cb
    // Reconfigure hardware for the untilize path
    compute_kernel_lib::untilize<
        per_core_block_dim,
        intermediate_cb,
        output_cb,
        compute_kernel_lib::untilize_config::InitUninitMode::InitAndUninit,
        compute_kernel_lib::untilize_config::WaitMode::WaitBlock,
        compute_kernel_lib::untilize_config::ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure>(
        per_core_block_cnt);

#else
    static_assert(false, "eltwise_typecast_cross_layout.cpp requires TILIZE_INPUT or UNTILIZE_OUTPUT define");
#endif
}
