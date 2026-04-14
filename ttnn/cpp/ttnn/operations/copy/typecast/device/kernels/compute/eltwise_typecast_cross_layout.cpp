// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Cross-layout typecast compute kernel.
//
// Handles two modes via compile-time defines:
//
//   TILIZE_INPUT    (RM→TILE): tilize input_cb → intermediate_cb, then typecast intermediate_cb → output_cb
//   UNTILIZE_OUTPUT (TILE→RM): fused typecast + pack_untilize_dest — typecast tiles into DEST,
//                               then pack_untilize_dest writes DEST in RM format to output_cb

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
#endif

void kernel_main() {
    constexpr uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
    constexpr uint32_t per_core_block_dim = get_compile_time_arg_val(1);
    constexpr uint32_t input_cb = get_compile_time_arg_val(2);
    constexpr uint32_t output_cb = get_compile_time_arg_val(3);
#ifdef TILIZE_INPUT
    constexpr uint32_t intermediate_cb = get_compile_time_arg_val(4);
    constexpr uint32_t total_input_pages = get_compile_time_arg_val(5);
#endif

#if defined(TILIZE_INPUT)
    // RM→TILE: tilize all blocks into intermediate_cb, then typecast to output_cb.
    // Tilize pushes tiles to intermediate_cb per block; typecast consumes them.
    // CB double-buffering handles the producer-consumer flow.

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

    // Phase 2: typecast tiles from intermediate_cb → output_cb
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
    // TILE→RM: Fused typecast + untilize using pack_untilize_dest.
    //
    // For each tile-row (block):
    //   1. Typecast per_core_block_dim tiles into DEST registers 0..N-1
    //   2. pack_untilize_dest writes all DEST data to output_cb in RM format
    //
    // This avoids a two-phase approach that causes hardware state conflicts.
    // Constraint: per_core_block_dim must fit in DEST (max 8 tiles in half-sync 32-bit).

    init_sfpu(input_cb, output_cb);
    pack_untilize_dest_init<per_core_block_dim>(output_cb);

    for (uint32_t block_index = 0; block_index < per_core_block_cnt; block_index++) {
        tile_regs_acquire();

        // Typecast each tile in the row into a separate DEST register
        for (uint32_t tile_index = 0; tile_index < per_core_block_dim; ++tile_index) {
            cb_wait_front(input_cb, 1);
            copy_tile(input_cb, 0, tile_index);
            TYPECAST_LLK_INIT();
            TYPECAST_LLK(tile_index);
            cb_pop_front(input_cb, 1);
        }

        tile_regs_commit();
        tile_regs_wait();

        // Write all DEST tiles to output CB in RM format
        cb_reserve_back(output_cb, per_core_block_dim);
        pack_untilize_dest<per_core_block_dim>(output_cb);
        cb_push_back(output_cb, per_core_block_dim);

        tile_regs_release();
    }

    pack_untilize_uninit(output_cb);

#else
    static_assert(false, "eltwise_typecast_cross_layout.cpp requires TILIZE_INPUT or UNTILIZE_OUTPUT define");
#endif
}
