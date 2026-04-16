// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Unified typecast compute kernel.
//
// No defines:       same-layout typecast (TILE→TILE or RM→RM).
// TILIZE_INPUT:     RM→TILE — tilize first, then typecast.
// UNTILIZE_OUTPUT:  TILE→RM — fused typecast + pack_untilize_dest.

#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/typecast.h"
#include "experimental/circular_buffer.h"

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

    experimental::CircularBuffer cb_in(input_cb);
    experimental::CircularBuffer cb_out(output_cb);

#if defined(TILIZE_INPUT)
    constexpr uint32_t intermediate_cb = get_compile_time_arg_val(4);
    experimental::CircularBuffer cb_inter(intermediate_cb);

    // Phase 1: tilize RM data from input_cb → intermediate_cb on clean hardware.
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
        fp32_mode>(per_core_block_cnt);

    // Phase 2: typecast from intermediate_cb → output_cb.
    // init_sfpu must come after tilize — tilize needs clean hardware without SFPU state.
    init_sfpu(intermediate_cb, output_cb);
#if FP32_DEST_ACC_EN && !defined(TYPECAST_OUTPUT_32BIT)
    disable_fp32_dest_acc();
#endif
    // Sync: init_sfpu has MATH and PACK both writing to ALU_FORMAT_SPEC cfg register
    // (MATH sets SrcA/SrcB, PACK sets Dstacc) via RMWCIB on the same 32-bit word.
    // Stall CFG until both MATH and PACK config writes are committed, so pack_tile
    // reads the correct Dstacc value.
    constexpr auto stall_until_config_done = []() { TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::MATH | p_stall::PACK); };
    PACK((stall_until_config_done()));

    for (uint32_t block_index = 0; block_index < per_core_block_cnt; block_index++) {
        cb_out.reserve_back(per_core_block_dim);
        for (uint32_t tile_index = 0; tile_index < per_core_block_dim; ++tile_index) {
            tile_regs_acquire();
            cb_inter.wait_front(1);
            copy_tile(intermediate_cb, 0, 0);
            TYPECAST_LLK_INIT();
            TYPECAST_LLK(0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, output_cb);
            cb_inter.pop_front(1);
            tile_regs_release();
        }
        cb_out.push_back(per_core_block_dim);
    }

#elif defined(UNTILIZE_OUTPUT)
    // Fused typecast + pack_untilize_dest.
    init_sfpu(input_cb, output_cb);

#if FP32_DEST_ACC_EN
    disable_fp32_dest_acc();
    constexpr auto stall_until_config_done = []() { TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::MATH | p_stall::PACK); };
    PACK((stall_until_config_done()));
#endif

    pack_untilize_dest_init<per_core_block_dim>(output_cb);

    for (uint32_t block_index = 0; block_index < per_core_block_cnt; block_index++) {
        tile_regs_acquire();

        for (uint32_t tile_index = 0; tile_index < per_core_block_dim; ++tile_index) {
            cb_in.wait_front(1);
            copy_tile(input_cb, 0, tile_index);
            TYPECAST_LLK_INIT();
            TYPECAST_LLK(tile_index);
            cb_in.pop_front(1);
        }

        tile_regs_commit();
        tile_regs_wait();

        cb_out.reserve_back(per_core_block_dim);
        pack_untilize_dest<per_core_block_dim>(output_cb);
        cb_out.push_back(per_core_block_dim);

        tile_regs_release();
    }

    pack_untilize_uninit(output_cb);

#else
    // Same-layout typecast (TILE→TILE or RM→RM).
    init_sfpu(input_cb, output_cb);
    for (uint32_t block_index = 0; block_index < per_core_block_cnt; block_index++) {
        cb_out.reserve_back(per_core_block_dim);
        for (uint32_t tile_index = 0; tile_index < per_core_block_dim; ++tile_index) {
            tile_regs_acquire();
            cb_in.wait_front(1);
            copy_tile(input_cb, 0, 0);
            TYPECAST_LLK_INIT();
            TYPECAST_LLK(0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, output_cb);
            cb_in.pop_front(1);
            tile_regs_release();
        }
        cb_out.push_back(per_core_block_dim);
    }
#endif
}
