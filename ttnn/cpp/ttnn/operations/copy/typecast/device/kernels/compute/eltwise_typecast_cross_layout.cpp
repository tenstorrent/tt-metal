// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Cross-layout typecast compute kernel.
//
// TILIZE_INPUT    (RM→TILE): tilize first, then typecast.
// UNTILIZE_OUTPUT (TILE→RM): fused typecast + pack_untilize_dest.

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
    // When fp32_dest is enabled (for 32-bit type conversions), we must decide whether
    // to keep or disable it for Phase 2 packing:
    //   - 32-bit output (Int32, UInt32, Float32): keep fp32_dest — DEST needs 32 bits
    //     for correct representation, and pack_src_format matches 32-bit DEST mode.
    //   - 16-bit output (bf16, fp16, uint16, etc.): disable fp32_dest — pack_tile needs
    //     pack_src_format to match the 16-bit DEST mode.
    init_sfpu(intermediate_cb, output_cb);
#if FP32_DEST_ACC_EN && !defined(TYPECAST_OUTPUT_32BIT)
    disable_fp32_dest_acc();
#endif
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
    static_assert(false, "eltwise_typecast_cross_layout.cpp requires TILIZE_INPUT or UNTILIZE_OUTPUT define");
#endif
}
