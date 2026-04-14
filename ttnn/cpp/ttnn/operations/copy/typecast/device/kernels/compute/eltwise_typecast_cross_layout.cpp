// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Cross-layout typecast compute kernel.
//
// TILIZE_INPUT    (RM→TILE): tilize first, then typecast.
//   Phase 1: tilize(input_cb → intermediate_cb) — clean hardware, no SFPU state
//   Phase 2: typecast(intermediate_cb → output_cb) — after explicit reconfig
//
// UNTILIZE_OUTPUT (TILE→RM): fused typecast + pack_untilize_dest (single phase)

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
#endif

#if defined(TILIZE_INPUT)
    // RM→TILE: tilize first (clean hardware), then typecast.
    //
    // Phase 1: tilize RM data from input_cb → intermediate_cb (tile format, input dtype).
    // Tilize runs first on clean hardware — no SFPU state to interfere.
    //
    // Phase 2: typecast from intermediate_cb → output_cb (tile format, output dtype).
    // Explicit reconfig between phases ensures hardware is properly configured.

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
        fp32_mode>(per_core_block_cnt);

    // Phase 2: typecast from intermediate_cb → output_cb
    // init_sfpu fully reinitializes unpack/math/pack for the SFPU typecast path.
    // The tilize_uninit (called by tilize helper above) already cleaned up tilize state.
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
    // TILE→RM: Fused typecast + pack_untilize_dest.
    // Typecast tiles into DEST, then pack_untilize_dest writes DEST in RM format.

    init_sfpu(input_cb, output_cb);

    // Disable fp32 dest accumulation before pack_untilize_dest_init —
    // pack_untilize uses DST_ACCESS_STRIDED_MODE where strides are derived from
    // pack_src_format. With fp32_dest on and non-fp32 output, strides are wrong.
#if FP32_DEST_ACC_EN
    disable_fp32_dest_acc();
#endif

    pack_untilize_dest_init<per_core_block_dim>(output_cb);

    for (uint32_t block_index = 0; block_index < per_core_block_cnt; block_index++) {
        tile_regs_acquire();

        for (uint32_t tile_index = 0; tile_index < per_core_block_dim; ++tile_index) {
            cb_wait_front(input_cb, 1);
            copy_tile(input_cb, 0, tile_index);
            TYPECAST_LLK_INIT();
            TYPECAST_LLK(tile_index);
            cb_pop_front(input_cb, 1);
        }

        tile_regs_commit();
        tile_regs_wait();

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
