// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"

// Reuse the SwiGLU SFPU kernel introduced for moe_gpt — it ships its own Config
// struct (alpha=1.702, clamp_limit=7.0 by default) matching GPT-OSS exactly.
#ifdef TRISC_PACK
#include "ttnn/cpp/ttnn/operations/experimental/ccl/moe_gpt/device/kernels/swiglu_sfpu.h"
#endif

void kernel_main() {
    constexpr uint32_t cb_gate = get_compile_time_arg_val(0);
    constexpr uint32_t cb_up = get_compile_time_arg_val(1);
    constexpr uint32_t cb_out = get_compile_time_arg_val(2);
    constexpr uint32_t tiles_per_core = get_compile_time_arg_val(3);

    // Process up to 4 tile-pairs per dest-register batch (SFPU writes occupy 2x
    // dest tiles per pair: one for gate, one for up; in-place result lands back
    // in the gate slot).
    constexpr uint32_t tiles_per_iter = 4;

    init_sfpu(cb_gate, cb_out);
    PACK((ckernel::llk_math_eltwise_binary_sfpu_swiglu_init()));

    cb_wait_front(cb_gate, tiles_per_core);
    cb_wait_front(cb_up, tiles_per_core);
    cb_reserve_back(cb_out, tiles_per_core);

    for (uint32_t base = 0; base < tiles_per_core; base += tiles_per_iter) {
        const uint32_t n = (tiles_per_core - base) < tiles_per_iter ? (tiles_per_core - base) : tiles_per_iter;

        tile_regs_acquire();

        // Load gate tiles into even dest indices, up tiles into odd dest indices.
        copy_tile_to_dst_init_short(cb_gate);
        for (uint32_t i = 0; i < n; ++i) {
            copy_tile(cb_gate, base + i, /*dst_idx=*/i * 2);
        }
        copy_tile_to_dst_init_short(cb_up);
        for (uint32_t i = 0; i < n; ++i) {
            copy_tile(cb_up, base + i, /*dst_idx=*/i * 2 + 1);
        }

        // Run SwiGLU SFPU on each tile pair: result lands back in the gate dest slot.
        for (uint32_t i = 0; i < n; ++i) {
            PACK((ckernel::llk_math_eltwise_binary_sfpu_swiglu<false>(i * 2, i * 2 + 1, i * 2)));
        }

        tile_regs_commit();

        tile_regs_wait();
        for (uint32_t i = 0; i < n; ++i) {
            pack_tile(/*dst_idx=*/i * 2, cb_out);
        }
        tile_regs_release();
    }

    cb_push_back(cb_out, tiles_per_core);
    cb_pop_front(cb_gate, tiles_per_core);
    cb_pop_front(cb_up, tiles_per_core);
}
