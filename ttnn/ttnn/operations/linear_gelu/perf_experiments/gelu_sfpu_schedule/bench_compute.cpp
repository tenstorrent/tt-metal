// SPDX-FileCopyrightText: 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Isolated packer-thread GELU bench: math copies a chunk of fp32 tiles into DEST, the PACK thread runs the
// op's exact activation path (apply_activation_from_pack<GELU_TANH, 1>) `reps` times on it, then packs.
// GST_VARIANT (define) picks the SFPU function (gst_gelu.h); DO_GELU 0 = copy + pack floor.
#include "gst_gelu.h"

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/cb_api.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/sfpu_activation_helpers.hpp"

void kernel_main() {
    using namespace compute_kernel_lib;
    constexpr uint32_t cb_in = 0;
    constexpr uint32_t cb_out = 16;
    constexpr uint32_t num_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t chunk = get_compile_time_arg_val(1);
    constexpr uint32_t do_gelu = get_compile_time_arg_val(2);
    const uint32_t reps = get_arg_val<uint32_t>(0);

    compute_kernel_hw_startup(cb_in, cb_out);
    copy_tile_init(cb_in);
    if constexpr (do_gelu) {
        ActivationInitHelper<KernelActivation::GELU_TANH, 1>::init();
    }

    cb_reserve_back(cb_in, num_tiles);
    cb_push_back(cb_in, num_tiles);
    cb_wait_front(cb_in, num_tiles);

    for (uint32_t base = 0; base < num_tiles; base += chunk) {
        tile_regs_acquire();
        for (uint32_t i = 0; i < chunk; ++i) {
            copy_tile(cb_in, base + i, i);
        }
        tile_regs_commit();
        if constexpr (do_gelu) {
            for (uint32_t r = 0; r < reps; ++r) {
                apply_activation_from_pack<KernelActivation::GELU_TANH, 1>(chunk);
            }
        } else {
            tile_regs_wait();
        }
        cb_reserve_back(cb_out, chunk);
        for (uint32_t i = 0; i < chunk; ++i) {
            pack_tile(i, cb_out);
        }
        cb_push_back(cb_out, chunk);
        tile_regs_release();
    }
    cb_pop_front(cb_in, num_tiles);
}
