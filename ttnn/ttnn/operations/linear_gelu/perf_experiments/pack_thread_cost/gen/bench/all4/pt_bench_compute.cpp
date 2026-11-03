#define PT_NOSTALL 1
#define PT_BATCH 1
#define PT_NORECONFIG 1
#define PT_INTERLEAVE 1
// SPDX-FileCopyrightText: 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Isolated pack-thread bench: num_subblocks x (math acquire/commit with no FPU work -> pack thread GELU + pack of
// sb_tiles tiles via pt::gelu_pack_subblock, the same stage matmul_block runs in the op -> release / push). The unpack
// thread only drains the output CB. The pack thread binds, so (T(N2) - T(N1)) / (N2 - N1) = pack ns per subblock.
#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/common.h"
#include "api/compute/pack.h"
#include "api/compute/cb_api.h"
#include "pt_pack.hpp"

void kernel_main() {
    constexpr uint32_t cb_in = 0;
    constexpr uint32_t cb_out = 16;
    const uint32_t num_subblocks = get_arg_val<uint32_t>(0);
    const uint32_t sb_tiles = get_arg_val<uint32_t>(1);

    compute_kernel_hw_startup(cb_in, cb_in, cb_out);
    compute_kernel_lib::ActivationInitHelper<KernelActivation::GELU_TANH, kGeluLoadMacro>::init();

    for (uint32_t sb = 0; sb < num_subblocks; ++sb) {
        tile_regs_acquire();
        tile_regs_commit();
        cb_reserve_back(cb_out, sb_tiles);
        pt::gelu_pack_subblock(cb_out, sb_tiles);
        tile_regs_release();
        cb_push_back(cb_out, sb_tiles);
        cb_wait_front(cb_out, sb_tiles);
        cb_pop_front(cb_out, sb_tiles);
    }
}
