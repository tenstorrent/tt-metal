// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Per-tile SFPU compute kernel for binary_ng's multi-thread path. The readers deliver one in0 and one in1
// entry per output tile over strided DFBs; thread t of N computes the tiles t, t + N, ..., which the
// strided DFBs route to it in order, and its outputs reach the writers in tile order the same way.
// Compute waits and pops one tile at a time (a pop moves to the next tile counter when a thread has
// several) and builds its DataflowBuffer objects once: a short-lived one would drain the input DFBs on
// destruction while the readers run ahead. Broadcast and scalar operands arrive already expanded. Tiles
// past num_tiles are reader padding: they are unpacked, so the pop follows a real unpack, and dropped.

#include <cstdint>

#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"

#include "api/compute/eltwise_binary_sfpu.h"
// SFPU op headers that ARE ported to Quasar (internally ARCH_QUASAR-aware). bf16 multiply/divide use
// eltwise_binary_sfpu.h; the rest cover the other Quasar-supported SFPU binary ops.
#include "api/compute/add_int_sfpu.h"
#include "api/compute/mul_int_sfpu.h"
#include "api/compute/binary_max_min.h"
#include "api/compute/binary_comp.h"
// SFPU op headers NOT yet ported to Quasar: they unconditionally pull in WH/BH-only ckernel impls or
// symbols (InstrModLoadStore, DataFormat::UInt32, ckernel_sfpu_div_int32_floor.h, ...) absent from the
// Quasar ckernel tree, so they only compile off-Quasar. The float SFPU binary ops this kernel runs on
// Quasar never use them; exclude them there (mirrors the ARCH_QUASAR guard in eltwise_binary_sfpu.h).
#ifndef ARCH_QUASAR
#include "api/compute/binary_bitwise_sfpu.h"
#include "api/compute/binary_shift.h"
#include "api/compute/sub_int_sfpu.h"
#include "api/compute/div_int32_floor.h"
#include "api/compute/div_int32_sfpu.h"
#include "api/compute/binary_remainder.h"
#include "api/compute/binary_fmod.h"
#include "api/compute/quantization.h"
#include "api/compute/gcd.h"
#include "api/compute/lcm.h"
#include "api/compute/xlogy.h"
#include "api/compute/atan2.h"
#include "api/compute/isclose.h"
#endif

#include "api/kernel_thread_globals.h"
#include "experimental/kernel_args.h"
#include "eltwise_utils_common.hpp"
#include "eltwise_utils_sfpu_dfb.hpp"

void kernel_main() {
    const uint32_t num_tiles = get_arg(args::num_tiles);
    const uint32_t num_padded_tiles = get_arg(args::num_padded_tiles);

    constexpr auto dfb_lhs_id = static_cast<uint32_t>(dfb::pre_lhs);
    constexpr auto dfb_rhs_id = static_cast<uint32_t>(dfb::pre_rhs);
    constexpr auto dfb_out_id = static_cast<uint32_t>(dfb::out);

    compute_kernel_hw_startup(dfb_lhs_id, dfb_out_id);
    DataflowBuffer dfb_lhs(dfb_lhs_id);
    DataflowBuffer dfb_rhs(dfb_rhs_id);
    DataflowBuffer dfb_out(dfb_out_id);
    copy_init(dfb_lhs_id);
    BINARY_SFPU_INIT

    for (uint32_t t = get_my_thread_id(); t < num_padded_tiles; t += get_num_threads()) {
        const bool real = t < num_tiles;
        dfb_lhs.wait_front(1);
        dfb_rhs.wait_front(1);
        if (real) {
            dfb_out.reserve_back(1);
        }

        tile_regs_acquire();
        copy_init(dfb_lhs_id);
        copy_tile(dfb_lhs_id, 0, 0);
        reconfig_data_format_srca(dfb_lhs_id, dfb_rhs_id);
        copy_init(dfb_rhs_id);
        copy_tile(dfb_rhs_id, 0, 1);
        BINARY_SFPU_OP(0, 1, 0);
        reconfig_data_format_srca(dfb_rhs_id, dfb_lhs_id);
        tile_regs_commit();

        tile_regs_wait();
        if (real) {
            pack_tile(0, dfb_out_id);
        }
        tile_regs_release();

        if (real) {
            dfb_out.push_back(1);
        }
        dfb_lhs.pop_front(1);
        dfb_rhs.pop_front(1);
    }
}
