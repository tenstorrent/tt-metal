// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/where.h"
#include "api/compute/eltwise_unary/lerp.h"
#include "api/compute/eltwise_unary/mac.h"
#include "ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_utils_common.hpp"
#include "ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_utils_sfpu.hpp"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"
#include "ttnn/operations/eltwise/ternary/device/kernels/compute/ternary_sfpu_copy_init.hpp"

// Operands in in_a's formats share its copy init, which leaves the SFPU state the op init set, so the inits run once.
#if !defined(ARCH_QUASAR) && (defined(TRISC_UNPACK) || defined(TRISC_MATH))
constexpr bool init_once = unpack_src_format[dfb::in_a] == static_cast<std::uint32_t>(DataFormat::Float16_b) &&
                           copy_init_shared_with_c0<dfb::in_b>() && copy_init_shared_with_c0<dfb::in_c>();
#else
constexpr bool init_once = false;
#endif

ALWI void process_tile(uint32_t freq, uint32_t tile_start, uint32_t num_tiles_per_cycle) {
    using namespace ckernel;

    DataflowBuffer predicate_dfb(dfb::in_a);
    DataflowBuffer true_dfb(dfb::in_b);
    DataflowBuffer false_dfb(dfb::in_c);
    DataflowBuffer dfb_out(dfb::out);

    // 3-tensor broadcast-aware synchronization - wait for broadcast DFBs outside loop
#if BCAST_A
    predicate_dfb.wait_front(num_tiles_per_cycle);  // predicate_dfb is broadcast
#endif
#if BCAST_B
    true_dfb.wait_front(num_tiles_per_cycle);  // true_dfb is broadcast
#endif
#if BCAST_C
    false_dfb.wait_front(num_tiles_per_cycle);  // false_dfb is broadcast
#endif

    for (uint32_t j = tile_start; j < freq; ++j) {
        // Wait for non-broadcast DFBs inside loop
#if !BCAST_A
        predicate_dfb.wait_front(num_tiles_per_cycle);
#endif
#if !BCAST_B
        true_dfb.wait_front(num_tiles_per_cycle);
#endif
#if !BCAST_C
        false_dfb.wait_front(num_tiles_per_cycle);
#endif

        dfb_out.reserve_back(num_tiles_per_cycle);

        tile_regs_acquire();

        // Copy all 3 inputs to destination registers.
        // copy_init does not reconfigure the unpacker data format, and the operands may differ in dtype.
        reconfig_data_format_srca(dfb::in_c, dfb::in_a);
        copy_init_unless<init_once>(dfb::in_a);
        copy_tile(dfb::in_a, 0, 0);  // predicate to reg 0, 3, 6, ...

        reconfig_data_format_srca(dfb::in_a, dfb::in_b);
        copy_init_unless<init_once>(dfb::in_b);
        copy_tile(dfb::in_b, 0, 1);  // true to reg 1, 4, 7, ...

        reconfig_data_format_srca(dfb::in_b, dfb::in_c);
        copy_init_unless<init_once>(dfb::in_c);
        copy_tile(dfb::in_c, 0, 2);  // false to reg 2, 5, 8, ...

        // Perform the ternary operation
        if constexpr (!init_once) {
            TERNARY_SFPU_OP_INIT();
        }
        TERNARY_SFPU_OP_FUNC(0, 1, 2, 0);

        tile_regs_commit();

        tile_regs_wait();

        pack_tile(0, dfb::out);  // result is stored in predicate register
        tile_regs_release();

        dfb_out.push_back(num_tiles_per_cycle);

        // Pop non-broadcast DFBs inside loop
#if !BCAST_A
        predicate_dfb.pop_front(num_tiles_per_cycle);
#endif
#if !BCAST_B
        true_dfb.pop_front(num_tiles_per_cycle);
#endif
#if !BCAST_C
        false_dfb.pop_front(num_tiles_per_cycle);
#endif
    }

    // Pop broadcast DFBs outside loop
#if BCAST_A
    predicate_dfb.pop_front(num_tiles_per_cycle);
#endif
#if BCAST_B
    true_dfb.pop_front(num_tiles_per_cycle);
#endif
#if BCAST_C
    false_dfb.pop_front(num_tiles_per_cycle);
#endif
}

void kernel_main() {
    uint32_t num_tiles = get_arg(args::num_tiles);
    uint32_t tile_freq = get_arg(args::tile_freq);
    uint32_t tile_start = get_arg(args::tile_start);

    constexpr uint32_t num_tiles_per_cycle = get_arg(args::num_tiles_per_cycle);

    if (num_tiles == 0) {
        return;
    }

    compute_kernel_hw_startup(dfb::in_a, dfb::out);
    copy_init(dfb::in_a);
    if constexpr (init_once) {
        TERNARY_SFPU_OP_INIT();
    }

    uint32_t complete_iterations = (num_tiles + tile_start) / tile_freq;
    uint32_t remaining_iterations = (num_tiles + tile_start) % tile_freq;

    for (uint32_t i = 0; i < complete_iterations; ++i, tile_start = 0) {
        process_tile(tile_freq, tile_start, num_tiles_per_cycle);
    }

    if (remaining_iterations > 0) {
        process_tile(remaining_iterations, tile_start, num_tiles_per_cycle);
    }
}
