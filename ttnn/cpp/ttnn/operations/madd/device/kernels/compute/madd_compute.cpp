// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <compute_kernel_api.h>
#include <compute_kernel_api/common.h>
#include <compute_kernel_api/eltwise_binary.h>

// #include "api/debug/dprint.h"
// #include "tt_metal/hw/inc/api/debug/dprint_pages.h"

#include <cstdint>

// --- Sample code for DPRINT debugging ---
//         DPRINT << "Processing page " << i << " / " << num_pages << ENDL();

//         DPRINT << "Reading tile A from CB " << cb_srcA_index << ENDL();
//         cb_wait_front(cb_srcA_index, 1);
// #ifdef DEBUG_PRINT_ENABLED
//         tt::compute::common::print_full_tile(cb_srcA_index, 0, false); // Math doesn't have direct access to CB data
// #endif
//         DPRINT << "Reading tile B from CB " << cb_srcB_index << ENDL();
// ---------------------------------------

void kernel_main() {
    constexpr uint32_t num_pages = get_compile_time_arg_val(0);
    constexpr uint32_t cb_srcA_index = get_compile_time_arg_val(1);
    constexpr uint32_t cb_srcB_index = get_compile_time_arg_val(2);
    constexpr uint32_t cb_srcC_index = get_compile_time_arg_val(3);
    constexpr uint32_t cb_zero_index = get_compile_time_arg_val(4);
    constexpr uint32_t cb_out_index = get_compile_time_arg_val(5);

    constexpr uint32_t dst0 = 0;
    constexpr bool acc_to_dest = true;

    binary_op_init_common(cb_srcA_index, cb_srcB_index, cb_out_index);

    // Wait for zero tile to be available (reader should generate it once)
    cb_wait_front(cb_zero_index, 1);

    for (uint32_t i = 0; i < num_pages; ++i) {
        cb_wait_front(cb_srcA_index, 1);
        cb_wait_front(cb_srcB_index, 1);

        tile_regs_acquire();  // math core acquires dst, init dsts to zero
        mul_tiles_init(cb_srcA_index, cb_srcB_index);
        mul_tiles(cb_srcA_index, cb_srcB_index, 0, 0, dst0);

        cb_pop_front(cb_srcA_index, 1);
        cb_pop_front(cb_srcB_index, 1);

        cb_wait_front(cb_srcC_index, 1);

        // Load C and add with result in DST[0] + cb_srcC_index -> DST[0]
        binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(
            cb_srcC_index);
        binary_dest_reuse_tiles<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(
            cb_srcC_index, 0, 0);

        // add_tiles_init(cb_srcC_index, cb_zero_index, acc_to_dest);
        // add_tiles(cb_srcC_index, cb_zero_index, 0, 0, dst0);

        tile_regs_commit();
        cb_pop_front(cb_srcC_index, 1);

        cb_reserve_back(cb_out_index, 1);  // Pack
        tile_regs_wait();                  // Called by packer
        pack_tile(dst0, cb_out_index, 0);  // Pack
        tile_regs_release();               // Pack

        cb_push_back(cb_out_index, 1);  // Pack
    }

    cb_pop_front(cb_zero_index, 1);
}
