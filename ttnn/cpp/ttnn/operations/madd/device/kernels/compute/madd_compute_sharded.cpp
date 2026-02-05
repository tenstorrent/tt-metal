// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <compute_kernel_api.h>
#include <compute_kernel_api/common.h>
#include <compute_kernel_api/eltwise_binary.h>

#include "tt_metal/tools/profiler/kernel_profiler.hpp"

// #include "api/debug/dprint.h"
// #include "tt_metal/hw/inc/api/debug/dprint_pages.h"

#include <cstdint>

#ifdef PROFILE_BUILD_MADD
#define MADD_PROFILE(name) DeviceZoneScopedN(name)
#else
#define MADD_PROFILE(name)
#endif

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

    constexpr uint32_t NUM_DST_REGS = 8;

    {
        MADD_PROFILE("Long init");
        binary_op_init_common(cb_srcA_index, cb_srcB_index, cb_out_index);
    }

    {
        MADD_PROFILE("CB Waits");
        // Wait for zero tile to be available (reader should generate it once)
        cb_wait_front(cb_zero_index, 1);
        cb_wait_front(cb_srcA_index, num_pages);
        cb_wait_front(cb_srcB_index, num_pages);
        cb_wait_front(cb_srcC_index, num_pages);
    }
    {
        MADD_PROFILE("CB reserve");
        cb_reserve_back(cb_out_index, num_pages);
    }

    // Outer loop: process pages in chunks of up to 8 (using all 8 dst registers)
    for (uint32_t outer_idx = 0; outer_idx < num_pages; outer_idx += NUM_DST_REGS) {
        // Calculate how many pages to process in this iteration (up to 8)
        uint32_t pages_this_iter = (outer_idx + NUM_DST_REGS <= num_pages) ? NUM_DST_REGS : (num_pages - outer_idx);

        {
            MADD_PROFILE("Math waits");
            tile_regs_acquire();  // math core acquires dst registers
        }

        // Initialize mul once for the batch
        {
            MADD_PROFILE("Mul init");
            mul_tiles_init(cb_srcA_index, cb_srcB_index);
        }

        // Inner loop for mul: process up to 8 pages into dst0..dst7
        {
            MADD_PROFILE("Mul");
            for (uint32_t inner_idx = 0; inner_idx < pages_this_iter; ++inner_idx) {
                uint32_t page_idx = outer_idx + inner_idx;
                uint32_t dst_reg = inner_idx;
                mul_tiles(cb_srcA_index, cb_srcB_index, page_idx, page_idx, dst_reg);
            }
        }

        // Initialize add once for the batch
        {
            MADD_PROFILE("Add init");
            binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(
                cb_srcC_index);
        }

        // Inner loop for add: add C to each result in dst registers
        {
            MADD_PROFILE("Add");
            for (uint32_t inner_idx = 0; inner_idx < pages_this_iter; ++inner_idx) {
                uint32_t page_idx = outer_idx + inner_idx;
                uint32_t dst_reg = inner_idx;
                binary_dest_reuse_tiles<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(
                    cb_srcC_index, page_idx, dst_reg);
            }
        }

        {
            MADD_PROFILE("DST Commit");
            tile_regs_commit();
        }

        {
            MADD_PROFILE("Packer waits");
            tile_regs_wait();  // Called by packer
        }

        // Inner loop for packer: pack all dst registers to output
        {
            MADD_PROFILE("Packer Works");
            for (uint32_t inner_idx = 0; inner_idx < pages_this_iter; ++inner_idx) {
                uint32_t page_idx = outer_idx + inner_idx;
                uint32_t dst_reg = inner_idx;
                pack_tile(dst_reg, cb_out_index, page_idx);
            }
            tile_regs_release();
        }
    }

    {
        MADD_PROFILE("CB Push");
        cb_push_back(cb_out_index, num_pages);
    }
    {
        MADD_PROFILE("CB Pop");
        cb_pop_front(cb_srcA_index, num_pages);
        cb_pop_front(cb_srcB_index, num_pages);
        cb_pop_front(cb_srcC_index, num_pages);

        cb_pop_front(cb_zero_index, 1);
    }
}
