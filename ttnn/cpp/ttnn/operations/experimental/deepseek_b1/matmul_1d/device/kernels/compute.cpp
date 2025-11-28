// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

// Set to 1 to use original matmul APIs, 0 to use craqmm APIs
#define USE_ORIGINAL_MATMUL 0

#include "compute_kernel_api/matmul.h"
#include "compute_kernel_api/craqmm.h"
#include "compute_kernel_api/tile_move_copy.h"
#include <tools/profiler/kernel_profiler.hpp>
#include "debug/dprint.h"

namespace NAMESPACE {

template <bool clear_a, bool clear_b>
inline void _perf_math_loop_clear_valid(uint32_t iterations) {
    while (iterations-- > 0) {
        constexpr uint32_t cond_valid_a = clear_a ? ckernel::p_stall::SRCA_VLD : 0;
        constexpr uint32_t cond_valid_b = clear_b ? ckernel::p_stall::SRCB_VLD : 0;
#ifdef ARCH_QUASAR
        TTI_STALLWAIT(ckernel::p_stall::STALL_MATH, cond_valid_a, cond_valid_b, 0);
        TTI_CLEARDVALID((clear_b << 1) | clear_a, 0, 0, 0, 0, 0);
#else
        TTI_STALLWAIT(ckernel::p_stall::STALL_MATH, cond_valid_a | cond_valid_b);
        TTI_CLEARDVALID((clear_b << 1) | clear_a, 0);
#endif
    }
}

template <bool set_a, bool set_b>
inline void _perf_unpack_loop_set_valid(uint32_t iterations) {
    while (iterations-- > 0) {
        constexpr uint32_t cond_clear_a = set_a ? ckernel::p_stall::SRCA_CLR : 0;
        constexpr uint32_t cond_clear_b = set_b ? ckernel::p_stall::SRCB_CLR : 0;
        TTI_SETDVALID((set_b << 1) | set_a);
#ifdef ARCH_QUASAR
        TTI_STALLWAIT(ckernel::p_stall::STALL_TDMA, cond_clear_a, cond_clear_b, 0);
#else
        TTI_STALLWAIT(ckernel::p_stall::STALL_TDMA, cond_clear_a | cond_clear_b);
#endif
    }
}

void MAIN {
    constexpr uint32_t in0_block_w = get_compile_time_arg_val(0);              // inner block size in tiles
    constexpr uint32_t in0_block_num_tiles = get_compile_time_arg_val(1);      // out_subblock_h*in0_block_w
    constexpr uint32_t in1_block_num_tiles = get_compile_time_arg_val(2);      // out_subblock_w*in0_block_w
    constexpr uint32_t in1_block_w = get_compile_time_arg_val(3);              // out_subblock_w
    constexpr uint32_t out_subblock_h = get_compile_time_arg_val(4);           // inner row block size in tiles
    constexpr uint32_t out_subblock_w = get_compile_time_arg_val(5);           // inner column block size in tiles
    constexpr uint32_t out_subblock_num_tiles = get_compile_time_arg_val(6);   // out_subblock_h * out_subblock_w
    constexpr bool untilize_out = get_compile_time_arg_val(7);                 // untilize output

    constexpr uint32_t in0_cb_id = tt::CBIndex::c_0;
    constexpr uint32_t in1_cb_id = tt::CBIndex::c_1;
    constexpr uint32_t out_cb_id = tt::CBIndex::c_4;
    constexpr uint32_t mm_partials_cb_id = tt::CBIndex::c_5;
    constexpr uint32_t untilize_mode_out_cb_id = untilize_out ? mm_partials_cb_id : out_cb_id;

    constexpr uint32_t mm_out_cb_id = untilize_mode_out_cb_id;

    constexpr uint32_t in1_transpose_tile = false;

#if USE_ORIGINAL_MATMUL
    mm_block_init(
        in0_cb_id, in1_cb_id, mm_partials_cb_id, in1_transpose_tile, out_subblock_w, out_subblock_h, in0_block_w);
#else
    craqmm_block_init(in0_cb_id, in1_cb_id, mm_partials_cb_id, in1_transpose_tile, in0_block_w);
#endif

    cb_wait_front(in0_cb_id, in0_block_num_tiles);
    cb_wait_front(in1_cb_id, in1_block_num_tiles);

    tile_regs_acquire();

    // Compute output sub-block
    uint32_t dst_index = 0;  // start at 0, each call to craqmm_block internally increments dst_index
    uint32_t in0_index = 0;
    uint32_t in1_index = 0;
    // inner dim that we accumulate is the inner dim of in0/in1, which is in0_block_w

#if USE_ORIGINAL_MATMUL
    // Original matmul_block: nested loops over output tiles
    for (uint32_t inner_dim_idx = 0; inner_dim_idx < in0_block_w; ++inner_dim_idx) {
        matmul_block(
            in0_cb_id,
            in1_cb_id,
            in0_index,
            in1_index,
            dst_index,
            in1_transpose_tile,
            out_subblock_w,
            out_subblock_h,
            in0_block_w);
        in0_index++;
        in1_index += in1_block_w;
    }
#else
    {
        DeviceZoneScopedN("craqmm_block");
        volatile std::uint32_t* base_address = (std::uint32_t*)MEM_LLK_DEBUG_BASE;
        tensix_sync();
        UNPACK((base_address[1] = 1));
        MATH((base_address[2] = 2));
        PACK((base_address[3] = 3));
        while (base_address[1] != 1) {
            asm("nop");
        }
        while (base_address[2] != 2) {
            asm("nop");
        }
        while (base_address[3] != 3) {
            asm("nop");
        }
        UNPACK((base_address[5] = 5));
        MATH((base_address[6] = 6));
        PACK((base_address[7] = 7));
        while (base_address[5] != 5) {
            asm("nop");
        }
        while (base_address[6] != 6) {
            asm("nop");
        }
        while (base_address[7] != 7) {
            asm("nop");
        }
        UNPACK((base_address[1] = 0));
        MATH((base_address[2] = 0));
        PACK((base_address[3] = 0));
        while (base_address[1] != 0) {
            asm("nop");
        }
        while (base_address[2] != 0) {
            asm("nop");
        }
        while (base_address[3] != 0) {
            asm("nop");
        }
        UNPACK((base_address[5] = 0));
        MATH((base_address[6] = 0));
        PACK((base_address[7] = 0));
        uint64_t start = ckernel::read_wall_clock();
        // UNPACK: Single call with MOP looping over kt_dim (in0_block_w) internally
        // The MOP replay buffer handles unpacking both SrcA and SrcB in a tight hardware loop
        craqmm_block_unpack(in0_cb_id, in1_cb_id, in0_index, in1_index, dst_index, in1_transpose_tile, in0_block_w);
        // Comment out the line above and uncomment the line bellow to measure math isolated
        // UNPACK((_perf_unpack_loop_set_valid<true, true>(in0_block_w)));
        // MATH: Outer loop around ckernel_template::run() to execute math operations
        // Each iteration processes one tile from the K dimension
        // for (uint32_t inner_dim_idx = 0; inner_dim_idx < in0_block_w; ++inner_dim_idx) {
        //     craqmm_block_math(in0_cb_id, in1_cb_id, in0_index, in1_index, dst_index, in1_transpose_tile,
        //     in0_block_w);
        //     // in0_index and in1_index are not used by compute; so not actually needed
        // }
        craqmm_block_math(in0_cb_id, in1_cb_id, in0_index, in1_index, dst_index, in1_transpose_tile, in0_block_w);
        // Comment out the line above and uncomment the line bellow to measure unpack isolated
        // MATH((_perf_math_loop_clear_valid<true, true>(in0_block_w)));
        tensix_sync();
        uint64_t end = ckernel::read_wall_clock();
        uint64_t kernel_runtime = (end - start);
        DPRINT << "craqmm_block kernel_runtime: " << kernel_runtime << ENDL();
    }
#endif

    tile_regs_commit();
    // Pack out to output buffer
    cb_reserve_back(mm_out_cb_id, out_subblock_num_tiles);
    tile_regs_wait();

    uint32_t start_dst_index = 0;
    // Comment out the line bellow to measure only unpack
    pack_tile_block(start_dst_index, mm_out_cb_id, out_subblock_num_tiles);

    tile_regs_release();
    cb_push_back(mm_out_cb_id, out_subblock_num_tiles);

    cb_pop_front(in0_cb_id, in0_block_num_tiles);
    cb_pop_front(in1_cb_id, in1_block_num_tiles);
}
}  // namespace NAMESPACE
