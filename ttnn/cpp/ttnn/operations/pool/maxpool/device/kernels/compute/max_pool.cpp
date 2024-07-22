// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

// #include "compute_kernel_api.h"
#include "compute_kernel_api/tilize.h"
#include "compute_kernel_api/reduce.h"
#include "compute_kernel_api/pack_untilize.h"
// #include "tools/profiler/kernel_profiler.hpp"

#define DEBUG_PRINT 0

#if DEBUG_PRINT == 1
    #include "debug/dprint.h"

    inline void print_full_tile(uint32_t cb_id, uint32_t tile_id = 0, bool untilize = false) {
        UNPACK(( DPRINT << "======" << ENDL() ));
        for (uint16_t r = 0; r < 32; ++ r) {
            SliceRange sr = SliceRange{.h0 = r, .h1 = (uint16_t)(r+1), .hs = 1, .w0 = 0, .w1 = 32, .ws = 1};
            UNPACK(( DPRINT << (uint)r << TileSlice(cb_id, tile_id, sr, true, untilize) << ENDL() ));
        }
        UNPACK(( DPRINT << "++++++" << ENDL() ));
    }


    inline void print_cb_details(uint32_t cb_id) {
        UNPACK(DPRINT("cb_id " << cb_id << ": { "
                << "size: " << cb_interface[cb_id].fifo_size << ", "
                << "limit: " << cb_interface[cb_id].fifo_limit << ", "
                << "page_size: " << cb_interface[cb_id].fifo_page_size << ", "
                << "num_pages: " << cb_interface[cb_id].fifo_num_pages << ", "
                << "rd_ptr: " << cb_interface[cb_id].fifo_rd_ptr << ", "
                << "wr_ptr: " << cb_interface[cb_id].fifo_wr_ptr << ", "
                << "wr_tile_ptr: " << cb_interface[cb_id].fifo_wr_tile_ptr << " }"));
    }
#endif

inline void tilize(uint32_t out_nelems,
                   uint32_t in_cb_id,
                   uint32_t in_ntiles_hw,
                   uint32_t in_ntiles_c,
                   uint32_t in_ntiles_hwc,
                   uint32_t window_size_hw,
                   uint32_t out_cb_id) {
    tilize_init_short(in_cb_id, in_ntiles_hwc);
    for (uint32_t out_elem_i = 0; out_elem_i < out_nelems; ++ out_elem_i) {
        cb_wait_front(in_cb_id, 1);
        cb_reserve_back(out_cb_id, in_ntiles_hwc);
        tilize_block(in_cb_id, in_ntiles_hwc, out_cb_id);  // TODO: need to ensure the ordering for reduction when in_ntiles_hw > 1
        // print_full_tile(in_cb_id, 0, false);
        // PDPRINT("OUT TILE :: " << TileSlice(out_cb_id, 0, srr, true, true));
        // print_cb_details(in_cb_id);
        cb_push_back(out_cb_id, in_ntiles_hwc);
        cb_pop_front(in_cb_id, 1);
    }
    tilize_uninit(in_cb_id);
}

inline void reduce_h_orig(uint32_t out_nelems,
                     uint32_t in_cb_id,
                     uint32_t in_scalar_cb_id,
                     uint32_t in_ntiles_hw,
                     uint32_t in_ntiles_c,
                     uint32_t in_ntiles_hwc,
                     uint32_t out_ntiles_c,
                     uint32_t out_cb_id) {
    cb_wait_front(in_cb_id, in_ntiles_hwc * out_nelems);
    cb_reserve_back(out_cb_id, out_ntiles_c * out_nelems);
    reduce_init_delta<false, PoolType::MAX, ReduceDim::REDUCE_COL>(out_cb_id);
    uint32_t base_tile_id = 0;
    for (uint32_t c_i = 0; c_i < in_ntiles_c * out_nelems; ++c_i) {
        // add to accumulator all the in_ntiles_hw in a column of tiles
        acquire_dst(tt::DstMode::Half);
        uint32_t dst_i = 0; // TODO [AS]: Use more than one dst tile at a time
        for (uint32_t hw_i = 0; hw_i < in_ntiles_hw; ++hw_i) {
            uint32_t tile_i = base_tile_id + hw_i;
            reduce_tile(in_cb_id, in_scalar_cb_id, tile_i, 0, dst_i);
        }
        pack_tile(dst_i, out_cb_id);
        release_dst(tt::DstMode::Half);
        base_tile_id += in_ntiles_hw;
    }
    reduce_revert_delta(out_cb_id);
    cb_push_back(out_cb_id, out_ntiles_c * out_nelems);
    cb_pop_front(in_cb_id, in_ntiles_hwc * out_nelems);
}

template<uint32_t in_ntiles_hw, uint32_t in_ntiles_c>
inline void reduce_h(uint32_t out_nelems,
                     uint32_t in_cb_id,
                     uint32_t in_scalar_cb_id,
                     uint32_t in_ntiles_hwc,
                     uint32_t out_ntiles_c,
                     uint32_t out_cb_id) {
    cb_wait_front(in_cb_id, in_ntiles_hwc * out_nelems);
    reduce_init_delta_no_pack(in_cb_id, in_scalar_cb_id);
    pack_untilize_dst_init_short<in_ntiles_c>();
    cb_reserve_back(out_cb_id, out_ntiles_c * out_nelems);
    tile_regs_acquire();
    for (uint32_t c_i = 0; c_i < in_ntiles_c * out_nelems; ++c_i) {
        // add to accumulator all the in_ntiles_hw in a column of tiles
        reduce_tile(in_cb_id, in_scalar_cb_id, c_i, 0, c_i);
    }
    tile_regs_wait();
    tile_regs_commit();
    pack_untilize_dst<in_ntiles_c>(out_cb_id);
    tile_regs_release();
    pack_untilize_uninit();
    cb_push_back(out_cb_id, out_ntiles_c * out_nelems);
    cb_pop_front(in_cb_id, in_ntiles_hwc * out_nelems);
}

template<uint32_t in_ntiles_hw, uint32_t in_ntiles_c, uint32_t out_ntiles_c, uint32_t out_nelems>
inline void reduce_h_fused(
    const uint32_t in_cb_id,
    const uint32_t in_scalar_cb_id,
    const uint32_t in_ntiles_hwc,
    const uint32_t out_cb_id) {

    constexpr uint32_t num_output_tiles = out_ntiles_c * out_nelems;
    cb_reserve_back(out_cb_id, out_ntiles_c * out_nelems);
    tile_regs_acquire();
    for (uint32_t out_elem_i = 0; out_elem_i < out_nelems; ++ out_elem_i) {
        cb_wait_front(in_cb_id, 1);
        unpack_tilizeA_B_block(in_cb_id, in_scalar_cb_id, in_ntiles_hwc, 0 /*tile idx for Src b is 0 because only 1 tile of constants is loaded*/);

        for (uint32_t c_i = 0; c_i < in_ntiles_c; ++c_i) {
            reduce_tile_math(in_ntiles_c*out_elem_i + c_i);
        }
        cb_pop_front(in_cb_id, 1);
    }

    tile_regs_wait();
    tile_regs_commit();
    pack_untilize_dst<num_output_tiles>(out_cb_id);  // <-- PACK
    tile_regs_release();

    cb_push_back(out_cb_id, out_ntiles_c * out_nelems);
}

namespace NAMESPACE {

void MAIN {
    constexpr uint32_t in_cb_id = tt::CB::c_in0;
    constexpr uint32_t in_scalar_cb_id = tt::CB::c_in4;
    constexpr uint32_t in_tiled_cb_id = tt::CB::c_intermed0;
    constexpr uint32_t out_cb_id = tt::CB::c_out0;

    const uint32_t in_ntiles_hw = get_compile_time_arg_val(0);
    const uint32_t in_ntiles_c = get_compile_time_arg_val(1);
    const uint32_t in_ntiles_hwc = get_compile_time_arg_val(2);
    const uint32_t window_size_hw = get_compile_time_arg_val(3);
    const uint32_t out_h = get_compile_time_arg_val(4);
    const uint32_t out_w = get_compile_time_arg_val(5);
    const uint32_t out_ntiles_c = get_compile_time_arg_val(7);
    const uint32_t out_nelems = get_compile_time_arg_val(8);
    const uint32_t out_w_loop_count = get_compile_time_arg_val(9);
    const uint32_t nbatch = get_compile_time_arg_val(10);
    const uint32_t out_h_per_core = get_compile_time_arg_val(11);
    const uint32_t num_output_tiles = out_ntiles_c * out_nelems;

    // tilize_init(in_cb_id, in_ntiles_hwc, in_tiled_cb_id);
    tilizeA_B_reduce_init(in_cb_id, in_scalar_cb_id, in_ntiles_hwc, out_cb_id);
    pack_untilize_dst_init_short<num_output_tiles>(out_cb_id);

    #if DEBUG_PRINT == 1
        print_full_tile(in_cb_id);
        print_cb_details(in_tiled_cb_id);
        print_cb_details(out_cb_id);
        UNPACK(DPRINT << "nbatch = " << nbatch << ENDL());
        UNPACK(DPRINT << "out_h_per_core = " << out_h_per_core << ENDL());
        UNPACK(DPRINT << "out_w_loop_count = " << out_w_loop_count << ENDL());
    #endif

    cb_wait_front(in_scalar_cb_id, 1);
    for (uint32_t batch = 0; batch < nbatch; ++ batch) {
        for (uint32_t out_h_i = 0; out_h_i < out_h_per_core; ++out_h_i) {
            for (uint32_t out_w_i = 0; out_w_i < out_w_loop_count; ++out_w_i) {
                // NOTE: Assuming in_ntiles_hw < 8 for now.
                // TODO: subblocking to support this.
                // tilize
                // tilize(out_nelems, in_cb_id, in_ntiles_hw, in_ntiles_c, in_ntiles_hwc, window_size_hw, in_tiled_cb_id);
                // Reduce H
                // reduce_h_orig(out_nelems, in_tiled_cb_id, in_scalar_cb_id, in_ntiles_hw, in_ntiles_c, in_ntiles_hwc, out_ntiles_c, out_cb_id);
                // reduce_h<in_ntiles_hw, in_ntiles_c>(out_nelems, in_tiled_cb_id, in_scalar_cb_id, in_ntiles_hwc, out_ntiles_c, out_cb_id);
                reduce_h_fused<in_ntiles_hw, in_ntiles_c, out_ntiles_c, out_nelems>(in_cb_id, in_scalar_cb_id, in_ntiles_hwc, out_cb_id);
            }
        }
    }
    cb_pop_front(in_scalar_cb_id, 1);
}

}  // namespace NAMESPACE
