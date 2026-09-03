
// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/constants.hpp>
#include <functional>

#include "api/dataflow/dataflow_buffer.h"
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

struct Reader_Node {
    uint32_t DFB_A;
    uint32_t DFB_B;
    uint32_t dest_regs_amnt;
};

template <uint32_t dfb_length, bool is_fp_32, typename cur_llk_type>
void unroll_llk();

template <uint32_t num_dst_regs, typename cur_llk_type>
void unroll_inner_loop(uint32_t register_loops);

template <uint32_t total_tiles, uint32_t dfb_length, bool is_fp_32, typename... llk_nodes>
void chain_reads(llk_nodes...) {
    constexpr uint32_t iterations = total_tiles / dfb_length;
    constexpr uint32_t leftovers = total_tiles % dfb_length;
    for (uint32_t i = 0; i < iterations; i++) {
        (..., unroll_reads<dfb_length, dest_regs_amnt, reader_nodes>(i));
    }
    (..., unroll_reads<leftovers, dest_regs_amnt, reader_nodes>(iterations));
}

// basecase for recursion

template <uint32_t dfb_length, uint32_t dest_regs_amnt, typename cur_read_type>
void unroll_llk() {
    constexpr auto cur_read = cur_read_type::node;

    static_assert(dfp_length % dest_regs_amnt == 0, "graph_kernel: a must be a multiple of b");
    constexpr uint32_t dfb_iterations = ((dfb_length) / dest_regs_amnt);  // ceil div

    for (uint32_t i = 0; i < dfb_iterations; i++) {
        unroll_inner_loop<num_dst_regs, cur_read_type>(i);
    }
}

template <typename cur_llk_type>
void print_input_DFBs(uint32_t j, uint32_t wt) {
    constexpr auto cur_llk = cur_llk_type::node;
    // Commented out so code will compile on non debug print moded. Uncomment out for debug purposes
    // DPRINT_UNPACK("=============DFB_A=============\n");
    // UNPACK(tt::compute::common::print_full_tile(cur_llk.DFB_A, j, true));
    // DPRINT_UNPACK("=============DFB_B=============\n");
    // UNPACK(tt::compute::common::print_full_tile(cur_llk.DFB_B, dfb_b_index_policy<cur_llk_type>(j, wt), true));
}
template <uint32_t num_dst_regs, typename cur_llk_type>
void unroll_inner_loop(uint32_t register_loops) {
    constexpr auto cur_llk = cur_llk_type::node;
    uint32_t wt = register_loops * num_dst_regs;
    DataflowBuffer dfb_a(cur_llk.DFB_A);
    DataflowBuffer dfb_b(cur_llk.DFB_B);
    DataflowBuffer dfb_out(cur_llk.DFB_OUT);
    tile_regs_acquire();
    dfb_a.wait_front(num_dst_regs);
    if constexpr (cur_llk.fixed_DFB_B_index == 0xFFFF) {
        dfb_b.wait_front(num_dst_regs);
    } else if constexpr (cur_llk.fixed_DFB_B_index == 0xDDDD) {
        dfb_b.wait_front(num_dst_regs + (wt));
    } else {
        dfb_b.wait_front(cur_llk.fixed_DFB_B_index + 1);
    }
    for (uint32_t j = 0; j < num_dst_regs; j++) {
        if constexpr (cur_llk.debug_mode == 1) {
            print_input_DFBs<cur_llk_type>(j, wt);
        }
        cur_llk.llk(cur_llk.DFB_A, cur_llk.DFB_B, j, dfb_b_index_policy<cur_llk_type>(j, wt), j);
        if constexpr (cur_llk.debug_mode == 1) {
            // Commented out so code will compile on non debug print moded. Uncomment out for debug purposes
            //  DPRINT_MATH("=============DEST_OUT=============\n");
            //  dprint_tensix_dest_reg(j);
        }
    }
    dfb_a.pop_front(num_dst_regs);
    if constexpr (cur_llk.fixed_DFB_B_index == 0xFFFF) {
        dfb_b.pop_front(num_dst_regs);
    }
    tile_regs_commit();
    tile_regs_wait();
    dfb_out.reserve_back(num_dst_regs);
    for (uint32_t j = 0; j < num_dst_regs; j++) {
        pack_tile(j, cur_llk.DFB_OUT);
    }
    dfb_out.push_back(num_dst_regs);
    tile_regs_release();
}
