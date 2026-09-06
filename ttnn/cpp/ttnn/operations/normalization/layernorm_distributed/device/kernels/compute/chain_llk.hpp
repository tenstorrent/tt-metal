// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/constants.hpp>

#include "api/dataflow/dataflow_buffer.h"

using fn_compute_5 = void(uint32_t, uint32_t, uint32_t, uint32_t, uint32_t);
using fn_compute_6 = void(uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t);
class FN_compute {
    fn_compute_5* f_5 = nullptr;
    fn_compute_6* f_6 = nullptr;

public:
    constexpr FN_compute(fn_compute_5* func_5) : f_5(func_5), f_6(nullptr) {}
    constexpr FN_compute(fn_compute_6* func_6) : f_5(nullptr), f_6(func_6) {}

    constexpr void operator()(uint32_t var1, uint32_t var2, uint32_t var3, uint32_t var4, uint32_t var5) const {
        if (f_5) {
            f_5(var1, var2, var3, var4, var5);
        } else {
            f_6(var1, var2, var3, var4, var5, 0);
        }
    }
};
using fn_init = void(uint32_t, uint32_t, uint32_t);
using fn_compute = void(uint32_t, uint32_t, uint32_t, uint32_t, uint32_t);
struct LLK_Node {
    fn_init* llk_init;
    FN_compute llk;
    // Dataflow-buffer handles. Declared as uint32_t so a dfb::<name> binding token converts
    // straight in at constexpr time, and so the LLK entry points, which still take raw ids,
    // can be called with the field directly.
    uint32_t DFB_A;
    uint32_t DFB_B;
    uint32_t DFB_OUT;
    // If we do not want a fixed index
    // 0xFFFF: read and pop
    // 0xDDDD: read only, no pop
    // else: use a fixed index for DFB_B
    uint32_t fixed_DFB_B_index;
};
template <typename cur_llk_type>
uint32_t dfb_b_index_policy(uint32_t j, uint32_t wt) {
    constexpr auto cur_llk = cur_llk_type::node;
    if constexpr (cur_llk.fixed_DFB_B_index == 0xFFFF) {
        return j;
    } else if constexpr (cur_llk.fixed_DFB_B_index == 0xDDDD) {
        return wt + j;
    } else {
        return cur_llk.fixed_DFB_B_index;
    }
}

template <uint32_t dfb_length, bool is_fp_32, typename cur_llk_type>
void unroll_llk();

template <uint32_t num_dst_regs, typename cur_llk_type>
void unroll_inner_loop(uint32_t register_loops);

template <uint32_t total_tiles, uint32_t dfb_length, bool is_fp_32, typename... llk_nodes>
void chain_llk(llk_nodes...) {
    constexpr uint32_t iterations = total_tiles / dfb_length;
    constexpr uint32_t leftovers = total_tiles % dfb_length;
    for (uint32_t i = 0; i < iterations; i++) {
        (..., unroll_llk<dfb_length, is_fp_32, llk_nodes>());
    }
    (..., unroll_llk<leftovers, is_fp_32, llk_nodes>());
}

// basecase for recursion

template <uint32_t dfb_length, bool is_fp_32, typename cur_llk_type>
void unroll_llk() {
    constexpr auto cur_llk = cur_llk_type::node;
    constexpr uint32_t num_dst_regs = (is_fp_32 ? 4 : 8);

    constexpr uint32_t dfb_iterations = dfb_length / num_dst_regs;
    constexpr uint32_t dfb_leftovers = dfb_length % num_dst_regs;

    reconfig_data_format(cur_llk.DFB_A, cur_llk.DFB_B);
    pack_reconfig_data_format(cur_llk.DFB_OUT);
    cur_llk.llk_init(cur_llk.DFB_A, cur_llk.DFB_B, __builtin_LINE());
    for (uint32_t i = 0; i < dfb_iterations; i++) {
        unroll_inner_loop<num_dst_regs, cur_llk_type>(i);
    }
    unroll_inner_loop<dfb_leftovers, cur_llk_type>(dfb_iterations);
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
        cur_llk.llk(cur_llk.DFB_A, cur_llk.DFB_B, j, dfb_b_index_policy<cur_llk_type>(j, wt), j);
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
