// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/constants.hpp>
#include <functional>

#include "debug/dprint_pages.h"

using fn_init = void(uint32_t, uint32_t);
using fn_compute = void(uint32_t, uint32_t, uint32_t, uint32_t, uint32_t);
struct LLK_Node {
    fn_init* llk_init;
    fn_compute* llk;
    uint32_t CB_A;
    uint32_t CB_B;
    uint32_t CB_OUT;
    // If we do not want a fixed index
    // 0xFFFF: read and pop
    // 0xDDDD: read only, no pop
    // else: use a fixed index for CB_B
    uint32_t fixed_CB_B_index;
    // Note: These values were chosen for readability, and can be changed if wanted
    // If we do not want a fixed index 0xFFFF is the default value
    uint32_t fixed_dest_reg;
    // Debug mode: 0 false
    // Debug mode: 1 print all data in and out
    uint32_t debug_mode;
};
// chain_llk(add_node,sub_node,mul_node);
// chain_llk(add_node,sub_node);

// struct add_node {
//     static constexpr LLK_Node node{
//         .llk_init =add_tile_init,
//         .llk= add_tiles,
//         .CB_A= 0,
//         .CB_B =1,
//         .CB_OUT=16,
//         .fixed_CB_B_index = 0xFFFF,
//         .fixed_dest_reg=0xFFFF,
//     };
// };
// chain_llk()
template <typename cur_llk_type>
uint32_t cb_b_index_policy(uint32_t j) {
    constexpr auto cur_llk = cur_llk_type::node;
    if constexpr (cur_llk.fixed_CB_B_index == 0xFFFF) {
        return j;
    } else {
        return cur_llk.fixed_CB_B_index;
    }
}

template <uint32_t cb_length, bool is_fp_32, typename cur_llk_type>
void unroll_llk();

template <uint32_t num_dst_regs, typename cur_llk_type>
void unroll_inner_loop();

template <uint32_t total_tiles, uint32_t cb_length, bool is_fp_32, typename... llk_nodes>
void chain_llk(llk_nodes...) {
    constexpr uint32_t iterations = total_tiles / cb_length;
    constexpr uint32_t leftovers = total_tiles % cb_length;
    for (uint32_t i = 0; i < iterations; i++) {
        (..., unroll_llk<cb_length, is_fp_32, llk_nodes>());
    }
    (..., unroll_llk<leftovers, is_fp_32, llk_nodes>());
}

// //basecase for recursion
// template <uint32_t cb_length, bool is_fp_32, typename... func_nodes >
// void unroll_llk(){
//     return;
// }

template <uint32_t cb_length, bool is_fp_32, typename cur_llk_type>
void unroll_llk() {
    constexpr auto cur_llk = cur_llk_type::node;
    constexpr uint32_t num_dst_regs = (is_fp_32 ? 4 : 8);
    constexpr uint32_t cb_iterations = cb_length / num_dst_regs;
    constexpr uint32_t cb_leftovers = cb_length % num_dst_regs;

    reconfig_data_format(cur_llk.CB_A, cur_llk.CB_B);
    pack_reconfig_data_format(cur_llk.CB_OUT);
    cur_llk.llk_init(cur_llk.CB_A, cur_llk.CB_B);
    for (uint32_t i = 0; i < cb_iterations; i++) {
        unroll_inner_loop<num_dst_regs, cur_llk_type>();
    }
    unroll_inner_loop<cb_leftovers, cur_llk_type>();
}

template <typename cur_llk_type>
void print_input_CBs(uint32_t j) {
    constexpr auto cur_llk = cur_llk_type::node;
    UNPACK(DPRINT << "=============CB_A==============" << ENDL());
    UNPACK(tt::compute::common::print_full_tile(cur_llk.CB_A, j, true));
    UNPACK(DPRINT << "=============CB_B==============" << ENDL());
    UNPACK(tt::compute::common::print_full_tile(cur_llk.CB_B, cb_b_index_policy<cur_llk_type>(j), true));
}
template <uint32_t num_dst_regs, typename cur_llk_type>
void unroll_inner_loop() {
    constexpr auto cur_llk = cur_llk_type::node;
    tile_regs_acquire();
    cb_wait_front(cur_llk.CB_A, num_dst_regs);
    if constexpr (cur_llk.fixed_CB_B_index == 0xFFFF) {
        cb_wait_front(cur_llk.CB_B, num_dst_regs);
    } else {
        cb_wait_front(cur_llk.CB_B, cur_llk.fixed_CB_B_index + 1);
    }
    for (uint32_t j = 0; j < num_dst_regs; j++) {
        // TODO Add path for fixed dest reg
        if constexpr (cur_llk.debug_mode == 1) {
            print_input_CBs<cur_llk_type>(j);
        }
        cur_llk.llk(cur_llk.CB_A, cur_llk.CB_B, j, cb_b_index_policy<cur_llk_type>(j), j);
        if constexpr (cur_llk.debug_mode == 1) {
            MATH(DPRINT << "=============DEST_OUT==============" << ENDL());
            dprint_tensix_dest_reg(j);
        }
    }
    cb_pop_front(cur_llk.CB_A, num_dst_regs);
    if constexpr (cur_llk.fixed_CB_B_index == 0xFFFF) {
        cb_pop_front(cur_llk.CB_B, num_dst_regs);
    }
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(cur_llk.CB_OUT, num_dst_regs);
    for (uint32_t j = 0; j < num_dst_regs; j++) {
        pack_tile(j, cur_llk.CB_OUT);
    }
    cb_push_back(cur_llk.CB_OUT, num_dst_regs);
    tile_regs_release();
}
