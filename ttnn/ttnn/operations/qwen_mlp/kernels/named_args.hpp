// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "api/compile_time_args.h"

#define CT_ARG(name) get_named_compile_time_arg_val(name)
#define RT_ARG(name) get_arg_val<uint32_t>(get_named_compile_time_arg_val("rt_" name))

struct MatmulShape {
    uint32_t cb_in0;
    uint32_t cb_in1;
    uint32_t cb_out;
    uint32_t cb_partial;
    uint32_t M_block;
    uint32_t K_block;
    uint32_t N_block;
    uint32_t sub_h;
    uint32_t sub_w;
    uint32_t M_blocks;
    uint32_t N_blocks;
    uint32_t K_blocks;
    uint32_t in1_slot_tiles;
    uint32_t partial_slot_tiles;
    uint32_t in0_row_tiles;
    uint32_t in1_row_tiles;
    uint32_t out_row_tiles;
    uint32_t M_logical;
    uint32_t N_logical;
    uint32_t out_N_logical;
    bool swiglu;
};

#define MATMUL_SHAPE(phase)                  \
    MatmulShape{                             \
        CT_ARG("cb_in0"),                    \
        CT_ARG(phase "_cb_in1"),             \
        CT_ARG(phase "_cb_out"),             \
        CT_ARG(phase "_cb_partial"),         \
        CT_ARG(phase "_M_block"),            \
        CT_ARG(phase "_K_block"),            \
        CT_ARG(phase "_N_block"),            \
        CT_ARG(phase "_sub_h"),              \
        CT_ARG(phase "_sub_w"),              \
        CT_ARG(phase "_M_blocks"),           \
        CT_ARG(phase "_N_blocks"),           \
        CT_ARG(phase "_K_blocks"),           \
        CT_ARG(phase "_in1_slot_tiles"),     \
        CT_ARG(phase "_partial_slot_tiles"), \
        CT_ARG(phase "_in0_row_tiles"),      \
        CT_ARG(phase "_in1_row_tiles"),      \
        CT_ARG(phase "_out_row_tiles"),      \
        CT_ARG(phase "_M_logical"),          \
        CT_ARG(phase "_N_logical"),          \
        CT_ARG(phase "_out_N_logical"),      \
        CT_ARG(phase "_swiglu") == 1}
