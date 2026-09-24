// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_addrmod.h"
#include "ckernel_template.h"
#include "ckernel_trisc_common.h"
#include "llk_defs.h"
#include "llk_math_common.h"

using namespace ckernel;

constexpr std::uint32_t GATHER_MATH_ROWS         = 16;
constexpr std::uint32_t GATHER_MATH_WINDOW_ELEMS = 32;
constexpr std::uint32_t GATHER_REPLAY_BUF_LEN    = 2;

template <std::uint32_t GATHER_ELEM_CNT>
inline void _llk_math_gather_compress_addrmod_()
{
    static_assert(GATHER_ELEM_CNT == 32 || GATHER_ELEM_CNT == 64, "Gather/compress supports 32 or 64 elements per call");

    if constexpr (GATHER_ELEM_CNT <= GATHER_MATH_WINDOW_ELEMS)
    {
        addr_mod_t {
            .srca     = {.incr = 0, .clr = 0, .cr = 0},
            .srcb     = {.incr = 0, .clr = 0, .cr = 0},
            .dest     = {.incr = 0, .clr = 0, .cr = 0},
            .fidelity = {.incr = 2, .clr = 0}}
            .set(ADDR_MOD_0);

        addr_mod_t {
            .srca     = {.incr = 0, .clr = 0, .cr = 0},
            .srcb     = {.incr = 0, .clr = 0, .cr = 0},
            .dest     = {.incr = 0, .clr = 0, .cr = 0},
            .fidelity = {.incr = 0, .clr = 1}}
            .set(ADDR_MOD_1);
    }
    else
    {
        constexpr std::uint8_t srcb_window_step = GATHER_MATH_WINDOW_ELEMS / 8;

        addr_mod_t {
            .srca     = {.incr = 0, .clr = 0, .cr = 0},
            .srcb     = {.incr = srcb_window_step, .clr = 0, .cr = 0},
            .dest     = {.incr = 0, .clr = 0, .cr = 0},
            .fidelity = {.incr = 1, .clr = 0}}
            .set(ADDR_MOD_0);

        addr_mod_t {
            .srca     = {.incr = 0, .clr = 0, .cr = 0},
            .srcb     = {.incr = static_cast<std::uint8_t>(64 - srcb_window_step), .clr = 0, .cr = 0},
            .dest     = {.incr = 0, .clr = 0, .cr = 0},
            .fidelity = {.incr = 1, .clr = 0}}
            .set(ADDR_MOD_1);
    }
}

template <std::uint32_t GATHER_ELEM_CNT>
inline void _llk_math_gather_compress_mop_config_()
{
    static_assert(GATHER_ELEM_CNT == 32 || GATHER_ELEM_CNT == 64, "Gather/compress supports 32 or 64 elements per call");

    constexpr std::uint32_t MOP_OUTER_LOOP = 1;
    constexpr std::uint32_t MOP_INNER_LOOP = (GATHER_ELEM_CNT <= GATHER_MATH_WINDOW_ELEMS) ? 1 : GATHER_ELEM_CNT / GATHER_MATH_ROWS;

    load_replay_buf<0, GATHER_REPLAY_BUF_LEN>(
        []
        {
            TTI_MOVB2D(0, 0, ADDR_MOD_0, p_movd2b::MOV_4_ROWS, 0, 0);
            TTI_MOVB2D(0, 0, ADDR_MOD_1, p_movd2b::MOV_4_ROWS, 0, 0);
        });

    ckernel_template temp(MOP_OUTER_LOOP, MOP_INNER_LOOP, TT_OP_REPLAY(0, GATHER_REPLAY_BUF_LEN, 0, 0, 0, 0));
    temp.set_end_op(TT_OP_CLEARDVALID(0, 0, 0, 0, 0b0010, 0));
    temp.program_bank0_sw_cntl(instrn_buffer);
}

template <std::uint32_t GATHER_ELEM_CNT>
inline void _llk_math_gather_compress_init_()
{
    static_assert(GATHER_ELEM_CNT == 32 || GATHER_ELEM_CNT == 64, "Gather/compress supports 32 or 64 elements per call");
    _llk_math_gather_compress_addrmod_<GATHER_ELEM_CNT>();
    _llk_math_gather_compress_mop_config_<GATHER_ELEM_CNT>();
}

inline void _llk_math_gather_compress_(const std::uint32_t dst_index)
{
    _set_dst_write_addr_<DstTileShape::Tile32x32>(dst_index);

    // Run MOP
    ckernel::ckernel_template::run_bank0_sw_cntl(instrn_buffer);
}
