// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_ops.h"

#define TTI_UNPACR_COMMON(Unpack_block_selection, AddrMode, SetDatValid) \
    TTI_UNPACR(                                                          \
        Unpack_block_selection,                                          \
        AddrMode,                                                        \
        0 /*CfgContextCntInc*/,                                          \
        0 /*CfgContextId*/,                                              \
        0 /*AddrCntContextId*/,                                          \
        1 /*OvrdThreadId*/,                                              \
        SetDatValid,                                                     \
        0 /*srcb_bcast*/,                                                \
        0 /*ZeroWrite2*/,                                                \
        0 /*AutoIncContextID*/,                                          \
        0 /*RowSearch*/,                                                 \
        0 /*SearchCacheFlush*/,                                          \
        1 /*Last*/)

#define TT_OP_UNPACR_COMMON(Unpack_block_selection, AddrMode, SetDatValid) \
    TT_OP_UNPACR(                                                          \
        Unpack_block_selection,                                            \
        AddrMode,                                                          \
        0 /*CfgContextCntInc*/,                                            \
        0 /*CfgContextId*/,                                                \
        0 /*AddrCntContextId*/,                                            \
        1 /*OvrdThreadId*/,                                                \
        SetDatValid,                                                       \
        0 /*srcb_bcast*/,                                                  \
        0 /*ZeroWrite2*/,                                                  \
        0 /*AutoIncContextID*/,                                            \
        0 /*RowSearch*/,                                                   \
        0 /*SearchCacheFlush*/,                                            \
        1 /*Last*/)

#define TTI_UNPACR_COMMON_EXPLICIT_CONTEXT(Unpack_block_selection, AddrMode, CfgContextId, SetDatValid) \
    TTI_UNPACR(                                                                                         \
        Unpack_block_selection,                                                                         \
        AddrMode,                                                                                       \
        0 /*CfgContextCntInc*/,                                                                         \
        CfgContextId,                                                                                   \
        0 /*AddrCntContextId*/,                                                                         \
        1 /*OvrdThreadId*/,                                                                             \
        SetDatValid,                                                                                    \
        0 /*srcb_bcast*/,                                                                               \
        0 /*ZeroWrite2*/,                                                                               \
        0 /*AutoIncContextID*/,                                                                         \
        0 /*RowSearch*/,                                                                                \
        0 /*SearchCacheFlush*/,                                                                         \
        1 /*Last*/)

#define TT_OP_UNPACR_COMMON_EXPLICIT_CONTEXT(Unpack_block_selection, AddrMode, CfgContextId, SetDatValid) \
    TT_OP_UNPACR(                                                                                         \
        Unpack_block_selection,                                                                           \
        AddrMode,                                                                                         \
        0 /*CfgContextCntInc*/,                                                                           \
        CfgContextId,                                                                                     \
        0 /*AddrCntContextId*/,                                                                           \
        1 /*OvrdThreadId*/,                                                                               \
        SetDatValid,                                                                                      \
        0 /*srcb_bcast*/,                                                                                 \
        0 /*ZeroWrite2*/,                                                                                 \
        0 /*AutoIncContextID*/,                                                                           \
        0 /*RowSearch*/,                                                                                  \
        0 /*SearchCacheFlush*/,                                                                           \
        1 /*Last*/)

#define TTI_UNPACR_COMMON_EXPLICIT_CONTEXT_AND_COUNTER(Unpack_block_selection, AddrMode, CfgContextId, AddrCntContextId, SetDatValid) \
    TTI_UNPACR(                                                                                                                       \
        Unpack_block_selection,                                                                                                       \
        AddrMode,                                                                                                                     \
        0 /*CfgContextCntInc*/,                                                                                                       \
        CfgContextId,                                                                                                                 \
        AddrCntContextId,                                                                                                             \
        1 /*OvrdThreadId*/,                                                                                                           \
        SetDatValid,                                                                                                                  \
        0 /*srcb_bcast*/,                                                                                                             \
        0 /*ZeroWrite2*/,                                                                                                             \
        0 /*AutoIncContextID*/,                                                                                                       \
        0 /*RowSearch*/,                                                                                                              \
        0 /*SearchCacheFlush*/,                                                                                                       \
        1 /*Last*/)

#define TT_OP_UNPACR_COMMON_EXPLICIT_CONTEXT_AND_COUNTER(Unpack_block_selection, AddrMode, CfgContextId, AddrCntContextId, SetDatValid) \
    TT_OP_UNPACR(                                                                                                                       \
        Unpack_block_selection,                                                                                                         \
        AddrMode,                                                                                                                       \
        0 /*CfgContextCntInc*/,                                                                                                         \
        CfgContextId,                                                                                                                   \
        AddrCntContextId,                                                                                                               \
        1 /*OvrdThreadId*/,                                                                                                             \
        SetDatValid,                                                                                                                    \
        0 /*srcb_bcast*/,                                                                                                               \
        0 /*ZeroWrite2*/,                                                                                                               \
        0 /*AutoIncContextID*/,                                                                                                         \
        0 /*RowSearch*/,                                                                                                                \
        0 /*SearchCacheFlush*/,                                                                                                         \
        1 /*Last*/)

#define TTI_PACR_COMMON(AddrMode, ZeroWrite, PackSel, Flush, Last) \
    TTI_PACR(                                                      \
        0 /*CfgContext*/,                                          \
        0 /*RowPadZero*/,                                          \
        0 /*DstAccessMode*/,                                       \
        AddrMode,                                                  \
        0 /*AddrCntContext*/,                                      \
        ZeroWrite,                                                 \
        PackSel,                                                   \
        0 /*OvrdThreadId*/,                                        \
        0 /*Concat*/,                                              \
        0 /*CtxtCtrl*/,                                            \
        Flush,                                                     \
        Last)

#define TT_OP_PACR_COMMON(AddrMode, ZeroWrite, PackSel, Flush, Last) \
    TT_OP_PACR(                                                      \
        0 /*CfgContext*/,                                            \
        0 /*RowPadZero*/,                                            \
        0 /*DstAccessMode*/,                                         \
        AddrMode,                                                    \
        0 /*AddrCntContext*/,                                        \
        ZeroWrite,                                                   \
        PackSel,                                                     \
        0 /*OvrdThreadId*/,                                          \
        0 /*Concat*/,                                                \
        0 /*CtxtCtrl*/,                                              \
        Flush,                                                       \
        Last)

#define TTI_REG2FLOP_COMMON(SizeSel, FlopIndex, RegIndex) TTI_REG2FLOP(SizeSel, 0 /*TargetSel*/, 0 /*ByteOffset*/, 0 /*ContextId*/, FlopIndex, RegIndex)

#define TT_OP_REG2FLOP_COMMON(SizeSel, FlopIndex, RegIndex) TT_OP_REG2FLOP(SizeSel, 0 /*TargetSel*/, 0 /*ByteOffset*/, 0 /*ContextId*/, FlopIndex, RegIndex)

// SETADC-family variants exposing the thread-id override, which the plain macros have no operand for.
// ThreadOverride picks whose per-thread ADC set the write lands on: 0 is the issuing thread, non-zero
// selects one explicitly -- pass p_setadc::THREAD_OVRD_*. SETADCXX has no override. See
// tt-isa-documentation WormholeB0/TensixTile/TensixCoprocessor/SETADC.md for the field semantics.
#define SETADC_THREAD_OVRD_VALUE_SHIFT 16
#define SETADC_THREAD_OVRD_CH1_SHIFT   3

#define TTI_SETADC_THREAD_OVERRIDE(CntSetMask, ChannelIndex, DimensionIndex, ThreadId, Value) \
    TTI_SETADC(CntSetMask, ChannelIndex, DimensionIndex, (((ThreadId) << SETADC_THREAD_OVRD_VALUE_SHIFT) | (Value)))

#define TT_SETADC_THREAD_OVERRIDE(CntSetMask, ChannelIndex, DimensionIndex, ThreadId, Value) \
    TT_SETADC(CntSetMask, ChannelIndex, DimensionIndex, (((ThreadId) << SETADC_THREAD_OVRD_VALUE_SHIFT) | (Value)))

#define TT_OP_SETADC_THREAD_OVERRIDE(CntSetMask, ChannelIndex, DimensionIndex, ThreadId, Value) \
    TT_OP_SETADC(CntSetMask, ChannelIndex, DimensionIndex, (((ThreadId) << SETADC_THREAD_OVRD_VALUE_SHIFT) | (Value)))

#define TTI_SETADCXY_THREAD_OVERRIDE(CntSetMask, ThreadId, Ch1_Y, Ch1_X, Ch0_Y, Ch0_X, BitMask) \
    TTI_SETADCXY(CntSetMask, (((ThreadId) << SETADC_THREAD_OVRD_CH1_SHIFT) | (Ch1_Y)), Ch1_X, Ch0_Y, Ch0_X, BitMask)

#define TT_SETADCXY_THREAD_OVERRIDE(CntSetMask, ThreadId, Ch1_Y, Ch1_X, Ch0_Y, Ch0_X, BitMask) \
    TT_SETADCXY(CntSetMask, (((ThreadId) << SETADC_THREAD_OVRD_CH1_SHIFT) | (Ch1_Y)), Ch1_X, Ch0_Y, Ch0_X, BitMask)

#define TT_OP_SETADCXY_THREAD_OVERRIDE(CntSetMask, ThreadId, Ch1_Y, Ch1_X, Ch0_Y, Ch0_X, BitMask) \
    TT_OP_SETADCXY(CntSetMask, (((ThreadId) << SETADC_THREAD_OVRD_CH1_SHIFT) | (Ch1_Y)), Ch1_X, Ch0_Y, Ch0_X, BitMask)

#define TTI_SETADCZW_THREAD_OVERRIDE(CntSetMask, ThreadId, Ch1_W, Ch1_Z, Ch0_W, Ch0_Z, BitMask) \
    TTI_SETADCZW(CntSetMask, (((ThreadId) << SETADC_THREAD_OVRD_CH1_SHIFT) | (Ch1_W)), Ch1_Z, Ch0_W, Ch0_Z, BitMask)

#define TT_SETADCZW_THREAD_OVERRIDE(CntSetMask, ThreadId, Ch1_W, Ch1_Z, Ch0_W, Ch0_Z, BitMask) \
    TT_SETADCZW(CntSetMask, (((ThreadId) << SETADC_THREAD_OVRD_CH1_SHIFT) | (Ch1_W)), Ch1_Z, Ch0_W, Ch0_Z, BitMask)

#define TT_OP_SETADCZW_THREAD_OVERRIDE(CntSetMask, ThreadId, Ch1_W, Ch1_Z, Ch0_W, Ch0_Z, BitMask) \
    TT_OP_SETADCZW(CntSetMask, (((ThreadId) << SETADC_THREAD_OVRD_CH1_SHIFT) | (Ch1_W)), Ch1_Z, Ch0_W, Ch0_Z, BitMask)

// ZEROACC that clears nothing, so the instruction is a no-op apart from applying its AddrMode to the
// address counters.
#define TTI_ZEROACC_ADDRMOD_ONLY(AddrMode) TTI_ZEROACC(p_zeroacc::CLR_16, 0 /*use_32_bit_mode*/, 0 /*clear_zero_flags*/, AddrMode, p_zeroacc::WHERE_NOP)

#define TT_ZEROACC_ADDRMOD_ONLY(AddrMode) TT_ZEROACC(p_zeroacc::CLR_16, 0 /*use_32_bit_mode*/, 0 /*clear_zero_flags*/, AddrMode, p_zeroacc::WHERE_NOP)

#define TT_OP_ZEROACC_ADDRMOD_ONLY(AddrMode) TT_OP_ZEROACC(p_zeroacc::CLR_16, 0 /*use_32_bit_mode*/, 0 /*clear_zero_flags*/, AddrMode, p_zeroacc::WHERE_NOP)
