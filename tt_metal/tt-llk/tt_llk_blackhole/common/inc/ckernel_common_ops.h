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

// ThreadOverride (writing another thread's ADC counters) is unsupported, and the compiler rejects it.

// SFPLOAD into LReg 8 or above loads nothing, so the instruction does nothing but apply its AddrMode to the
// RWCs, as an FPU instruction would (FidelityPhase excepted).
#define TTI_ADDRMOD_ONLY(AddrMode) TTI_SFPLOAD(p_sfpu::LCONST_0_8373, ckernel::InstrModLoadStore::FP16B, AddrMode, 0 /*dest_reg_addr*/)

#define TT_ADDRMOD_ONLY(AddrMode) TT_SFPLOAD(p_sfpu::LCONST_0_8373, ckernel::InstrModLoadStore::FP16B, AddrMode, 0 /*dest_reg_addr*/)

#define TT_OP_ADDRMOD_ONLY(AddrMode) TT_OP_SFPLOAD(p_sfpu::LCONST_0_8373, ckernel::InstrModLoadStore::FP16B, AddrMode, 0 /*dest_reg_addr*/)
