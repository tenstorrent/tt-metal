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
