// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <common/inc/ckernel_ops.h>

#define TRISC_OP_SWIZZLE(x) ((((x) >> 30) & 0x3) | (((x) & 0x3FFFFFFF) << 2))
#define INSTRUCTION_WORD(x) __asm__ __volatile__(".word (%0)" : : "i"((x)))

#undef TTI_INSN
#define TTI_INSN(ENCODING) INSTRUCTION_WORD(TRISC_OP_SWIZZLE(ENCODING))
#undef TT_INSN
#define TT_INSN(ENCODING) ckernel::instrn_buffer[0] = (ENCODING)
