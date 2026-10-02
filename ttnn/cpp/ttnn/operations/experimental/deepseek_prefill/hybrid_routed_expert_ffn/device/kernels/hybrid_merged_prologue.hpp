// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// The two-pass contract the three union kernels share.
//
// A program may hold only one kernel per processor per core, so the two halves cannot be placed
// as separate kernels on the same grid -- both bodies are compiled into one binary per RISC-V
// instead. Pass A runs every expert at or below the model's measured token threshold on the fused
// implementation, pass B runs the rest on the unified one.
//
// The passes are strictly ordered, never interleaved, because they share L1: the two halves'
// circular buffers are overlaid on one arena and their semaphores are drawn from one 16-id
// budget. That ordering is also what makes each half see the whole grid, exactly as it does when
// the two ops are dispatched back to back today.

#pragma once

#include <cstdint>

// The host names both halves' bases; a union binary reaching here without them would silently
// rebase to zero and read the other half's arguments.
#if defined(HYB_MERGED) && (!defined(HYB_UNIFIED_CT_BASE) || !defined(HYB_UNIFIED_RT_BASE))
#error "union kernels need -DHYB_UNIFIED_CT_BASE and -DHYB_UNIFIED_RT_BASE from the program factory"
#endif
