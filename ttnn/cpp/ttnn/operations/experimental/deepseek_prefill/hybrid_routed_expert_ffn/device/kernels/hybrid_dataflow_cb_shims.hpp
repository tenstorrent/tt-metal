// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Gives the union reader and writer ONE out-of-line copy of each circular-buffer call.
//
// Same trade and same reason as hybrid_llk_shims.hpp: the whole program config is written into
// the kernel-config ring on every dispatch, the union program carries both halves' binaries, and
// text is what this op is short of. The CB API is FORCE_INLINE, which is right for a kernel with
// a few call sites and wrong for one carrying two implementations.
//
// Interception is by MACRO rather than by unqualified lookup, which is what lets it reach both
// halves: most CB calls are written in the halves' transport helpers, and those are included at
// file scope, outside the namespace a using-declaration would sit in. Include order is therefore
// the mechanism, and this header has to come FIRST:
//
//   1. it pulls in dataflow_api.h, so the four real definitions are parsed under their own names;
//   2. it defines forwarding wrappers named `hyb_<original>`;
//   3. it renames them, so every header parsed after -- circular_buffer.h, both halves and their
//      helpers -- lands on the wrapper.
//
// The handful of call sites inside dataflow_api.h itself keep the inline version, because an
// unqualified call to a non-dependent name is bound where it is written, not where it is
// instantiated. That is the price of having the definitions in scope at all.

#pragma once

// The compute kernel reaches its circular buffers through the LLK layer instead, which
// hybrid_llk_shims.hpp intercepts one level down.
#if !defined(COMPILE_FOR_TRISC)

// Under their own names, so the definitions and dataflow_api.h's own uses of them are the real ones.
#include "api/dataflow/dataflow_api.h"

// noclone as well as noinline: under -flto, IPA constant propagation will otherwise clone a
// specialization per distinct argument pair and put the copies straight back.
#define HYB_CB_SHIM __attribute__((noinline, noclone))

HYB_CB_SHIM inline void hyb_cb_push_back(const int32_t operand, const int32_t num_pages) {
    cb_push_back(operand, num_pages);
}

HYB_CB_SHIM inline void hyb_cb_pop_front(int32_t operand, int32_t num_pages) { cb_pop_front(operand, num_pages); }

HYB_CB_SHIM inline void hyb_cb_reserve_back(int32_t operand, int32_t num_pages) { cb_reserve_back(operand, num_pages); }

HYB_CB_SHIM inline void hyb_cb_wait_front(int32_t operand, int32_t num_pages) { cb_wait_front(operand, num_pages); }

#define cb_push_back hyb_cb_push_back
#define cb_pop_front hyb_cb_pop_front
#define cb_reserve_back hyb_cb_reserve_back
#define cb_wait_front hyb_cb_wait_front

#endif  // !COMPILE_FOR_TRISC
