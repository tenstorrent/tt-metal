// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Union compute kernel: both implementations in ONE binary on the TRISC triple.
//
// This is the two-op routed-expert forward folded into one dispatch: the fused half runs every
// expert at or below the model's measured token threshold, then the unified half runs the rest --
// the same work on the same full core grid, but as one program so the layer can be overlapped
// with combine.
//
// The pass order is the REVERSE of the two-op forward, which dispatches unified first, and it is
// NOT interchangeable. Results do not depend on it -- every expert is claimed by exactly one
// half, so the passes write disjoint output rows -- but the machine does: put the unified half
// first and it deadlocks at the first wrap of its gate/up credit pipeline, with the rotating in1
// senders stuck in up_go_sem.wait_min and every reader behind them in cb_reserve_back. It does so
// even with the fused pass deleted outright, and the circular-buffer credit state and the shared
// semaphore block are identical at the unified half's entry either way -- so the fused half
// leaves something else behind that the unified half needs. Until that is known, fused runs
// first, and a reordering here has to be re-validated on the x_tile path above the threshold,
// where this bites.
//
// The fused half is included first and keeps index 0 of both argument lists, so it needs no
// rebasing and its body is exactly the standalone kernel. The unified half follows at the bases
// the program factory computed, applied by the shims its wrapper pulls in. Neither body is
// renumbered; see hybrid_arg_shims.hpp for why that works.

#define HYB_MERGED 1
#include "hybrid_merged_prologue.hpp"
// FIRST, ahead of any header that pulls in the LLK pack/unpack lib: this one works by renaming
// the helpers for everything parsed after it, so what it reaches is exactly what follows it.
#include "hybrid_llk_shims.hpp"
// Declarations only -- the real CB API is not in scope until a half pulls it in.
#include "hybrid_cb_shims.hpp"

#define HYB_NS hyb_fused
#define HYB_CT_BASE 0
#define HYB_RT_BASE 0
#include "moe_fused_swiglu_compute.cpp"
#undef HYB_NS
#undef HYB_CT_BASE
#undef HYB_RT_BASE

// The namespace wrapper isolates SYMBOLS, not the preprocessor: a macro either half defines at
// file scope is still live when the other half is compiled. These are each half's PRIVATE macros
// and must not cross. BINARY_ACT_TILE is the one that bites -- the two halves define it with
// different arity ((g,u,o) vs (fp32,g,u,o)), so a leak miscompiles rather than merely warning.
//
// Host-supplied defines (FP32_DEST_ACC_EN, PACKER_L1_ACC, the activation selectors,
// UNIFIED_RE_GRID_Y) are deliberately NOT cleared: both halves are meant to see the one
// op-level choice, and clearing them would let a half silently fall back to its own default.
#undef FUSED_BINARY_ACT
#undef BINARY_ACT_INIT
#undef BINARY_ACT_TILE
#undef MaybeDeviceZoneScope
#undef CT
#undef CT_COUNT
#undef MOE_DECLARE_CT_ENUM
#undef MOE_CT_ENUMERATOR

#define HYB_NS hyb_unified
#define HYB_CT_BASE HYB_UNIFIED_CT_BASE
#define HYB_RT_BASE HYB_UNIFIED_RT_BASE
#include "compute/fused_swiglu.cpp"
#undef HYB_NS
#undef HYB_CT_BASE
#undef HYB_RT_BASE

// After both halves, so it sees the dataflow API they pulled in; compiles to nothing on a compute
// kernel, which has no NoC of its own.
// Both halves have pulled in the CB API by now, so the shims declared above can be defined.
#define HYB_CB_SHIMS_DEFINE 1
#include "hybrid_cb_shims.hpp"

#include "hybrid_pass_barrier.hpp"

void kernel_main() {
#ifdef HYB_RUN_FUSED_PASS
    // Pass A: every expert at or below the threshold, on the whole grid.
    hyb_fused::kernel_main();
    // Both halves' buffers and semaphores share this core's L1, so pass B cannot start anywhere
    // until pass A has finished everywhere.
    hybrid_pass_barrier();
#endif
    // Pass B: the rest.
    hyb_unified::kernel_main();
}
