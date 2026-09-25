// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Union compute kernel: both implementations in ONE binary on the TRISC triple.
//
// This is the two-op routed-expert forward folded into one dispatch: the unified half runs every
// expert above the model's measured token threshold, then the fused half runs the rest -- the same
// work on the same full core grid, but as one program so the layer can be overlapped with combine.
//
// Unified first because combine, overlapped, still has the last expert released to do after the
// routed expert ends; ending on the small, fused experts keeps that tail short. For the routed
// expert alone the order is a wash: every expert is claimed by exactly one half, so the passes
// write disjoint output rows.
//
// What a swap DOES have to carry with it is the once-per-kernel hardware startup, which belongs
// to whichever half runs first -- hybrid_compute.cpp owns it for that reason. Bound to a half
// instead of to a position, a reorder leaves UNPACK/MATH/PACK unconfigured and wedges the grid
// with no diagnostic.
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
    // The one-time hardware startup, for the half called first and nowhere else. It is MMIO
    // against UNPACK/MATH/PACK and requires them idle, so the header allows exactly one call,
    // ahead of every other compute API call. Keeping it next to the call order is the point: a
    // half that starts the hardware itself would tie the startup to its identity rather than to
    // its position, and reordering the passes would then leave the units unconfigured -- which
    // hangs the grid with no diagnostic.
    hyb_unified::hyb_hw_startup();
    // First pass: every expert above the threshold, on the whole grid.
    hyb_unified::kernel_main();
#ifdef HYB_RUN_FUSED_PASS
    // Both halves' buffers and semaphores share this core's L1, so the second pass cannot start anywhere
    // until the first has finished everywhere.
    hybrid_pass_barrier();
    // Second pass: the rest.
    hyb_fused::kernel_main();
#endif
}
