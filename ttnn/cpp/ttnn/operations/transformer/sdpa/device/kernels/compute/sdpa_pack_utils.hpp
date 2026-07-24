// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Shared code-size helpers for SDPA compute kernels.

#pragma once

#include <cstdint>

#include "api/compute/compute_kernel_api.h"

// --- Outlined out-of-order pack (code-size) ---
// pack_tile<true>() (absolute-address pack) inlines the full
// llk_pack -> program_packer_destination GPR->FLOP address-programming sequence at every
// call site. That inlined sequence is the single largest contributor to the PACK-thread
// (TRISC2) text and overflows the TENSIX kernel-config buffer under watcher on Wormhole.
// Outlining to one noinline copy trades a jal/ret per pack for a large code-size reduction.
//
// The jal/ret is NOT free on the hot inner loop. Empirically (wan2_2 BH perf check), the
// softmax-exp pack in sub_exp_block_bcast_cols is the only perf-critical site: outlining it
// alone caused the full regression (~70% -> ~66% math util), and re-inlining only it fully
// recovers perf. So we keep that one site inlined via pack_tile<true>() directly and outline
// everything else (output/SV drain, salad correction/sum, mask L1-accumulate) through this
// wrapper. That keeps almost all of the Wormhole code-size win (only the exp pack's ~few
// static copies return) with no measurable BH perf cost. On MATH/UNPACK threads pack_tile is
// a no-op, so the outlined wrapper collapses to an empty inline function (zero overhead).
#ifdef TRISC_PACK
__attribute__((noinline, noclone)) static void sdpa_pack_tile_ooo(uint32_t dst, uint32_t cb, uint32_t idx) {
    llk_pack<DST_ACCUM_MODE, true, PackMode::Default>(dst, cb, idx);
}
#else
ALWI void sdpa_pack_tile_ooo(uint32_t, uint32_t, uint32_t) {}
#endif
