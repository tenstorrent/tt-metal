// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// RAW-LLK BYPASS — experiment only, never included by the op.
//
// Helper bypassed: `compute_kernel_lib::tilize<...>` (tilize_helpers.hpp), and
// under it the compute API `ckernel::tilize_block` (api/compute/tilize.h:171).
//
// Mechanism being exploited
// -------------------------
// `tilize_block` is the REGULAR (non-fast) tilize path.  Its per-block body is
//
//     UNPACK(llk_unpack_tilize_block(icb, block, ...));      // whole block, once
//     for (t = 0; t < block; ++t) {
//         MATH(llk_math_wait_for_dest_available());
//         PACK(llk_packer_wait_for_math_done());
//         MATH(llk_math_eltwise_unary_datacopy<...>(0 /*dst index*/, icb));
//         PACK(llk_pack<...>(0 /*tile index*/, ocb, t));
//         MATH(llk_math_dest_section_done<...>());
//         PACK(llk_pack_dest_section_done<...>());
//     }
//
// i.e. ONE DEST acquire/commit/release round trip PER TILE, always on DEST slot
// 0.  In half-sync mode DEST holds 8 tiles (16-bit) / 4 tiles (32-bit acc) per
// section, so the regular path uses 1/8 (resp. 1/4) of the DEST it is allowed
// and pays a math<->pack semaphore round trip for every single tile.  The FAST
// path (`fast_tilize_block`) already does the batched thing — it fills a whole
// DEST section per acquire — which is exactly why the fast path is fast.
//
// `tilize_block_wide` below is `tilize_block` with the DEST window widened to
// the full section: N tiles are datacopied into DEST slots 0..N-1 under ONE
// acquire, then packed out under one commit.  Same unpack MOP, same datacopy
// LLK, same pack LLK, same data formats, same DST_ACCUM_MODE / DST_SYNC_MODE /
// MATH_FIDELITY — the ONLY change is how many tiles share a DEST section.
// Output is therefore bit-identical by construction (tilize is a pure byte
// permutation and no arithmetic is performed either way).
//
// Why a raw bypass rather than a helper flag: `compute_kernel_lib::tilize`
// hard-wires `tilize_block` for the non-fast path and exposes no DEST-window
// parameter (see tilize_helpers.inl:246).  There is no compute-API entry point
// for "regular tilize, N tiles per DEST section" at all.  Classified
// `capability`, not ergonomics.

#pragma once

#include "api/compute/tilize.h"
#include "ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp"

namespace tilize_ct {

using namespace ckernel;

// Regular (non-fast) tilize of one 1 x `block` tile-row, with `window` tiles per
// DEST section instead of one.  `window` must be <= the DEST section capacity
// (compute_kernel_lib::DEST_AUTO_LIMIT).
template <uint32_t window>
ALWI void tilize_block_wide(uint32_t icb, uint32_t block, uint32_t ocb) {
    static_assert(window >= 1, "window must be >= 1");

    UNPACK((llk_unpack_tilize_block(icb, block, 0 /*input_tile_index*/)));

    uint32_t done = 0;
    while (done < block) {
        const uint32_t left = block - done;
        const uint32_t n = (left < window) ? left : window;

        MATH((llk_math_wait_for_dest_available()));
        PACK((llk_packer_wait_for_math_done()));

        for (uint32_t i = 0; i < n; ++i) {
            MATH((
                llk_math_eltwise_unary_datacopy<DataCopyType::A2D, DST_ACCUM_MODE, BroadcastType::NONE, UnpackToDestEn>(
                    i /*dst index*/, icb)));
        }
        for (uint32_t i = 0; i < n; ++i) {
            PACK((llk_pack<DST_ACCUM_MODE, true /*out_of_order*/, PackMode::Default>(
                i /*dst tile index*/, ocb, done + i /*ocb tile index*/)));
        }

        MATH((llk_math_dest_section_done<DST_ACCUM_MODE>()));
        PACK((llk_pack_dest_section_done<DST_ACCUM_MODE>()));

        done += n;
    }
}

}  // namespace tilize_ct
