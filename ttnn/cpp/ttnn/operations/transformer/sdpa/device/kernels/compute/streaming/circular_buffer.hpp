// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Shared streaming buffer protocol, independent of numerical policy. Include
// after compute_common.hpp and the compute API, just like compute_streaming.hpp.
// These functions retain their original linkage and inlining attributes: both
// affect device code size and the hot PACK-thread path.
#include "api/dataflow/circular_buffer.h"

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

static __attribute__((noinline, noclone)) void sdpa_cb_push_back_out_of_line(uint32_t cb_id, uint32_t num_tiles) {
    CircularBuffer(cb_id).push_back(num_tiles);
}

static __attribute__((noinline, noclone)) void sdpa_cb_pop_front_out_of_line(uint32_t cb_id, uint32_t num_tiles) {
    CircularBuffer(cb_id).pop_front(num_tiles);
}

// Publish tiles to UNPACK while retaining the PACK write origin for absolute
// row offsets. The caller owns the reservation and must balance publication
// and consumption; this is not an ordinary append to a circular buffer.
ALWI void cb_push_back_hold_wr_ptr(uint32_t cb_id, uint32_t num_tiles) {
    CircularBuffer(cb_id).push_back(num_tiles);
    PACK(({
        auto& intf = get_local_cb_interface(cb_id);
        intf.fifo_wr_ptr -= num_tiles * intf.fifo_page_size;
        uint32_t fifo_start = intf.fifo_limit - intf.fifo_size;
        if (intf.fifo_wr_ptr < fifo_start) {
            intf.fifo_wr_ptr += intf.fifo_size;
        }
    }));
}
