// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The permute block operation shared by every g6_reshape regime: RISC-V copies from staging into the output
// image. Two realizations share the copy primitive and the image fences below:
//  - permute_block: execute a move plan of copy entries {src L1 addr, (image offset << 8) | bytes} recorded by
//    a gather walk (tile regimes, rm bank fold);
//  - plan-free (rm linear fold, in g6_reshape_rm_mover.cpp): the copy sources are closed-form in the staged
//    stream range, so the copies run straight off the fold cursor with no plan written or read back.

#pragma once

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "ckernel.h"

namespace g6_reshape {

// Before the CPU writes a block's image: the previous block's writes may still source the image, and staging
// lines the RISC-V cached for an earlier block are stale after the NoC refilled them.
FORCE_INLINE void begin_image(uint8_t noc) {
    noc_async_writes_flushed(noc);
    invalidate_l1_cache();
}

// Before the NoC sends image bytes the CPU stored: CPU stores can retire before they land in L1, and the NoC
// reads the image as a separate client. `last_byte` is the last image byte written.
FORCE_INLINE void fence_image(uint32_t last_byte) {
    (void)ckernel::load_blocking(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(last_byte & ~3u));
}

// Copy `bytes` from L1 `src` to L1 `dst`. kFixedWords > 0: exactly kFixedWords 32-bit words (word-aligned by
// construction). kFixedWords == 0: 32-bit copies when src, dst and length allow, else 16-bit copies
// (kElem == 2 only).
template <uint32_t kElem, uint32_t kFixedWords = 0>
FORCE_INLINE void copy_run(uint32_t src, uint32_t dst, uint32_t bytes) {
    if (kFixedWords > 0 || kElem == 4 || ((src | dst | bytes) & 3) == 0) {
        volatile tt_l1_ptr uint32_t* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src);
        volatile tt_l1_ptr uint32_t* d = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst);
        if constexpr (kFixedWords > 0) {
#pragma GCC unroll 16
            for (uint32_t w = 0; w < kFixedWords; ++w) {
                d[w] = s[w];
            }
        } else {
            const uint32_t words = bytes / 4;
#pragma GCC unroll 1
            for (uint32_t w = 0; w < words; ++w) {
                d[w] = s[w];
            }
        }
    } else {
        volatile tt_l1_ptr uint16_t* s = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(src);
        volatile tt_l1_ptr uint16_t* d = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(dst);
        for (uint32_t w = 0; w < bytes / 2; ++w) {
            d[w] = s[w];
        }
    }
}

template <uint32_t kElem, uint32_t kFixedWords>
FORCE_INLINE void permute_block(
    volatile tt_l1_ptr uint32_t* copies, uint32_t num_copies, uint32_t image_base, uint8_t noc) {
    begin_image(noc);
    uint32_t last_byte = image_base;
    for (uint32_t i = 0; i < num_copies; ++i) {
        const uint32_t src = copies[2 * i];
        const uint32_t packed = copies[2 * i + 1];
        const uint32_t dst = image_base + (packed >> 8);
        const uint32_t bytes = kFixedWords > 0 ? kFixedWords * 4 : packed & 0xFF;
        copy_run<kElem, kFixedWords>(src, dst, bytes);
        last_byte = dst + bytes - 1;
    }
    fence_image(last_byte);
}

}  // namespace g6_reshape
