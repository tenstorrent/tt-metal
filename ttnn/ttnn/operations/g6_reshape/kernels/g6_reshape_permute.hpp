// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The permute block operation shared by every g6_reshape regime: execute a move plan of copy entries
// {src L1 addr, (image offset << 8) | bytes} from staging into the output image with RISC-V copies.

#pragma once

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "ckernel.h"

namespace g6_reshape {

// kFixedWords > 0: every entry moves exactly kFixedWords 32-bit words (word-aligned by construction).
// kFixedWords == 0: each entry carries its byte count; 32-bit copies when src, dst and length allow, else
// 16-bit copies (kElem == 2 only).
template <uint32_t kElem, uint32_t kFixedWords>
FORCE_INLINE void permute_block(
    volatile tt_l1_ptr uint32_t* copies, uint32_t num_copies, uint32_t image_base, uint8_t noc) {
    // The previous block's writes still source the image; staging lines the RISC-V cached for an
    // earlier block are stale after the NoC refilled them.
    noc_async_writes_flushed(noc);
    invalidate_l1_cache();
    uint32_t last_word = image_base;
    for (uint32_t i = 0; i < num_copies; ++i) {
        const uint32_t src = copies[2 * i];
        const uint32_t packed = copies[2 * i + 1];
        const uint32_t dst = image_base + (packed >> 8);
        const uint32_t bytes = packed & 0xFF;
        if (kFixedWords > 0 || kElem == 4 || ((src | dst | bytes) & 3) == 0) {
            volatile tt_l1_ptr uint32_t* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src);
            volatile tt_l1_ptr uint32_t* d = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst);
            const uint32_t words = kFixedWords > 0 ? kFixedWords : bytes / 4;
#pragma GCC unroll 16
            for (uint32_t w = 0; w < words; ++w) {
                d[w] = s[w];
            }
        } else {
            volatile tt_l1_ptr uint16_t* s = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(src);
            volatile tt_l1_ptr uint16_t* d = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(dst);
            for (uint32_t w = 0; w < bytes / 2; ++w) {
                d[w] = s[w];
            }
        }
        last_word = (dst + bytes - 1) & ~3u;
    }
    // CPU stores can retire before they land in L1; the NoC reads the image as a separate client.
    (void)ckernel::load_blocking(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(last_word));
}

}  // namespace g6_reshape
