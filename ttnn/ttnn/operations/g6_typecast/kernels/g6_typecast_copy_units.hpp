// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// g6_typecast regime R0 `pure_copy` (Perf 1): the bit-identical arms (uint32 -> int32, int32 -> uint32)
// are a byte copy of the buffer — same page size, same page order, same bytes. No compute kernel.
//
// A mover owns `count` consecutive pages [page0, page0 + count) and, of each page, the byte range
// [off, off + len). Batches: if len <= buf_bytes, up to buf_bytes / len pages per batch (each page's slice
// at i * len in L1); otherwise one page at a time in sub-chunks of buf_bytes.
// f(page, n_pages, byte_off, bytes) is called once per batch.
#pragma once

#include <cstdint>

namespace g6_copy {

template <typename F>
FORCE_INLINE void for_each_batch(
    uint32_t page0, uint32_t count, uint32_t off, uint32_t len, uint32_t buf_bytes, F&& f) {
    const uint32_t end = page0 + count;
    if (len <= buf_bytes) {
        const uint32_t k = buf_bytes / len;
        for (uint32_t p = page0; p < end; p += k) {
            const uint32_t n = (end - p) < k ? (end - p) : k;
            f(p, n, off, len);
        }
    } else {
        for (uint32_t p = page0; p < end; ++p) {
            for (uint32_t o = off; o < off + len; o += buf_bytes) {
                const uint32_t b = (off + len - o) < buf_bytes ? (off + len - o) : buf_bytes;
                f(p, 1u, o, b);
            }
        }
    }
}

}  // namespace g6_copy
