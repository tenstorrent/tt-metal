// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Two-source K-row gather for deepseek_v4_prefill_sdpa: each key-list entry names a row of window_kv
// (WINDOW_ROW set) or of cache_kv. The reader and writer each gather one half of every K chunk on their own NoC
// into the shared cb_k_rm L1, as sparse_sdpa does. Include after dataflow_api.h and experimental_device_api.hpp.
#pragma once

#include <stdint.h>
#include "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/deepseek_v4_prefill_sdpa_common.hpp"

namespace deepseek_v4_prefill_sdpa {

constexpr uint32_t K_TRID_RING = 4;

template <uint32_t RingDepth, typename WindowAccessor, typename CacheAccessor>
FORCE_INLINE void gather_rows(
    Noc& noc,
    const WindowAccessor& window,
    const CacheAccessor& cache,
    uint32_t dst_l1,
    volatile tt_l1_ptr uint32_t* rows,
    uint32_t base,
    uint32_t lo,
    uint32_t hi,
    uint32_t row_bytes) {
    constexpr uint32_t D = RingDepth;
    const UnicastEndpoint local_l1;
    const uint32_t cnt = hi - lo;
    for (uint32_t i = 0; i < cnt; ++i) {
        const uint32_t p = lo + i;
        const uint32_t trid = (i % D) + 1;
        if (i >= D) {
            experimental::async_read_barrier_with_trid(noc, trid);  // free this trid slot before reuse
        }
        experimental::set_read_trid(noc, trid);
        const uint32_t row = rows[base + p];
        const uint32_t dst = dst_l1 + p * row_bytes;
        if (row & WINDOW_ROW) {
            noc.async_read(window, local_l1, row_bytes, {.page_id = row & ~WINDOW_ROW}, {.addr = dst});
        } else {
            noc.async_read(cache, local_l1, row_bytes, {.page_id = row}, {.addr = dst});
        }
    }
    const uint32_t to_drain = (cnt < D) ? cnt : D;
    for (uint32_t d = 0; d < to_drain; ++d) {
        experimental::async_read_barrier_with_trid(noc, ((cnt - to_drain + d) % D) + 1);
    }
    experimental::set_read_trid(noc, 0);  // restore untagged
}

}  // namespace deepseek_v4_prefill_sdpa
