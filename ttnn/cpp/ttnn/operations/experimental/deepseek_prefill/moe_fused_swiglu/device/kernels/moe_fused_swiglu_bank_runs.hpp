// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// moe_fused_swiglu — coalesced weight DRAM runs, shared by the reader and the writer.
//
// THE ONE definition of the op's read/write coalescing, so a change lands in one place rather than
// in the six it used to live in.
//
// SHARD_W > 0 — the DRAM ND-sharded weight stream. When a weight tensor is ND-sharded with an N
// extent of SHARD_W pages per shard, `TensorAccessor` places page (k, n) at
// `shard_in_bank * shard_volume + (n % SHARD_W)` (tensor_accessor.h `get_bank_and_offset`), so the
// pages of one shard row are PHYSICALLY CONTIGUOUS in one bank at `aligned_page_size` stride. A run
// therefore extends to the next SHARD_W boundary and issues as ONE NoC transaction.
//
// The shard HEIGHT does not enter: `page_offset_within_shard` is
// `(k % SH) * shard_strides[0] + (n % SHARD_W)` with `shard_strides[0] == SHARD_W`, so for a fixed
// k consecutive n stay contiguous whatever SH is. Only the N extent matters.
//
// SHARD_W == 0 — interleaved. Consecutive n land in DIFFERENT banks by construction, so there is no
// contiguous run to exploit and every page is its own transaction. That is correct, and slower.
// Coalescing interleaved placement means walking the stride-NUM_BANKS bank runs instead, which was
// built, measured a NET NEGATIVE over two full guard-set samples, and is deliberately not here.

#pragma once

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

#include "moe_fused_swiglu_common.hpp"

namespace moe_fused_swiglu {

template <uint32_t SHARD_W = 0>
struct WeightRuns {
    // Length of the maximal contiguous run starting at linear N index j inside [j, end).
    static FORCE_INLINE uint32_t run(uint32_t j, uint32_t end) {
        if constexpr (SHARD_W > 0) {
            const uint32_t to_shard_edge = SHARD_W - (j % SHARD_W);
            const uint32_t r = end - j;
            return (to_shard_edge < r) ? to_shard_edge : r;
        } else {
            return 1;
        }
    }

    // Read pages [j0, jend) of the N axis at tensor row `page_row_base` into L1 at
    // `l1_base + (j - j0) * page_bytes`, as maximal contiguous runs. Reads are ISSUED only — the
    // caller owns the barrier so several of these can be batched behind one.
    template <class Acc>
    static FORCE_INLINE void read(
        const Acc& acc, uint32_t page_row_base, uint32_t j0, uint32_t jend, uint32_t l1_base, uint32_t page_bytes) {
        uint32_t j = j0;
        uint32_t off = 0;
        while (j < jend) {
            const uint32_t len = run(j, jend);
            noc_async_read(acc.get_noc_addr(page_row_base + j), l1_base + off * page_bytes, len * page_bytes);
            j += len;
            off += len;
        }
    }

    // The write-back twin. Same run construction, same caller-owns-the-barrier contract.
    template <class Acc>
    static FORCE_INLINE void write(
        const Acc& acc, uint32_t page_row_base, uint32_t j0, uint32_t jend, uint32_t l1_base, uint32_t page_bytes) {
        uint32_t j = j0;
        uint32_t off = 0;
        while (j < jend) {
            const uint32_t len = run(j, jend);
            noc_async_write(l1_base + off * page_bytes, acc.get_noc_addr(page_row_base + j), len * page_bytes);
            j += len;
            off += len;
        }
    }
};

}  // namespace moe_fused_swiglu
