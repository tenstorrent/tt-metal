// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// moe_fused_swiglu — DRAM bank-run coalescing, shared by the reader and the writer.
//
// THE ONE definition of the op's read/write coalescing (it used to be four copies of the same
// while-loop plus two copies of `remap_n`/`run_len`, one pair per kernel — a `WRUN` turn had to land
// in six places to be consistent).
//
// Interleaved page -> bank is `page_id % NUM_BANKS`, with in-bank slot `page_id / NUM_BANKS` at
// stride `aligned_page_size` (dataflow_api_addrgen.h). For every tensor this op touches the N-axis
// row stride (HID_T / EMB_T) is a multiple of NUM_BANKS, so `bank(row*stride + n) == n % NUM_BANKS`:
// a stride-NUM_BANKS run of columns at a fixed row is physically contiguous inside ONE bank and
// therefore reads/writes as ONE NoC transaction. `remap()` re-indexes the logical N axis so that
// CONSECUTIVE linear indices walk one bank's slots; `run()` returns the maximal such run.
//
// REMAP == 0 (device with an unexpected bank count / alignment, or the WRUN=1 ablation) degrades to
// the naive one-transaction-per-page read, which is always correct.
//
// SHARD_W > 0 — THE DRAM-ND-SHARDED WEIGHT STREAM (MOE_SWIGLU_WSHARD). When the weight tensor is
// DRAM ND-sharded with an N extent of SHARD_W pages per shard, `TensorAccessor` places page (k, n)
// at `bank_page_offset = shard_in_bank * shard_volume + (n % SHARD_W) * 1` (tensor_accessor.h
// `get_bank_and_offset`), i.e. the pages of one shard row are PHYSICALLY CONTIGUOUS in one bank at
// `aligned_page_size` stride. So a run may extend to the next SHARD_W boundary with NO remap and no
// bank arithmetic — which is why this mode is orthogonal to REMAP and needs neither `slots` nor a
// power-of-two bank count. Interleaved placement cannot express this: there consecutive `n` land in
// DIFFERENT banks by construction.
//
// The shard HEIGHT does not enter: `page_offset_within_shard` is
// `(k % SH) * shard_strides[0] + (n % SHARD_W)` and `shard_strides[0] == SHARD_W`, so for a FIXED k
// consecutive n stay contiguous whatever SH is. Only the N extent matters, and it is the one number
// this template takes.

#pragma once

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

#include "moe_fused_swiglu_common.hpp"

namespace moe_fused_swiglu {

template <bool REMAP, uint32_t NUM_BANKS, uint32_t WRUN, uint32_t SHARD_W = 0>
struct BankRuns {
    // Logical N index -> physical page column, such that j, j+1, ... land in one bank's
    // consecutive slots. `slots` is the per-bank slot count of that axis (extent / NUM_BANKS).
    // The ND-sharded stream is already contiguous in its own N order, so it never remaps.
    static FORCE_INLINE uint32_t remap(uint32_t j, uint32_t slots) {
        if constexpr (REMAP && SHARD_W == 0) {
            return (j / slots) + NUM_BANKS * (j % slots);
        } else {
            return j;
        }
    }

    // Length of the maximal bank-contiguous run starting at linear index j inside [j, end),
    // capped by the WRUN knob.
    static FORCE_INLINE uint32_t run(uint32_t j, uint32_t end, uint32_t slots) {
        if constexpr (SHARD_W > 0) {
            // ND-sharded: the run extends to this shard's N boundary. NOT capped by WRUN — that knob
            // caps the INTERLEAVED remap's transaction size (where a long run costs NoC command count
            // for DRAM-side locality, measured a net negative at PERF 10). Here the whole slice is one
            // request with no remap and no extra commands, so there is nothing to trade off.
            const uint32_t to_shard_edge = SHARD_W - (j % SHARD_W);
            const uint32_t r = end - j;
            return (to_shard_edge < r) ? to_shard_edge : r;
        } else if constexpr (REMAP) {
            uint32_t r = end - j;
            const uint32_t to_bank_edge = slots - (j % slots);
            if (to_bank_edge < r) {
                r = to_bank_edge;
            }
            if (WRUN < r) {
                r = WRUN;
            }
            return r;
        } else {
            return 1;
        }
    }

    // Read pages [j0, jend) of the N axis at tensor row `page_row_base` into L1 at
    // `l1_base + (j - j0) * page_bytes`, as maximal bank-contiguous runs. Reads are ISSUED only —
    // the caller owns the barrier so several of these can be batched behind one.
    template <class Acc>
    static FORCE_INLINE void read(
        const Acc& acc,
        uint32_t page_row_base,
        uint32_t j0,
        uint32_t jend,
        uint32_t slots,
        uint32_t l1_base,
        uint32_t page_bytes) {
        uint32_t j = j0;
        uint32_t off = 0;
        while (j < jend) {
            const uint32_t len = run(j, jend, slots);
            noc_async_read(
                acc.get_noc_addr(page_row_base + remap(j, slots)), l1_base + off * page_bytes, len * page_bytes);
            j += len;
            off += len;
        }
    }

    // The write-back twin. Same run construction, same caller-owns-the-barrier contract.
    template <class Acc>
    static FORCE_INLINE void write(
        const Acc& acc,
        uint32_t page_row_base,
        uint32_t j0,
        uint32_t jend,
        uint32_t slots,
        uint32_t l1_base,
        uint32_t page_bytes) {
        uint32_t j = j0;
        uint32_t off = 0;
        while (j < jend) {
            const uint32_t len = run(j, jend, slots);
            noc_async_write(
                l1_base + off * page_bytes, acc.get_noc_addr(page_row_base + remap(j, slots)), len * page_bytes);
            j += len;
            off += len;
        }
    }
};

}  // namespace moe_fused_swiglu
