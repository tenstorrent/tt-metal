// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// plus_one worker: one data-movement kernel of `worker_threads` threads per cluster; each thread is its own
// reader, incrementer and writer (op_design.md, Regime A `segment_split_in_place`).
//
// Raw NoC API instead of a kernel-library helper (op_design.md, "Helpers considered and rejected"): the
// eltwise helpers are compute-kernel (NEO) helpers over DataflowBuffer tiles, and this op has neither a NEO
// kernel nor a ring (a DM kernel looping a ring through itself is refused on Gen2; an integer pass through
// unpack/DEST/pack is not proven exact, rule R1). The increment runs on the hart's own integer ALU over words
// already in its L1 scratch region; no helper offers that.
//
// Thread t of this cluster takes the cluster's units unit_start + t, + t + T, ... (T = worker_threads, the
// plan's count, never get_num_threads(): a thread that did not run leaves its units unincremented). Per block
// of up to block_units of its units:
//   load_block       tagged reads (trid 0), one per unit, into this thread's line-aligned region; ONE barrier
//   increment_block  ONE scoped_lock over the block (L2 invalidate on acquire, flush on release), +1 as uint32
//   store_block      the same bytes written back to the same page offsets (trid 0); ONE barrier
// Every transfer is tagged (user trid 0) and the tag is handed back as 0 at the end.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/kernel_thread_globals.h"
#include "api/scratchpad.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/tensor_accessor.h"
#include "experimental/kernel_args.h"

namespace {
constexpr uint32_t kElemBytes = sizeof(uint32_t);
constexpr uint32_t kInt32Max = 0x7FFFFFFFu;

// One unit of work: elements [off, off + n) of page `row` (the last segment of a row may be short).
struct Segment {
    uint32_t row;
    uint32_t off;
    uint32_t n;
};
}  // namespace

void kernel_main() {
    constexpr uint32_t T = get_arg(args::worker_threads);
    constexpr bool skip_negative = get_arg(args::skip_negative_entries) != 0;
    constexpr bool tag_threads = get_arg(args::tag_threads) != 0;
    constexpr uint32_t line_bytes = get_arg(args::cache_line_bytes);

    const uint32_t unit_start = get_arg(args::unit_start);
    const uint32_t unit_count = get_arg(args::unit_count);
    const uint32_t row_elems = get_arg(args::row_elems);
    const uint32_t seg_elems = get_arg(args::seg_elems);
    const uint32_t segs_per_row = get_arg(args::segs_per_row);
    const uint32_t slot_stride_bytes = get_arg(args::slot_stride_bytes);
    const uint32_t block_units = get_arg(args::block_units);
    const uint32_t region_stride_bytes = get_arg(args::region_stride_bytes);
    const uint32_t scratch_depth = get_arg(args::scratch_depth);

    const uint32_t tid = get_my_thread_id();
    // The probe makes thread t add 1 + t, so a test can see which thread incremented which element.
    const uint32_t increment = tag_threads ? 1u + tid : 1u;
    constexpr NocOptVals user_txn = {.trid = 0};

    Noc noc;
    Scratchpad<uint32_t> pad(scratch::scratch_units);
    const auto acc = TensorAccessor(tensor::t);

    // This thread's region: line-aligned by address, disjoint from every other thread's.
    const uint32_t pad_base = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&pad[0]));
    const uint32_t first_region = ((pad_base + line_bytes - 1) / line_bytes) * line_bytes - pad_base;
    const uint32_t region_offset = first_region + tid * region_stride_bytes;
    const uint32_t block_stride_bytes = block_units * slot_stride_bytes;

    // The unit -> segment mapping, defined once (host twin: SplitPlan.unit_extent).
    auto segment_of = [&](uint32_t unit) -> Segment {
        const uint32_t row = unit / segs_per_row;
        const uint32_t off = (unit % segs_per_row) * seg_elems;
        return {row, off, (row_elems - off) < seg_elems ? (row_elems - off) : seg_elems};
    };

    // Units of this thread: unit_start + tid + k*T for k < my_units.
    const uint32_t my_units = tid < unit_count ? (unit_count - tid + T - 1) / T : 0;

    uint32_t k = 0;
    for (uint32_t block_idx = 0; k < my_units; ++block_idx) {
        const uint32_t n_units = (my_units - k) < block_units ? (my_units - k) : block_units;
        const uint32_t block_offset = region_offset + (block_idx % scratch_depth) * block_stride_bytes;

        // load_block: one tagged read per unit, exactly the unit's bytes; one barrier for the block.
        for (uint32_t s = 0; s < n_units; ++s) {
            const Segment seg = segment_of(unit_start + tid + (k + s) * T);
            noc.async_read<NocOptions::TXN_ID>(
                acc,
                pad,
                seg.n * kElemBytes,
                {.page_id = seg.row, .offset_bytes = seg.off * kElemBytes},
                {.offset_bytes = block_offset + s * slot_stride_bytes},
                user_txn);
        }
        noc.async_read_barrier<NocOptions::TXN_ID>(user_txn);

        // increment_block: one lock over the whole block; uint32 arithmetic wraps (R1).
        {
            auto lock = pad.scoped_lock(block_offset / kElemBytes, n_units * slot_stride_bytes / kElemBytes);
            for (uint32_t s = 0; s < n_units; ++s) {
                const uint32_t n = segment_of(unit_start + tid + (k + s) * T).n;
                const uint32_t base = (block_offset + s * slot_stride_bytes) / kElemBytes;
                for (uint32_t i = 0; i < n; ++i) {
                    const uint32_t x = pad[base + i];
                    if constexpr (skip_negative) {
                        // The word read as int32 is in [0, INT32_MAX) iff, as uint32, it is below INT32_MAX (R2).
                        if (x < kInt32Max) {
                            pad[base + i] = x + increment;
                        }
                    } else {
                        pad[base + i] = x + increment;
                    }
                }
            }
        }  // lock released: the block is flushed to L1 before any write reads it

        // store_block: the same bytes back to the same page offsets; one barrier for the block.
        for (uint32_t s = 0; s < n_units; ++s) {
            const Segment seg = segment_of(unit_start + tid + (k + s) * T);
            noc.async_write<NocOptions::TXN_ID>(
                pad,
                acc,
                seg.n * kElemBytes,
                {.offset_bytes = block_offset + s * slot_stride_bytes},
                {.page_id = seg.row, .offset_bytes = seg.off * kElemBytes},
                user_txn);
        }
        noc.async_write_barrier<NocOptions::TXN_ID>(user_txn);

        k += n_units;
    }

    // Hand the hart's tag back as the user id 0 (already the only id used), for the programs that follow.
    noc_async_read_set_trid(0, noc.get_noc_id());
    noc_async_write_set_trid(0, noc.get_noc_id());
}
