// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// transfer_alignment reader (NCRISC / NoC0).
//
// Extracts one `span_bytes` sub-page span from each source row. A noc_async_read
// can only move a range when the SOURCE byte address and the DESTINATION byte
// address are congruent modulo the alignment window (`align`). The whole point of
// this kernel is what happens when the span start is NOT congruent — a case that
// cannot be expressed as a single transfer.
//
// One compile-time flag (`aligned`) selects the strategy; everything else (the row
// loop, the output CB, the congruence-landing offset) is identical, so any measured
// delta is attributable purely to the read strategy.
//
//   aligned == 1  (candidate, the dodge): the span start `s_aligned` has residue 0
//       and the destination landing is placed at residue 0, so (src % align) ==
//       (dst % align). ONE direct read of exactly `span_bytes` — no scratch, no
//       realign.
//
//   aligned == 0  (baseline, the trap): the span start `s_misaligned` has a non-zero
//       residue, so it is not congruent with the residue-0 destination and cannot be
//       read directly. Round the source DOWN to the alignment boundary, over-read
//       `span_bytes + residue` bytes into an aligned scratch CB, then do a local L1
//       realign pass moving the useful `span_bytes` into the destination. An over-read
//       plus one extra per-span L1 pass — the hidden alignment-residue tax.
//
// The useful span always lands at a residue-0 offset inside the reader->writer CB, so
// the writer's span->DRAM write is congruent and byte-identical for both variants.

#include "api/dataflow/dataflow_api.h"

constexpr uint32_t cb_scratch = 0;  // reader-internal aligned over-read scratch (misaligned path)
constexpr uint32_t cb_span = 16;    // extracted span -> writer

// Offset that lifts `base` to the next alignment boundary, so (base + off) has
// residue 0. `align` is a power of two (queried alignment window).
static inline uint32_t landing_offset(uint32_t base, uint32_t align) {
    const uint32_t r = base & (align - 1);
    return r == 0 ? 0u : (align - r);
}

void kernel_main() {
    constexpr uint32_t align = get_compile_time_arg_val(0);           // congruence window (bytes)
    constexpr uint32_t span_bytes = get_compile_time_arg_val(1);      // useful span payload
    constexpr uint32_t s_aligned = get_compile_time_arg_val(2);       // congruent span byte offset (residue 0)
    constexpr uint32_t s_misaligned = get_compile_time_arg_val(3);    // non-congruent span byte offset
    constexpr uint32_t residue = get_compile_time_arg_val(4);         // s_misaligned % align (non-zero)
    constexpr uint32_t row_page_bytes = get_compile_time_arg_val(5);  // whole source row page
    constexpr uint32_t kernel_iters = get_compile_time_arg_val(6);
    constexpr uint32_t aligned = get_compile_time_arg_val(7);  // 1 = aligned, 0 = misaligned
    constexpr auto src_args = TensorAccessorArgs<8>();

    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_row = get_arg_val<uint32_t>(1);
    const uint32_t num_rows = get_arg_val<uint32_t>(2);

    const auto src_acc = TensorAccessor(src_args, src_addr, row_page_bytes);

    for (uint32_t it = 0; it < kernel_iters; ++it) {
        for (uint32_t r = 0; r < num_rows; ++r) {
            const uint32_t row = start_row + r;
            const uint64_t page_base = src_acc.get_noc_addr(row);

            cb_reserve_back(cb_span, 1);
            const uint32_t dst_base = get_write_ptr(cb_span);
            const uint32_t dst = dst_base + landing_offset(dst_base, align);  // residue-0 landing

            if constexpr (aligned) {
                // Congruent: (src % align) == (dst % align) == 0. One direct read.
                noc_async_read(page_base + s_aligned, dst, span_bytes);
                noc_async_read_barrier();
            } else {
                // Non-congruent span start. Round the source down to the alignment
                // boundary and over-read span_bytes + residue into an aligned scratch,
                // then realign the useful span into the residue-0 destination.
                cb_reserve_back(cb_scratch, 1);
                const uint32_t sc_base = get_write_ptr(cb_scratch);
                const uint32_t sc = sc_base + landing_offset(sc_base, align);  // residue-0 scratch landing
                noc_async_read(page_base + (s_misaligned - residue), sc, span_bytes + residue);
                noc_async_read_barrier();
                cb_push_back(cb_scratch, 1);
                cb_wait_front(cb_scratch, 1);
                // Local realign pass: the useful bytes start at +residue inside the aligned
                // scratch; move them into the residue-0 destination. Source and destination
                // residues differ, so this cannot be a NoC copy — it is a CPU memmove over
                // the L1 SRAM, the extra per-span pass the misaligned read forces.
                invalidate_l1_cache();  // the NoC just wrote scratch; drop stale data-cache lines
                memmove(reinterpret_cast<void*>(dst), reinterpret_cast<void*>(sc + residue), span_bytes);
                cb_pop_front(cb_scratch, 1);
            }

            cb_push_back(cb_span, 1);
        }
    }
}
