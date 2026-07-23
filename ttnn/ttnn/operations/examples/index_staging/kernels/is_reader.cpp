// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// index_staging reader (NCRISC / NoC0).
//
// Computes out[w] = src[idx[w]] for one row: an index-driven access where the
// index list is arbitrary. Each selected element is a single bfloat16 value
// (ELEM_BYTES = 2), but the DRAM read granularity is a whole aligned line
// (ALIGN_BYTES = 32 = ELEMS_PER_LINE elements) — you cannot read 2 bytes from an
// arbitrary DRAM offset. That mismatch is the whole point.
//
// This kernel implements BOTH competing access strategies, selected by one
// compile-time flag (`staged`); everything else (index staging, the local
// extract loop, the output CB, the row loop) is byte-identical between them, so
// any measured delta is attributable purely to the access strategy.
//
//   staged == 0  (remote_per_index, baseline): for each of the W indices, issue
//       a SEPARATE remote NoC read of the whole aligned 32-byte DRAM line that
//       *contains* element idx[w], then extract the 2 bytes actually wanted. W
//       remote reads that move 16x the needed bytes — the classic "waste a whole
//       line for one element". Reads are pipelined (all issued, one barrier), so
//       this is the strongest honest baseline: the penalty is transaction count
//       and wasted bandwidth, not read-latency serialization.
//
//   staged == 1  (l1_staged, candidate): issue ONE bulk contiguous read of the
//       whole source row into an L1 CB (each byte fetched once), then extract
//       every element locally in SRAM. One remote transaction of exactly the
//       useful bytes instead of W transactions of 16x the useful bytes.
//
// Both variants run the identical W-element local extract loop, so that cost is
// common and never biases the comparison.

#include "api/dataflow/dataflow_api.h"

constexpr uint32_t cb_idx = 0;   // index row staged in L1 (reader-internal)
constexpr uint32_t cb_line = 1;  // baseline: W aligned scratch lines
constexpr uint32_t cb_src = 2;   // candidate: bulk source row
constexpr uint32_t cb_out = 16;  // selected output row -> writer

void kernel_main() {
    constexpr uint32_t src_page_bytes = get_compile_time_arg_val(0);  // whole source row (W * elem_bytes)
    constexpr uint32_t idx_page_bytes = get_compile_time_arg_val(1);  // index row page
    constexpr uint32_t elem_bytes = get_compile_time_arg_val(2);      // select unit (2 B = one bfloat16)
    constexpr uint32_t align_bytes = get_compile_time_arg_val(3);     // DRAM read granularity (32 B line)
    constexpr uint32_t W = get_compile_time_arg_val(4);               // number of indices per row
    constexpr uint32_t kernel_iters = get_compile_time_arg_val(5);
    constexpr uint32_t staged = get_compile_time_arg_val(6);  // 1 = l1_staged, 0 = remote_per_index
    constexpr auto src_args = TensorAccessorArgs<7>();
    constexpr auto idx_args = TensorAccessorArgs<src_args.next_compile_time_args_offset()>();

    constexpr uint32_t elems_per_line = align_bytes / elem_bytes;  // 16 bfloat16 per 32-byte line

    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t idx_addr = get_arg_val<uint32_t>(1);
    const uint32_t start_row = get_arg_val<uint32_t>(2);
    const uint32_t num_rows = get_arg_val<uint32_t>(3);

    const auto src_acc = TensorAccessor(src_args, src_addr, src_page_bytes);
    const auto idx_acc = TensorAccessor(idx_args, idx_addr, idx_page_bytes);

    for (uint32_t it = 0; it < kernel_iters; ++it) {
        for (uint32_t r = 0; r < num_rows; ++r) {
            const uint32_t row = start_row + r;

            // Stage the index row into L1 (identical for both variants).
            cb_reserve_back(cb_idx, 1);
            const uint32_t idx_l1 = get_write_ptr(cb_idx);
            noc_async_read(idx_acc.get_noc_addr(row), idx_l1, idx_page_bytes);
            noc_async_read_barrier();
            cb_push_back(cb_idx, 1);
            cb_wait_front(cb_idx, 1);
            volatile tt_l1_ptr uint32_t* idxp = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_idx));

            cb_reserve_back(cb_out, 1);
            volatile tt_l1_ptr uint16_t* outp = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(cb_out));
            const uint64_t src_base = src_acc.get_noc_addr(row);

            if constexpr (staged) {
                // ONE bulk read of the whole source row, then L1-local extract.
                cb_reserve_back(cb_src, 1);
                const uint32_t src_l1 = get_write_ptr(cb_src);
                noc_async_read(src_base, src_l1, src_page_bytes);
                noc_async_read_barrier();
                cb_push_back(cb_src, 1);
                cb_wait_front(cb_src, 1);
                volatile tt_l1_ptr uint16_t* srcp =
                    reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_read_ptr(cb_src));
                for (uint32_t w = 0; w < W; ++w) {
                    outp[w] = srcp[idxp[w]];  // element idx[w], SRAM-local
                }
                cb_pop_front(cb_src, 1);
            } else {
                // One remote read per index: each pulls the whole aligned line
                // containing element idx[w] (16x the needed bytes). Pipelined:
                // all W reads issued into scratch, then a single barrier.
                cb_reserve_back(cb_line, 1);
                const uint32_t line_l1 = get_write_ptr(cb_line);
                for (uint32_t w = 0; w < W; ++w) {
                    const uint32_t line = idxp[w] / elems_per_line;  // which aligned line
                    noc_async_read(src_base + line * align_bytes, line_l1 + w * align_bytes, align_bytes);
                }
                noc_async_read_barrier();
                cb_push_back(cb_line, 1);
                cb_wait_front(cb_line, 1);
                volatile tt_l1_ptr uint16_t* linep =
                    reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_read_ptr(cb_line));
                for (uint32_t w = 0; w < W; ++w) {
                    outp[w] = linep[w * elems_per_line + (idxp[w] % elems_per_line)];  // extract the one wanted element
                }
                cb_pop_front(cb_line, 1);
            }

            cb_push_back(cb_out, 1);
            cb_pop_front(cb_idx, 1);
        }
    }
}
