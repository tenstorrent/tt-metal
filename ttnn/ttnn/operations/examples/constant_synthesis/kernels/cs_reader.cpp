// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// constant_synthesis reader (NCRISC / NoC0).
//
// Produces the source bytes for a constant-valued output. Which is the whole
// point: a constant-valued output needs NO source bytes — you can invent it
// on-core instead of moving it across the NoC. This kernel implements BOTH
// strategies, selected by one compile-time flag (`synthesize`); the writer that
// fans pages out to DRAM issues the identical NoC write pattern for both (see
// cs_writer.cpp), so any measured delta is attributable purely to whether the
// source bytes are READ from DRAM or INVENTED on-core.
//
//   synthesize == 0  (stream_from_dram, baseline): stream every output page from
//       a DRAM-resident constant tensor — `block` async reads in flight per
//       barrier (double-buffered so reads overlap the writer's writes). Real
//       DRAM read traffic: the read half of the roofline paid in full for bytes
//       that are all identical.
//
//   synthesize == 1  (synthesize, candidate): build ONE output page of the
//       constant in L1, once (a handful of local word stores, zero DRAM reads),
//       and hand that resident template to the writer to replicate to every
//       page. Zero source bytes read.

#include "api/dataflow/dataflow_api.h"

constexpr uint32_t cb_data = 0;  // reader -> writer: output pages of the constant

void kernel_main() {
    constexpr uint32_t page_bytes = get_compile_time_arg_val(0);  // one output page (W * elem_bytes)
    constexpr uint32_t value_lo = get_compile_time_arg_val(1);    // bf16 bit pattern of the constant
    constexpr uint32_t synthesize = get_compile_time_arg_val(2);  // 1 = invent on-core, 0 = read from DRAM
    constexpr uint32_t kernel_iters = get_compile_time_arg_val(3);
    constexpr uint32_t block = get_compile_time_arg_val(4);  // reads in flight per barrier (baseline)
    constexpr auto src_args = TensorAccessorArgs<5>();

    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_row = get_arg_val<uint32_t>(1);
    const uint32_t num_rows = get_arg_val<uint32_t>(2);

    [[maybe_unused]] const auto src_acc = TensorAccessor(src_args, src_addr, page_bytes);

    for (uint32_t it = 0; it < kernel_iters; ++it) {
        if constexpr (synthesize) {
            // Invent the constant on-core: build ONE output page in L1, once. Zero DRAM reads.
            // Two bf16 lanes fit one 32-bit word, so the bulk is word-replicated stores; a
            // trailing bf16 (odd W) is written as a single element.
            cb_reserve_back(cb_data, 1);
            const uint32_t l1 = get_write_ptr(cb_data);
            const uint32_t val4 = (value_lo << 16) | value_lo;
            volatile tt_l1_ptr uint32_t* wp = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1);
            const uint32_t n_words = page_bytes >> 2;
            for (uint32_t i = 0; i < n_words; ++i) {
                wp[i] = val4;
            }
            if (page_bytes & 0x2) {  // one trailing bf16 element
                reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1)[n_words << 1] = static_cast<uint16_t>(value_lo);
            }
            cb_push_back(cb_data, 1);
        } else {
            // Move the constant the obvious way: stream every output page from the
            // DRAM-resident constant tensor, `block` reads in flight per barrier.
            uint32_t p = 0;
            while (p < num_rows) {
                const uint32_t b = (num_rows - p) < block ? (num_rows - p) : block;
                cb_reserve_back(cb_data, b);
                const uint32_t l1 = get_write_ptr(cb_data);
                for (uint32_t i = 0; i < b; ++i) {
                    noc_async_read(src_acc.get_noc_addr(start_row + p + i), l1 + i * page_bytes, page_bytes);
                }
                noc_async_read_barrier();  // ONE barrier for `b` reads -> up to `block` in flight
                cb_push_back(cb_data, b);
                p += b;
            }
        }
    }
}
