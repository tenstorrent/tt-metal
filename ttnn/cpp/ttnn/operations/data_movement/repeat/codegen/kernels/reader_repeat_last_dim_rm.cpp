// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reader for last-dim (width) repeat on RM interleaved tensors.
//
// Each input stick has stick_size real bytes. The output stick is
// stick_size * NUM_REPEATS bytes — formed by replicating the input stick
// NUM_REPEATS times *contiguously* in L1 (the within-stick bytes must be packed
// so the materialized output row is correct), then pushing the full output page
// to the CB for the writer.
//
// 64B page-alignment fix (matches ops/expand's proven RM path):
//   - The DRAM read copies in_read_size bytes (the aligned input page; stays
//     within the page) so the DRAM->L1 transfer is 64B-aligned. Only the first
//     stick_size bytes are real and get replicated.
//   - Each output page occupies its own out_l1_stride (== aligned output page)
//     slot, so the next output page begins at an aligned slot boundary and the
//     writer can move the whole aligned page without over-reading.
// The within-stick replication is an L1-local copy at stick_size granularity
// (NOC L1->L1 for >=16B, RISC memcpy for sub-16B), which has no DRAM NOC
// alignment constraint. Sim tolerated the old sub-64B DRAM transfers implicitly;
// silicon requires the aligned transfers above (silicon PCC 0.28 pre-fix).
//
// CT args: stick_size, in_read_size, out_l1_stride, TensorAccessorArgs(in_t),
//          cb_id, NUM_REPEATS, BATCH
// RT args: src_addr, num_pages, start_page
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t num_pages = get_arg_val<uint32_t>(1);
    uint32_t start_page = get_arg_val<uint32_t>(2);

    constexpr uint32_t stick_size = get_compile_time_arg_val(0);
    constexpr uint32_t in_read_size = get_compile_time_arg_val(1);
    constexpr uint32_t out_l1_stride = get_compile_time_arg_val(2);
    constexpr auto src_args = TensorAccessorArgs<3>();
    constexpr uint32_t cb_id = get_compile_time_arg_val(src_args.next_compile_time_args_offset());
    constexpr uint32_t NUM_REPEATS = get_compile_time_arg_val(src_args.next_compile_time_args_offset() + 1);
    constexpr uint32_t BATCH = get_compile_time_arg_val(src_args.next_compile_time_args_offset() + 2);

    const auto s = TensorAccessor(src_args, src_addr);

    uint32_t src_page = start_page;
    uint32_t pages_left = num_pages;

    while (pages_left > 0) {
        uint32_t batch = (pages_left < BATCH) ? pages_left : BATCH;
        cb_reserve_back(cb_id, batch);
        uint32_t l1_base = get_write_ptr(cb_id);

        for (uint32_t t = 0; t < batch; t++) {
            uint32_t l1_addr = l1_base + t * out_l1_stride;

            // Read the aligned input page into the first position of this slot.
            uint64_t noc_addr = s.get_noc_addr(src_page);
            noc_async_read(noc_addr, l1_addr, in_read_size);
            src_page++;
        }

        // Wait for all DRAM reads to complete before L1-to-L1 copies.
        noc_async_read_barrier();

        // Replicate each stick's real bytes NUM_REPEATS-1 more times within slot.
        if constexpr (NUM_REPEATS > 1) {
            if constexpr (stick_size >= 16 && stick_size % 16 == 0) {
                // Large 16B-aligned sticks: NOC L1->L1 DMA. Both size and the
                // dest offset (l1_addr + r*stick_size, l1_addr already 16B-aligned)
                // must be 16B-aligned for the NOC copy — hence the % 16 gate.
                for (uint32_t t = 0; t < batch; t++) {
                    uint32_t l1_addr = l1_base + t * out_l1_stride;
                    uint64_t src_noc = get_noc_addr(l1_addr);
                    for (uint32_t r = 1; r < NUM_REPEATS; r++) {
                        noc_async_read(src_noc, l1_addr + r * stick_size, stick_size);
                    }
                }
                noc_async_read_barrier();
            } else {
                // Sub-16B OR non-16B-aligned sticks (e.g. W=12 -> 24B): NOC L1->L1
                // needs 16B-aligned size+dest, so a non-aligned stick lands the
                // copy at a misaligned dest and corrupts. Use RISC copies, with
                // word-sized specializations for common W=1/sub-16B sticks.
                for (uint32_t t = 0; t < batch; t++) {
                    uint32_t l1_addr = l1_base + t * out_l1_stride;
                    if constexpr (stick_size == 2) {
                        uint16_t v = *reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1_addr);
                        for (uint32_t r = 1; r < NUM_REPEATS; r++) {
                            *reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1_addr + r * stick_size) = v;
                        }
                    } else if constexpr (stick_size == 4) {
                        uint32_t v = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_addr);
                        for (uint32_t r = 1; r < NUM_REPEATS; r++) {
                            *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_addr + r * stick_size) = v;
                        }
                    } else if constexpr (stick_size == 8) {
                        uint64_t v = *reinterpret_cast<volatile tt_l1_ptr uint64_t*>(l1_addr);
                        for (uint32_t r = 1; r < NUM_REPEATS; r++) {
                            *reinterpret_cast<volatile tt_l1_ptr uint64_t*>(l1_addr + r * stick_size) = v;
                        }
                    } else if constexpr (stick_size == 12) {
                        volatile tt_l1_ptr uint32_t* src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_addr);
                        uint32_t v0 = src[0];
                        uint32_t v1 = src[1];
                        uint32_t v2 = src[2];
                        for (uint32_t r = 1; r < NUM_REPEATS; r++) {
                            volatile tt_l1_ptr uint32_t* dst =
                                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_addr + r * stick_size);
                            dst[0] = v0;
                            dst[1] = v1;
                            dst[2] = v2;
                        }
                    } else {
                        volatile tt_l1_ptr uint8_t* src = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(l1_addr);
                        for (uint32_t r = 1; r < NUM_REPEATS; r++) {
                            volatile tt_l1_ptr uint8_t* dst =
                                reinterpret_cast<volatile tt_l1_ptr uint8_t*>(l1_addr + r * stick_size);
                            for (uint32_t b = 0; b < stick_size; b++) {
                                dst[b] = src[b];
                            }
                        }
                    }
                }
            }
        }

        cb_push_back(cb_id, batch);
        pages_left -= batch;
    }
}
