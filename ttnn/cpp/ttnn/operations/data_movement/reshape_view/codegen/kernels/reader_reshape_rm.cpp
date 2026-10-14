// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Row-major reshape reader. Coalesced: sticks land in the CB as read. Non-aligned: each old stick is read into an
// alignment-matched scratch slot, then repacked into new-stick CB pages with RISC copies.
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#if !defined(ARCH_QUASAR)
#include "ckernel.h"
#endif

// Re-stores a copy's final halfword after a compiler barrier, so it is the last RISC store in program order, then
// reads it back. Wormhole orders a load only behind a store with the same start address, so this read-back is what
// makes the copy visible to the NOC and the CB consumer before the push; a fence does not order a later push.
FORCE_INLINE void drain_halfword_stores(uint32_t last_addr, uint16_t value) {
    asm volatile("" ::: "memory");
    volatile tt_l1_ptr uint16_t* last = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(last_addr);
    *last = value;
#if defined(ARCH_QUASAR)
    (void)*last;
#else
    (void)ckernel::load_blocking(last);
#endif
}

struct ArgsCoalesced {
    uint32_t src_addr;
    uint32_t num_reads;
    uint32_t sticks_per_read;
    uint32_t sticks_per_cb_push;
    uint32_t start_stick;
};

struct ArgsNonaligned {
    uint32_t src_addr;
    uint32_t num_reads;
    uint32_t sticks_per_read;
    uint32_t start_stick;
    uint32_t old_stick_size;
    uint32_t new_stick_size;
    uint32_t ratio;
    uint32_t split;
    uint32_t cb_scratch_id;
};

void kernel_main() {
    constexpr uint32_t cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t stick_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t aligned_page_size = get_compile_time_arg_val(2);
    constexpr bool NONALIGNED = get_compile_time_arg_val(3) != 0;
    // Sticks the non-aligned path gathers per NOC barrier.
    constexpr uint32_t NABATCH = get_compile_time_arg_val(4);
    constexpr auto src_args = TensorAccessorArgs<5>();

    // Reads num_reads work units of sticks_per_read sticks each; the CB push count can differ from the read count.
    if constexpr (!NONALIGNED) {
        const auto* a = reinterpret_cast<const ArgsCoalesced*>(get_arg_addr(0));
        const auto s = TensorAccessor(src_args, a->src_addr, aligned_page_size);
        Noc noc;
        CircularBuffer cb(cb_id);
        uint32_t i_stick = a->start_stick;
        for (uint32_t iter = 0; iter < a->num_reads; ++iter) {
            cb.reserve_back(a->sticks_per_cb_push);
            uint32_t l1_offset = 0;
            for (uint32_t i = 0; i < a->sticks_per_read; ++i) {
                noc.async_read(
                    s, cb, stick_bytes, {.page_id = i_stick, .offset_bytes = 0}, {.offset_bytes = l1_offset});
                l1_offset += stick_bytes;
                i_stick++;
            }
            noc.async_read_barrier();
            cb.push_back(a->sticks_per_cb_push);
        }
    } else {
        const auto* a = reinterpret_cast<const ArgsNonaligned*>(get_arg_addr(0));
        const auto s = TensorAccessor(src_args, a->src_addr, aligned_page_size);

        Noc noc;
        CircularBuffer cb(cb_id);
        CircularBuffer cb_scratch(a->cb_scratch_id);

        cb_scratch.reserve_back(1);
        uint32_t scratch_addr = cb_scratch.get_write_ptr();
        uint32_t i_stick = a->start_stick;
        // A NOC read requires source and local-L1 destination addresses to have
        // equal residues modulo the architecture's read alignment (64B on BH,
        // 32B on WH). Keep one 64B alignment margin per slot and shift only the
        // L1 destination, so the source transaction stays exactly one physical
        // page and never crosses a page or allocation boundary.
        constexpr uint32_t READ_ALIGNMENT = 64;
        constexpr uint32_t READ_ALIGNMENT_MASK = READ_ALIGNMENT - 1;
        const uint32_t read_size = aligned_page_size;
        const uint32_t scratch_slot_size = aligned_page_size + READ_ALIGNMENT;

        if (a->split) {
            // old > new: each old stick produces ratio new sticks.
            uint32_t remaining = a->num_reads;
            const uint32_t new_half = a->new_stick_size / 2;
            const uint32_t cb_page_bytes = (a->new_stick_size + 15) & ~15u;  // L1-aligned
            while (remaining > 0) {
                uint32_t batch = (remaining < NABATCH) ? remaining : NABATCH;
                uint32_t byte_offsets[NABATCH];
                for (uint32_t b = 0; b < batch; ++b) {
                    uint64_t noc_addr = s.get_noc_addr(i_stick + b, 0, noc.get_noc_id());
                    uint32_t scratch_slot = scratch_addr + b * scratch_slot_size;
                    byte_offsets[b] = (noc_addr - scratch_slot) & READ_ALIGNMENT_MASK;
                    noc.async_read(
                        s,
                        cb_scratch,
                        read_size,
                        {.page_id = i_stick + b, .offset_bytes = 0},
                        {.offset_bytes = b * scratch_slot_size + byte_offsets[b]});
                }
                noc.async_read_barrier();
                invalidate_l1_cache();
                for (uint32_t b = 0; b < batch; ++b) {
                    // One reserve for all ratio pages: one semaphore check instead of ratio.
                    cb.reserve_back(a->ratio);
                    uint32_t cb_base = cb.get_write_ptr();
                    uint16_t* src_ptr =
                        reinterpret_cast<uint16_t*>(scratch_addr + b * scratch_slot_size + byte_offsets[b]);
                    // When new_stick_size is 16B-aligned (cb_page_bytes == new_stick_size), the ratio destination
                    // pages and the source are one contiguous block of ratio*new_half uint16, so it is copied as
                    // uint64 words when both ends are 8B-aligned; otherwise each page is repacked per element.
                    if (cb_page_bytes == a->new_stick_size && (reinterpret_cast<uintptr_t>(src_ptr) & 7u) == 0 &&
                        (cb_base & 7u) == 0) {
                        const uint32_t n16 = a->ratio * new_half;  // total uint16 elems
                        const uint32_t n64 = n16 >> 2;             // full uint64 words
                        uint64_t* d64 = reinterpret_cast<uint64_t*>(cb_base);
                        const uint64_t* s64 = reinterpret_cast<const uint64_t*>(src_ptr);
                        for (uint32_t w = 0; w < n64; ++w) {
                            d64[w] = s64[w];
                        }
                        uint16_t* d16 = reinterpret_cast<uint16_t*>(cb_base);
                        for (uint32_t w = n64 << 2; w < n16; ++w) {
                            d16[w] = src_ptr[w];
                        }
                    } else {
                        for (uint32_t r = 0; r < a->ratio; ++r) {
                            uint16_t* dst_ptr = reinterpret_cast<uint16_t*>(cb_base + r * cb_page_bytes);
                            uint32_t off = r * new_half;
                            for (uint32_t w = 0; w < new_half; ++w) {
                                dst_ptr[w] = src_ptr[off + w];
                            }
                        }
                    }
                    drain_halfword_stores(
                        cb_base + (a->ratio - 1) * cb_page_bytes + a->new_stick_size - 2,
                        src_ptr[a->ratio * new_half - 1]);
                    cb.push_back(a->ratio);
                }
                i_stick += batch;
                remaining -= batch;
            }
        } else {
            // new >= old: ratio old sticks pack into one new stick.
            const uint32_t old_half = a->old_stick_size / 2;
            uint16_t last_value = 0;
            for (uint32_t iter = 0; iter < a->num_reads; ++iter) {
                cb.reserve_back(1);
                uint16_t* cb_dst = reinterpret_cast<uint16_t*>(cb.get_write_ptr());
                uint32_t dst_off = 0;
                uint32_t spr = a->sticks_per_read;
                uint32_t done = 0;
                while (done < spr) {
                    uint32_t batch = ((spr - done) < NABATCH) ? (spr - done) : NABATCH;
                    uint32_t byte_offsets[NABATCH];
                    for (uint32_t b = 0; b < batch; ++b) {
                        uint64_t noc_addr = s.get_noc_addr(i_stick + b, 0, noc.get_noc_id());
                        uint32_t scratch_slot = scratch_addr + b * scratch_slot_size;
                        byte_offsets[b] = (noc_addr - scratch_slot) & READ_ALIGNMENT_MASK;
                        noc.async_read(
                            s,
                            cb_scratch,
                            read_size,
                            {.page_id = i_stick + b, .offset_bytes = 0},
                            {.offset_bytes = b * scratch_slot_size + byte_offsets[b]});
                    }
                    noc.async_read_barrier();
                    invalidate_l1_cache();
                    for (uint32_t b = 0; b < batch; ++b) {
                        uint16_t* src_ptr =
                            reinterpret_cast<uint16_t*>(scratch_addr + b * scratch_slot_size + byte_offsets[b]);
                        for (uint32_t w = 0; w < old_half; ++w) {
                            cb_dst[dst_off + w] = src_ptr[w];
                        }
                        last_value = src_ptr[old_half - 1];
                        dst_off += old_half;
                    }
                    i_stick += batch;
                    done += batch;
                }
                drain_halfword_stores(reinterpret_cast<uint32_t>(cb_dst + dst_off - 1), last_value);
                cb.push_back(1);
            }
        }
    }
}
