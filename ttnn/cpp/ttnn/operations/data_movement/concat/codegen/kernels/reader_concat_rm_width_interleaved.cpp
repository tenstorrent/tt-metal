// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Concat reader for RM width-dim concat (interleaved).
// Each output stick = [in0_stick | in1_stick].
// Handles BH 64B alignment: uses aligned page sizes for TensorAccessor
// and scratch buffer for non-aligned reads.
//
// Fast path (ports the N-way reader's batched-barrier discipline): when both
// payloads qualify for direct into-CB reads (in0_aligned && in1_direct), the
// reader issues in0's and in1's reads back to back for BATCH sticks and
// drains them with ONE barrier per batch, then hands the whole batch to the
// writer.  The legacy discipline serialized TWO blocking DRAM round-trips per
// output stick with zero NOC overlap.  Batches address packed slot offsets
// relative to one reserve_back(batch) write pointer, so the builder asserts
// the CB depth is a multiple of BATCH (a batch never straddles the FIFO
// wrap).  Unaligned/scratch paths keep their per-read barriers.
//
// CT args layout:
//   [0]   cb_in
//   [1]   IN0_STICK_SIZE (actual bytes for in0 stick)
//   [2]   IN1_STICK_SIZE (actual bytes for in1 stick)
//   [3]   OUT_PAGE_SIZE  (aligned output page size)
//   [4]   IN0_PAGE_SIZE  (aligned page size for in0)
//   [5]   IN1_PAGE_SIZE  (aligned page size for in1)
//   [6]   cb_scratch
//   [7]   IN1_NOC_ALIGNMENT (source transport's legal destination alignment)
//   [8]   BATCH (sticks per reserve_back/barrier on the direct fast path)
//   [9..] TensorAccessorArgs for tensor 0, then tensor 1
//
// RT args layout:
//   [0]   num_sticks (total output sticks this core produces)
//   [1]   src_addr_0
//   [2]   src_addr_1
//   [3]   start_stick_0 (start stick id in tensor 0)
//   [4]   start_stick_1 (start stick id in tensor 1)
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "ckernel.h"

// Copy `size` bytes from `src_addr` to `dst_addr` using 16-bit stores only, for
// a segment whose packed destination offset or size is odd (a 1-byte dtype with
// an odd width).  Every halfword the segment touches is written as a whole; the
// two edge halfwords are read-modify-write so the neighbouring byte survives.
// Returns the address of the final halfword store -- the exact address the drain
// must load, because ckernel::load_blocking has no 8-bit form and Wormhole
// matches a load to a store by start address alone.
//
// The neighbouring byte is never written concurrently, so the RMW cannot lose
// an update:
//   * A direct NOC read (in0 when aligned, in1 when in1_direct) starts at 0 or at
//     IN0_STICK_SIZE (a multiple of IN1_NOC_ALIGNMENT) and spans a whole physical
//     page, so it begins and ends on an even address.  An edge halfword therefore
//     never straddles a direct segment's boundary.
//   * The only other neighbour is the other staged segment (or page padding),
//     written by this RISC in program order.
// The head RMW of one segment loads the halfword its predecessor's tail RMW
// just stored, at the identical start address, so Wormhole's start-address
// conflict check orders it after that store.
FORCE_INLINE uint32_t copy_bytes_halfword_rmw(uint32_t src_addr, uint32_t dst_addr, uint32_t size) {
    CoreLocalMem<volatile uint8_t> src(src_addr);
    const uint32_t end = dst_addr + size;
    const uint32_t last = (end - 1) & ~1U;
    for (uint32_t half = dst_addr & ~1U; half <= last; half += 2) {
        CoreLocalMem<volatile uint16_t> target(half);
        const bool lo_in = half >= dst_addr;
        const bool hi_in = half + 1 < end;
        uint16_t value = 0;
        if (!(lo_in && hi_in)) {
            value = target[0];
        }
        if (lo_in) {
            value = (value & 0xFF00U) | src[half - dst_addr];
        }
        if (hi_in) {
            value = (value & 0x00FFU) | (static_cast<uint16_t>(src[half + 1 - dst_addr]) << 8);
        }
        target[0] = value;
    }
    return last;
}

void kernel_main() {
    uint32_t num_sticks = get_arg_val<uint32_t>(0);
    const uint32_t src_addr_0 = get_arg_val<uint32_t>(1);
    const uint32_t src_addr_1 = get_arg_val<uint32_t>(2);
    uint32_t stick_id_0 = get_arg_val<uint32_t>(3);
    uint32_t stick_id_1 = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_in = get_compile_time_arg_val(0);
    constexpr uint32_t IN0_STICK_SIZE = get_compile_time_arg_val(1);
    constexpr uint32_t IN1_STICK_SIZE = get_compile_time_arg_val(2);
    constexpr uint32_t OUT_PAGE_SIZE = get_compile_time_arg_val(3);
    constexpr uint32_t IN0_PAGE_SIZE = get_compile_time_arg_val(4);
    constexpr uint32_t IN1_PAGE_SIZE = get_compile_time_arg_val(5);
    constexpr uint32_t cb_scratch = get_compile_time_arg_val(6);
    constexpr uint32_t IN1_NOC_ALIGNMENT = get_compile_time_arg_val(7);
    constexpr uint32_t BATCH = get_compile_time_arg_val(8);

    constexpr uint32_t ta_base = 9;
    constexpr auto ta0_args = TensorAccessorArgs<ta_base>();
    constexpr auto ta1_args = TensorAccessorArgs<ta0_args.next_compile_time_args_offset()>();

    // TensorAccessors use aligned page sizes for correct NOC address computation
    const auto s0 = TensorAccessor(ta0_args, src_addr_0, IN0_PAGE_SIZE);
    const auto s1 = TensorAccessor(ta1_args, src_addr_1, IN1_PAGE_SIZE);

    constexpr bool in0_aligned = (IN0_STICK_SIZE == IN0_PAGE_SIZE);
    constexpr bool in1_aligned = (IN1_STICK_SIZE == IN1_PAGE_SIZE);
    // Input 1 lands after input 0 inside the output page. Even when input 1's
    // source page is aligned, that destination is not a legal NOC endpoint if
    // the first payload is not aligned for input 1's source transport.  DRAM
    // therefore needs the architecture's DRAM alignment; an L1 source only
    // needs the L1 alignment. Stage through scratch and RISC-copy otherwise.
    constexpr bool in1_direct = in1_aligned && ((IN0_STICK_SIZE % IN1_NOC_ALIGNMENT) == 0);

    Noc noc;
    CircularBuffer input_cb(cb_in);
    CircularBuffer scratch_cb(cb_scratch);

    if constexpr (in0_aligned && in1_direct) {
        // Batched direct path (unconditional since the 2026-08 deletion wave
        // retired the DIRECT_BATCH gate).  Every destination offset below is
        // exactly the set the per-stick loop below reads into (slot strides
        // of OUT_PAGE_SIZE from the CB write pointer, payload offsets 0 and
        // IN0_STICK_SIZE) — only the barrier placement differs: all reads of
        // a batch are in flight together and drained once.
        while (num_sticks > 0) {
            const uint32_t batch = num_sticks < BATCH ? num_sticks : BATCH;
            input_cb.reserve_back(batch);
            for (uint32_t stick = 0; stick < batch; ++stick) {
                const uint32_t slot = stick * OUT_PAGE_SIZE;
                noc.async_read(s0, input_cb, IN0_STICK_SIZE, {.page_id = stick_id_0 + stick}, {.offset_bytes = slot});
                noc.async_read(
                    s1,
                    input_cb,
                    IN1_STICK_SIZE,
                    {.page_id = stick_id_1 + stick},
                    {.offset_bytes = slot + IN0_STICK_SIZE});
            }
            noc.async_read_barrier();
            input_cb.push_back(batch);
            stick_id_0 += batch;
            stick_id_1 += batch;
            num_sticks -= batch;
        }
        return;
    }

    // Reserve scratch buffer for non-aligned reads
    if constexpr (!in0_aligned || !in1_direct) {
        scratch_cb.reserve_back(1);
    }
    uint32_t scratch_addr = 0;
    if constexpr (!in0_aligned || !in1_direct) {
        scratch_addr = scratch_cb.get_write_ptr();
    }

    for (uint32_t i = 0; i < num_sticks; i++) {
        input_cb.reserve_back(1);
        uint32_t l1_addr = input_cb.get_write_ptr();

        // Read in0's stick
        if constexpr (in0_aligned) {
            noc.async_read(s0, input_cb, IN0_STICK_SIZE, {.page_id = stick_id_0}, {.offset_bytes = 0});
            noc.async_read_barrier();
        } else {
            // Non-aligned: read aligned page into scratch, copy actual bytes
            noc.async_read(s0, scratch_cb, IN0_PAGE_SIZE, {.page_id = stick_id_0}, {.offset_bytes = 0});
            noc.async_read_barrier();
            if constexpr ((IN0_STICK_SIZE & 1U) != 0) {
                (void)copy_bytes_halfword_rmw(scratch_addr, l1_addr, IN0_STICK_SIZE);
            } else {
                CoreLocalMem<volatile uint16_t> src_ptr(scratch_addr);
                CoreLocalMem<volatile uint16_t> dst_ptr(l1_addr);
                for (uint32_t w = 0; w < IN0_STICK_SIZE / 2; ++w) {
                    dst_ptr[w] = src_ptr[w];
                }
            }
        }

        // Read in1's stick right after in0's
        if constexpr (in1_direct) {
            noc.async_read(s1, input_cb, IN1_STICK_SIZE, {.page_id = stick_id_1}, {.offset_bytes = IN0_STICK_SIZE});
            noc.async_read_barrier();
        } else {
            noc.async_read(s1, scratch_cb, IN1_PAGE_SIZE, {.page_id = stick_id_1}, {.offset_bytes = 0});
            noc.async_read_barrier();
            if constexpr (((IN0_STICK_SIZE | IN1_STICK_SIZE) & 1U) != 0) {
                (void)copy_bytes_halfword_rmw(scratch_addr, l1_addr + IN0_STICK_SIZE, IN1_STICK_SIZE);
            } else {
                CoreLocalMem<volatile uint16_t> src_ptr(scratch_addr);
                CoreLocalMem<volatile uint16_t> dst_ptr(l1_addr + IN0_STICK_SIZE);
                for (uint32_t w = 0; w < IN1_STICK_SIZE / 2; ++w) {
                    dst_ptr[w] = src_ptr[w];
                }
            }
        }

        if constexpr (!in0_aligned || !in1_direct) {
            // The staged copies above end in a 16-bit store (a plain halfword
            // loop, or copy_bytes_halfword_rmw for an odd size/offset), so the
            // drain below can always name the final store.  An input read
            // straight into the CB needs an even IN0_STICK_SIZE (its offset is
            // a multiple of IN1_NOC_ALIGNMENT), so an odd one is always staged.
            static_assert(!in1_direct || IN0_STICK_SIZE % 2 == 0, "a direct in1 read must start on an even offset");
            // A RISC store can retire before its write-request reaches L1, and
            // the NoC and the CB consumer are separate L1 clients with no
            // program order against this core.  push_back only bumps a stream
            // register -- a different memory region -- so neither it nor a fence
            // orders the copy against it.  Read the last stored halfword back
            // instead: the read response proves L1 processed that write, and the
            // store queue is FIFO into one region, so every earlier store landed
            // with it.  The load must start at exactly the last store's address
            // -- Wormhole detects store/load conflicts on the starting byte
            // address alone, so a load of the containing word does not order a
            // halfword stored partway into it.
            // The halfword holding the last staged byte: staged_end - 2 for an
            // even end, staged_end - 1 for an odd one (the RMW's address).
            constexpr uint32_t staged_end = in1_direct ? IN0_STICK_SIZE : IN0_STICK_SIZE + IN1_STICK_SIZE;
            (void)ckernel::load_blocking(
                reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1_addr + ((staged_end - 1) & ~1U)));
        }

        input_cb.push_back(1);

        stick_id_0++;
        stick_id_1++;
    }
}
