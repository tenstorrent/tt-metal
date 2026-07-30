// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reader kernel (reader RISC, NOC_0). Streams this producer's assigned input tokens out of DRAM into the
// L1 ring the producer sends from. The producer writes L1 -> eth over NOC_1, so the two do not contend.
//
// This core serves SEVERAL assignments — its plane's share of the movements to every other chip on the
// axis (see assign_movements_to_producers in the program factory). The assignments are walked in order and
// their tokens form ONE continuous stream through the ring: `read`, `filled` and `freed` never restart at
// an assignment boundary. That is why no per-assignment handshake is needed — both sides hold the same
// compile-time assignment table, so the stream is self-describing by position, and the producer switches
// destination when its own running count crosses the boundary.
//
// The ring is hand-rolled rather than a metal CB because the op's telemetry and packet headers live at
// fixed offsets from the L1 allocator base — where a CB would be allocated — and the host telemetry
// readback depends on those offsets being predictable without knowing anything about allocation.
//
// Two monotonic single-writer counters, each bumped by a NoC atomic to our OWN core (the proven idiom for
// cross-RISC visibility on one core; a plain store can sit in a write buffer where the other RISC will not
// see it). `filled` is ours, `freed` is the producer's, and each side keeps its own local count and works
// on the difference, so there is no read-modify-write to race.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc_semaphore.h"

constexpr uint32_t TELEM_T_START_LO = 4;
constexpr uint32_t TELEM_T_START_HI = 5;

inline uint64_t wall_clock() {
#if defined(RISCV_DEBUG_REG_WALL_CLOCK_L) && defined(RISCV_DEBUG_REG_WALL_CLOCK_H)
    volatile uint32_t tt_reg_ptr* lo = reinterpret_cast<volatile uint32_t tt_reg_ptr*>(RISCV_DEBUG_REG_WALL_CLOCK_L);
    volatile uint32_t tt_reg_ptr* hi = reinterpret_cast<volatile uint32_t tt_reg_ptr*>(RISCV_DEBUG_REG_WALL_CLOCK_H);
    const uint32_t low = lo[0];  // latches high
    return static_cast<uint64_t>(low) | (static_cast<uint64_t>(hi[0]) << 32);
#else
    return 0;
#endif
}

void kernel_main() {
    constexpr uint32_t num_l1_slots = get_compile_time_arg_val(0);
    constexpr uint32_t token_size_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t slot_tail_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t batch = get_compile_time_arg_val(3);
    constexpr uint32_t ring_addr = get_compile_time_arg_val(4);
    constexpr uint32_t filled_addr = get_compile_time_arg_val(5);
    constexpr uint32_t freed_addr = get_compile_time_arg_val(6);
    constexpr uint32_t my_noc_x = get_compile_time_arg_val(7);
    constexpr uint32_t my_noc_y = get_compile_time_arg_val(8);
    constexpr uint32_t telemetry_addr = get_compile_time_arg_val(9);
    constexpr uint32_t dram_in_base_addr = get_compile_time_arg_val(10);
    constexpr uint32_t num_assignments = get_compile_time_arg_val(11);
    // The per-assignment table starts here: [in_base_token, num_tokens] each. Read through
    // kernel_compile_time_args (a constexpr std::array) rather than get_compile_time_arg_val, because the
    // latter needs a literal index and this table is walked by a loop variable.
    // Forwarding buffer, plumbed from P9.1 and used from P9.2.
    constexpr uint32_t dram_fwd_base_addr = get_compile_time_arg_val(12);
    constexpr uint32_t fwd_chunks_per_quarter = get_compile_time_arg_val(13);
    constexpr uint32_t fwd_pages_per_chunk = get_compile_time_arg_val(14);
    constexpr uint32_t ASSIGN_BASE = 15;
    constexpr uint32_t ASSIGN_WORDS = 2;
    constexpr auto dram_in_args = TensorAccessorArgs<ASSIGN_BASE + ASSIGN_WORDS * num_assignments>();
    // A ring slot is the token plus the metadata tail the producer reads its routing from.
    constexpr uint32_t slot_stride = token_size_bytes + slot_tail_bytes;

    // Written only by the producer; we just read it.
    volatile tt_l1_ptr uint32_t* freed = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(freed_addr);
    const uint64_t my_filled_noc = get_noc_addr(my_noc_x, my_noc_y, filled_addr);
    const auto dram_in = TensorAccessor(dram_in_args, dram_in_base_addr);

    // The effective-bandwidth window opens HERE: the first DRAM read is now part of the measured cost,
    // not prep work factored out of it. The producer folds nothing into this — the host reads it directly.
    volatile tt_l1_ptr uint32_t* telem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(telemetry_addr);
    const uint64_t t_start = wall_clock();
    telem[TELEM_T_START_LO] = (uint32_t)(t_start & 0xFFFFFFFFu);
    telem[TELEM_T_START_HI] = (uint32_t)(t_start >> 32);

    // `read` counts tokens published across ALL assignments — the ring and both counters are continuous,
    // so an assignment boundary is invisible to the flow control.
    uint32_t read = 0;
    for (uint32_t a = 0; a < num_assignments; a++) {
        const uint32_t in_base_page = kernel_compile_time_args[ASSIGN_BASE + a * ASSIGN_WORDS + 0];
        const uint32_t assignment_tokens = kernel_compile_time_args[ASSIGN_BASE + a * ASSIGN_WORDS + 1];
        uint32_t done = 0;
        while (done < assignment_tokens) {
            const uint32_t n = (assignment_tokens - done) < batch ? (assignment_tokens - done) : batch;
            // Wait for n free slots. `read - freed` is what we have published but the producer has not
            // released yet, so free = num_l1_slots - (read - freed).
            while (true) {
                invalidate_l1_cache();
                if (read - *freed + n <= num_l1_slots) {
                    break;
                }
            }
            for (uint32_t i = 0; i < n; i++) {
                const uint32_t slot = (read + i) % num_l1_slots;
                noc_async_read(
                    dram_in.get_noc_addr(in_base_page + done + i), ring_addr + slot * slot_stride, token_size_bytes);
            }
            noc_async_read_barrier();  // the data is in L1 before we say it is
            read += n;
            done += n;
            noc_semaphore_inc(my_filled_noc, n);
        }
    }
}
