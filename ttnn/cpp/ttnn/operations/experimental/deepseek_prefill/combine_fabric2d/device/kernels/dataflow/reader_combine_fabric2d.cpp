// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reader kernel (reader RISC, NOC_0). Feeds the L1 ring the producer sends from, and — from phase 9 — is
// where ALL routing decisions are made. The producer writes L1 -> eth over NOC_1, so the two do not contend.
//
// Phase 9: the fabric no longer forwards. Every packet the producer sends travels exactly ONE hop, to the
// chip across its own cable. A token bound further than that neighbour therefore cannot go straight to its
// final destination; it is staged into the NEIGHBOUR's forwarding buffer and re-sent from there by the
// neighbour's reader on the same (plane, direction). So this kernel has two phases:
//
//   A. Its OWN assignments — its plane's share of this chip's movements. Each destination is either our
//      immediate neighbour (=> CMD_FINAL_WRITE, the neighbour writes it straight into its output region) or
//      further away (=> CMD_FORWARD into the neighbour's forwarding buffer). A forwarding assignment ends
//      with a SENTINEL token so the downstream reader knows where the chunk stops.
//
//   B. The chunks that arrived in OUR quarter of the forwarding buffer, written by the upstream chip's
//      producer on the same (plane, direction). Each is pushed one more hop: final-write if its destination
//      is now our neighbour, re-forward otherwise. A sentinel is passed on only when re-forwarding — a final
//      write leaves no chunk downstream to terminate.
//
// The stream ends with a CMD_END slot, because its length is not knowable up front: how much this core
// re-forwards depends on chunk sizes decided upstream.
//
// The forwarding-buffer page layout is deliberately [token][final_addr][dst_chip], which is EXACTLY the
// first token_size+16 bytes of a ring slot. Re-forwarding is therefore a single DRAM read of
// token_size+16 bytes straight into the slot base: payload and both metadata words land where the producer
// already expects them, with no unpacking.
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

// Slot metadata tail, as uint64_t indices from (slot_base + token_size_bytes). Must match the producer.
constexpr uint32_t TAIL_FINAL_ADDR = 0;
constexpr uint32_t TAIL_DST_CHIP = 1;
constexpr uint32_t TAIL_CMD = 2;
constexpr uint32_t TAIL_THIS_ADDR = 3;
constexpr uint64_t CMD_END = 0;
constexpr uint64_t CMD_FINAL_WRITE = 1;
constexpr uint64_t CMD_FORWARD = 2;
// A sentinel carries no usable token; it exists only to mark the end of a forwarding chunk. UINT64_MAX can
// never collide with a real chip id.
constexpr uint64_t SENTINEL_DST_CHIP = UINT64_MAX;

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
    constexpr uint32_t dram_out_base_addr = get_compile_time_arg_val(11);
    constexpr uint32_t dram_fwd_base_addr = get_compile_time_arg_val(12);
    constexpr uint32_t fwd_chunks_per_quarter = get_compile_time_arg_val(13);
    constexpr uint32_t fwd_pages_per_chunk = get_compile_time_arg_val(14);
    // Our quarter of the forwarding buffer. (plane, direction) identifies the upstream producer uniquely
    // from the downstream chip's point of view, so we WRITE quarter q of the neighbour's buffer and READ
    // quarter q of our own — the same q, because every chip runs this same code.
    constexpr uint32_t my_quarter = get_compile_time_arg_val(15);
    constexpr uint32_t num_incoming_chunks = get_compile_time_arg_val(16);
    constexpr uint32_t fwd_sem_addr = get_compile_time_arg_val(17);
    // The chip across our own cable. A destination equal to this is one hop away, so the neighbour can put
    // it straight into its output region.
    constexpr uint32_t nbr_chip_id = get_compile_time_arg_val(18);
    constexpr uint32_t num_assignments = get_compile_time_arg_val(19);
    // Per-assignment table: [in_base_token, num_tokens, out_base_token, dst_chip_id]. Read through
    // kernel_compile_time_args (a constexpr std::array) because get_compile_time_arg_val needs a literal
    // index and this table is walked by a loop variable.
    constexpr uint32_t ASSIGN_BASE = 20;
    constexpr uint32_t ASSIGN_WORDS = 4;
    constexpr uint32_t ACCESSOR_BASE = ASSIGN_BASE + ASSIGN_WORDS * num_assignments;
    constexpr auto dram_in_args = TensorAccessorArgs<ACCESSOR_BASE>();
    constexpr auto dram_out_args = TensorAccessorArgs<dram_in_args.next_compile_time_args_offset()>();
    constexpr auto dram_fwd_args = TensorAccessorArgs<dram_out_args.next_compile_time_args_offset()>();
    constexpr uint32_t slot_stride = token_size_bytes + slot_tail_bytes;
    // Re-forwarding reads the token AND the two metadata words that follow it in the page.
    constexpr uint32_t FWD_EXTRA_BYTES = 16;
    constexpr uint32_t fwd_read_bytes = token_size_bytes + FWD_EXTRA_BYTES;

    // Written only by the producer; we just read it.
    volatile tt_l1_ptr uint32_t* freed = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(freed_addr);
    const uint64_t my_filled_noc = get_noc_addr(my_noc_x, my_noc_y, filled_addr);
    // Bumped by the UPSTREAM chip's producer as it fills our quarter. A GlobalSemaphore, so the framework
    // zeroes it before launch — the phase-6 lesson about raw L1 counters underflowing on stale values.
    volatile tt_l1_ptr uint32_t* fwd_arrived = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(fwd_sem_addr);

    const auto dram_in = TensorAccessor(dram_in_args, dram_in_base_addr);
    const auto dram_out = TensorAccessor(dram_out_args, dram_out_base_addr);
    const auto dram_fwd = TensorAccessor(dram_fwd_args, dram_fwd_base_addr);

    // The effective-bandwidth window opens HERE: the first DRAM read is part of the measured cost, not prep
    // work factored out of it. The producer folds nothing into this — the host reads it directly.
    volatile tt_l1_ptr uint32_t* telem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(telemetry_addr);
    const uint64_t t_start = wall_clock();
    telem[TELEM_T_START_LO] = (uint32_t)(t_start & 0xFFFFFFFFu);
    telem[TELEM_T_START_HI] = (uint32_t)(t_start >> 32);

    // Page of a chunk within OUR quarter. The same formula serves both directions: the quarter index is ours
    // either way, only the chip differs, and the buffer's device-local address is uniform across the mesh.
    auto fwd_page = [](uint32_t chunk, uint32_t token) -> uint32_t {
        return my_quarter * fwd_chunks_per_quarter * fwd_pages_per_chunk + chunk * fwd_pages_per_chunk + token;
    };

    // `published` counts slots handed to the producer across BOTH phases; the ring and both counters are
    // continuous, so a phase or chunk boundary is invisible to the flow control.
    uint32_t published = 0;
    // Which chunk of the neighbour's quarter the next forwarding batch goes into. Our own forwarding
    // assignments take 0.., then our re-forwards continue from there — and the downstream reader walks its
    // chunks in exactly that order, which is what makes the two agree with no negotiation.
    uint32_t out_chunk = 0;

    // Publishing is batched so the atomic is amortised, matching the producer's batch.
    uint32_t pending_publish = 0;
    auto flush_publish = [&]() {
        if (pending_publish > 0) {
            noc_semaphore_inc(my_filled_noc, pending_publish);
            pending_publish = 0;
        }
    };

    // Claim one ring slot, blocking until the producer frees one. Publishes anything pending FIRST: never
    // make the producer wait on slots we are holding back, which would deadlock us against each other.
    auto claim_slot = [&]() -> uint32_t {
        invalidate_l1_cache();
        if (published - *freed >= num_l1_slots) {
            flush_publish();
            while (true) {
                invalidate_l1_cache();
                if (published - *freed < num_l1_slots) {
                    break;
                }
            }
        }
        return published % num_l1_slots;
    };

    // ---- Phase A: this producer's own assignments.
    for (uint32_t a = 0; a < num_assignments; a++) {
        const uint32_t in_base_page = kernel_compile_time_args[ASSIGN_BASE + a * ASSIGN_WORDS + 0];
        const uint32_t assignment_tokens = kernel_compile_time_args[ASSIGN_BASE + a * ASSIGN_WORDS + 1];
        const uint32_t out_base_page = kernel_compile_time_args[ASSIGN_BASE + a * ASSIGN_WORDS + 2];
        const uint32_t dst_chip_id = kernel_compile_time_args[ASSIGN_BASE + a * ASSIGN_WORDS + 3];
        const bool direct = (dst_chip_id == nbr_chip_id);

        for (uint32_t i = 0; i < assignment_tokens; i++) {
            const uint32_t slot = claim_slot();
            const uint32_t slot_addr = ring_addr + slot * slot_stride;
            noc_async_read(dram_in.get_noc_addr(in_base_page + i), slot_addr, token_size_bytes);
            volatile tt_l1_ptr uint64_t* tail =
                reinterpret_cast<volatile tt_l1_ptr uint64_t*>(slot_addr + token_size_bytes);
            // The FINAL destination address is computed here, once, and then travels with the token for
            // however many hops it takes. Valid on any chip because the output buffer's base address and
            // interleaved bank mapping are uniform across the mesh.
            const uint64_t final_addr = dram_out.get_noc_addr(out_base_page + i);
            if (direct) {
                tail[TAIL_CMD] = CMD_FINAL_WRITE;
                tail[TAIL_THIS_ADDR] = final_addr;
            } else {
                tail[TAIL_FINAL_ADDR] = final_addr;
                tail[TAIL_DST_CHIP] = (uint64_t)dst_chip_id;
                tail[TAIL_CMD] = CMD_FORWARD;
                tail[TAIL_THIS_ADDR] = dram_fwd.get_noc_addr(fwd_page(out_chunk, i));
            }
            noc_async_read_barrier();  // the token is in L1 before we say it is
            published++;
            if (++pending_publish >= batch) {
                flush_publish();
            }
        }

        if (!direct) {
            // Sentinel: terminates this chunk for the downstream reader. Carries no usable token, so no DRAM
            // read — only the tail matters. It occupies the chunk's last page, which is exactly why the
            // buffer is sized with one page more than tokens_per_movement.
            const uint32_t slot = claim_slot();
            volatile tt_l1_ptr uint64_t* tail =
                reinterpret_cast<volatile tt_l1_ptr uint64_t*>(ring_addr + slot * slot_stride + token_size_bytes);
            tail[TAIL_FINAL_ADDR] = 0;
            tail[TAIL_DST_CHIP] = SENTINEL_DST_CHIP;
            tail[TAIL_CMD] = CMD_FORWARD;
            tail[TAIL_THIS_ADDR] = dram_fwd.get_noc_addr(fwd_page(out_chunk, assignment_tokens));
            published++;
            pending_publish++;
            flush_publish();  // a chunk boundary: do not let it sit unpublished
            out_chunk++;
        }
    }
    flush_publish();

    // ---- Phase B: chunks that arrived in our own quarter, each pushed one more hop.
    uint32_t consumed = 0;  // forwarded tokens (sentinels included) taken out of our quarter
    for (uint32_t chunk = 0; chunk < num_incoming_chunks; chunk++) {
        // Every token of a chunk shares one final destination (a chunk IS one movement's tokens), so the
        // first token decides for the whole chunk whether this hop is the last one.
        bool reforward = false;
        bool decided = false;
        for (uint32_t i = 0;; i++) {
            // Do not outpace the upstream producer. It bumps this counter every fwd_bump_every tokens and
            // ALWAYS right after a sentinel, so a chunk boundary is never left unreachable.
            invalidate_l1_cache();
            if (*fwd_arrived <= consumed) {
                flush_publish();  // let the producer work while we wait on upstream
                while (true) {
                    invalidate_l1_cache();
                    if (*fwd_arrived > consumed) {
                        break;
                    }
                }
            }
            const uint32_t slot = claim_slot();
            const uint32_t slot_addr = ring_addr + slot * slot_stride;
            // ONE read brings the token AND the [final_addr][dst_chip] pair straight into the slot's tail,
            // because the page layout was chosen to match the slot layout exactly.
            noc_async_read(dram_fwd.get_noc_addr(fwd_page(chunk, i)), slot_addr, fwd_read_bytes);
            noc_async_read_barrier();
            consumed++;

            volatile tt_l1_ptr uint64_t* tail =
                reinterpret_cast<volatile tt_l1_ptr uint64_t*>(slot_addr + token_size_bytes);
            const uint64_t dst_chip = tail[TAIL_DST_CHIP];

            if (dst_chip == SENTINEL_DST_CHIP) {
                if (reforward) {
                    // Pass the terminator on, at the end of the chunk we have been writing downstream.
                    tail[TAIL_CMD] = CMD_FORWARD;
                    tail[TAIL_THIS_ADDR] = dram_fwd.get_noc_addr(fwd_page(out_chunk, i));
                    published++;
                    pending_publish++;
                }
                flush_publish();
                break;
            }

            if (!decided) {
                reforward = (dst_chip != (uint64_t)nbr_chip_id);
                decided = true;
            }
            if (reforward) {
                tail[TAIL_CMD] = CMD_FORWARD;
                tail[TAIL_THIS_ADDR] = dram_fwd.get_noc_addr(fwd_page(out_chunk, i));
            } else {
                tail[TAIL_CMD] = CMD_FINAL_WRITE;
                tail[TAIL_THIS_ADDR] = tail[TAIL_FINAL_ADDR];
            }
            published++;
            if (++pending_publish >= batch) {
                flush_publish();
            }
        }
        if (reforward) {
            out_chunk++;
        }
    }
    flush_publish();

    // ---- End of stream. The producer cannot know the length up front, so it stops on this.
    {
        const uint32_t slot = claim_slot();
        volatile tt_l1_ptr uint64_t* tail =
            reinterpret_cast<volatile tt_l1_ptr uint64_t*>(ring_addr + slot * slot_stride + token_size_bytes);
        tail[TAIL_CMD] = CMD_END;
        published++;
        pending_publish++;
        flush_publish();
    }
}
