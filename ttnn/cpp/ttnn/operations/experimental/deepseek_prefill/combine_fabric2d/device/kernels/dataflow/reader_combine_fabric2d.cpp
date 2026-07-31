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
    constexpr uint32_t schedule_len = get_compile_time_arg_val(20);
    // ---- Phase 10: the work itself lives in the control tensors, so what arrives here is where they are
    // and the shape constants needed to index them.
    constexpr uint32_t dram_meta_base_addr = get_compile_time_arg_val(21);
    constexpr uint32_t dram_counts_base_addr = get_compile_time_arg_val(22);
    constexpr uint32_t dram_region_base_addr = get_compile_time_arg_val(23);
    constexpr uint32_t dram_expert_offsets_base_addr = get_compile_time_arg_val(24);
    constexpr uint32_t num_routed_experts = get_compile_time_arg_val(25);
    constexpr uint32_t experts_per_chip = get_compile_time_arg_val(26);
    // First of the `experts_per_chip` columns THIS chip hosts. Everything this kernel does is confined to
    // those columns: they are the experts whose tokens are physically here.
    constexpr uint32_t my_expert_base = get_compile_time_arg_val(27);
    constexpr uint32_t num_experts_per_tok = get_compile_time_arg_val(28);
    constexpr uint32_t dispatch_group_size = get_compile_time_arg_val(29);
    // Our slice of the same-chip run, done after all the fabric work (see the local phase at the end).
    constexpr uint32_t local_split_idx = get_compile_time_arg_val(30);
    constexpr uint32_t local_split_count = get_compile_time_arg_val(31);
    constexpr uint32_t my_row = get_compile_time_arg_val(32);
    constexpr uint32_t control_addr = get_compile_time_arg_val(33);
    // Work schedule: one entry per unit of work, high bit set = "relay forwarding chunk k", clear = "own
    // assignment k". Which order to do them in is the factory's call (assignment_order); this kernel just
    // walks the list. Interleaving own work with relaying is what keeps downstream cores fed.
    constexpr uint32_t SCHED_BASE = 34;
    constexpr uint32_t SCHED_FWD = 0x80000000u;
    // Per-assignment table: [dst_chip_id, dst_row, split_idx, split_count]. Read through
    // kernel_compile_time_args (a constexpr std::array) because get_compile_time_arg_val needs a literal
    // index and these tables are walked by a loop variable.
    constexpr uint32_t ASSIGN_BASE = SCHED_BASE + schedule_len;
    constexpr uint32_t ASSIGN_WORDS = 4;
    constexpr uint32_t ACCESSOR_BASE = ASSIGN_BASE + ASSIGN_WORDS * num_assignments;
    constexpr auto dram_in_args = TensorAccessorArgs<ACCESSOR_BASE>();
    constexpr auto dram_out_args = TensorAccessorArgs<dram_in_args.next_compile_time_args_offset()>();
    constexpr auto dram_fwd_args = TensorAccessorArgs<dram_out_args.next_compile_time_args_offset()>();
    constexpr auto dram_meta_args = TensorAccessorArgs<dram_fwd_args.next_compile_time_args_offset()>();
    constexpr auto dram_counts_args = TensorAccessorArgs<dram_meta_args.next_compile_time_args_offset()>();
    constexpr auto dram_region_args = TensorAccessorArgs<dram_counts_args.next_compile_time_args_offset()>();
    constexpr auto dram_expert_offsets_args = TensorAccessorArgs<dram_region_args.next_compile_time_args_offset()>();
    constexpr uint32_t slot_stride = token_size_bytes + slot_tail_bytes;
    // Re-forwarding reads the token AND the two metadata words that follow it in the page.
    constexpr uint32_t FWD_EXTRA_BYTES = 16;
    constexpr uint32_t fwd_read_bytes = token_size_bytes + FWD_EXTRA_BYTES;
    // A token's routing metadata lands in a single 64-byte pad at the front of the control region, NOT in
    // the ring slot: a DRAM read needs a 64-byte-aligned L1 destination on Blackhole
    // (LOG_BASE_2_OF_DRAM_ALIGNMENT = 6) and no offset inside a slot's tail is 64-byte aligned. One pad
    // suffices because the record is consumed immediately after the barrier, before the slot is published,
    // so only ever one is in flight.
    constexpr uint32_t meta_scratch_addr = control_addr;
    constexpr uint32_t META_READ_BYTES = 64;
    constexpr uint32_t control_tables_addr = control_addr + 64;

    // Written only by the producer; we just read it.
    volatile tt_l1_ptr uint32_t* freed = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(freed_addr);
    const uint64_t my_filled_noc = get_noc_addr(my_noc_x, my_noc_y, filled_addr);
    // Bumped by the UPSTREAM chip's producer as it fills our quarter. A GlobalSemaphore, so the framework
    // zeroes it before launch — the phase-6 lesson about raw L1 counters underflowing on stale values.
    volatile tt_l1_ptr uint32_t* fwd_arrived = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(fwd_sem_addr);

    const auto dram_in = TensorAccessor(dram_in_args, dram_in_base_addr);
    const auto dram_out = TensorAccessor(dram_out_args, dram_out_base_addr);
    const auto dram_fwd = TensorAccessor(dram_fwd_args, dram_fwd_base_addr);
    const auto dram_meta = TensorAccessor(dram_meta_args, dram_meta_base_addr);
    const auto dram_counts = TensorAccessor(dram_counts_args, dram_counts_base_addr);
    const auto dram_region = TensorAccessor(dram_region_args, dram_region_base_addr);
    const auto dram_expert_offsets = TensorAccessor(dram_expert_offsets_args, dram_expert_offsets_base_addr);

    // ---- Control tensors, read once. All three are one row per page of `num_routed_experts` uint32:
    // expert_offsets has one row per ORIGIN chip (it is replicated along that axis so every chip sees all of
    // them), counts and region_offsets are a single row each. A few kB in total, so the whole slice comes in
    // rather than cherry-picking this chip's columns — a strided gather of 4-byte words would cost more NoC
    // transactions than the bytes saved.
    volatile tt_l1_ptr uint32_t* ctl_offsets = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(control_tables_addr);
    volatile tt_l1_ptr uint32_t* ctl_counts = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
        control_tables_addr + dispatch_group_size * num_routed_experts * 4);
    volatile tt_l1_ptr uint32_t* ctl_region = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
        control_tables_addr + (dispatch_group_size + 1) * num_routed_experts * 4);
    {
        constexpr uint32_t row_bytes = num_routed_experts * 4;
        for (uint32_t r = 0; r < dispatch_group_size; r++) {
            noc_async_read(dram_expert_offsets.get_noc_addr(r), control_tables_addr + r * row_bytes, row_bytes);
        }
        noc_async_read(dram_counts.get_noc_addr(0), (uint32_t)ctl_counts, row_bytes);
        noc_async_read(dram_region.get_noc_addr(0), (uint32_t)ctl_region, row_bytes);
        noc_async_read_barrier();
    }

    // Where origin chip `row`'s tokens for expert `e` start, and where they end. The runs are laid out in
    // origin-chip order inside the expert's region, so one run ends where the next begins; the last one ends
    // at the expert's total count past its region base, which is the only thing expert_offsets cannot say.
    auto run_begin = [&](uint32_t row, uint32_t e) -> uint32_t { return ctl_offsets[row * num_routed_experts + e]; };
    auto run_end = [&](uint32_t row, uint32_t e) -> uint32_t {
        return row + 1 < dispatch_group_size ? ctl_offsets[(row + 1) * num_routed_experts + e]
                                             : ctl_region[e] + ctl_counts[e];
    };

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

    // ---- Stream one token: read it and its routing metadata, work out where it must end up, hand the slot
    // to the producer. `chunk_page` is where it goes in the downstream forwarding chunk when this is not the
    // last hop; it is ignored for a direct write.
    auto stream_token = [&](uint32_t in_page, bool direct, uint32_t dst_chip_id, uint32_t chunk_page) {
        const uint32_t slot = claim_slot();
        const uint32_t slot_addr = ring_addr + slot * slot_stride;
        volatile tt_l1_ptr uint64_t* tail =
            reinterpret_cast<volatile tt_l1_ptr uint64_t*>(slot_addr + token_size_bytes);
        volatile tt_l1_ptr uint32_t* meta = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(meta_scratch_addr);
        // Both reads issued before the single barrier, so the small metadata read rides along with the
        // 14 kB token read instead of serialising behind it.
        noc_async_read(dram_in.get_noc_addr(in_page), slot_addr, token_size_bytes);
        noc_async_read(dram_meta.get_noc_addr(in_page), meta_scratch_addr, META_READ_BYTES);
        noc_async_read_barrier();  // the token AND its metadata are in L1 before we say the slot is ready

        // Metadata: [0] linearized_coord, [1] token_idx, [2] topk_idx. The destination chip is already known
        // from the run this token came out of, so only the slot within that chip's output is read here.
        const uint32_t out_page = meta[1] * num_experts_per_tok + meta[2];
        // The FINAL destination address is computed here, once, and then travels with the token for however
        // many hops it takes. Valid on any chip because the output buffer's base address and interleaved bank
        // mapping are uniform across the mesh.
        const uint64_t final_addr = dram_out.get_noc_addr(out_page);
        if (direct) {
            tail[TAIL_CMD] = CMD_FINAL_WRITE;
            tail[TAIL_THIS_ADDR] = final_addr;
        } else {
            tail[TAIL_FINAL_ADDR] = final_addr;
            tail[TAIL_DST_CHIP] = (uint64_t)dst_chip_id;
            tail[TAIL_CMD] = CMD_FORWARD;
            tail[TAIL_THIS_ADDR] = dram_fwd.get_noc_addr(fwd_page(out_chunk, chunk_page));
        }
        published++;
        if (++pending_publish >= batch) {
            flush_publish();
        }
    };

    // Our slice of a run: of [begin, end) take [begin + n*idx/count, begin + n*(idx+1)/count). Integer
    // arithmetic, so consecutive slices meet exactly — no gaps, no overlap, whatever n is.
    auto slice_begin = [](uint32_t begin, uint32_t n, uint32_t idx, uint32_t count) -> uint32_t {
        return begin + (uint32_t)(((uint64_t)n * idx) / count);
    };

    // ---- One own assignment: everything this chip owes ONE destination chip, over all the experts it hosts,
    // narrowed to this producer's slice of each run. The experts are merged into a single stream — a chunk is
    // one (this chip -> that chip) term, and which expert a token sat under is not something the forwarding
    // protocol needs to know.
    auto do_own_assignment = [&](uint32_t a) {
        const uint32_t dst_chip_id = kernel_compile_time_args[ASSIGN_BASE + a * ASSIGN_WORDS + 0];
        const uint32_t dst_row = kernel_compile_time_args[ASSIGN_BASE + a * ASSIGN_WORDS + 1];
        const uint32_t split_idx = kernel_compile_time_args[ASSIGN_BASE + a * ASSIGN_WORDS + 2];
        const uint32_t split_count = kernel_compile_time_args[ASSIGN_BASE + a * ASSIGN_WORDS + 3];
        const bool direct = (dst_chip_id == nbr_chip_id);

        uint32_t chunk_page = 0;  // position in the downstream chunk, continuous across our experts
        for (uint32_t le = 0; le < experts_per_chip; le++) {
            const uint32_t e = my_expert_base + le;
            const uint32_t begin = run_begin(dst_row, e);
            const uint32_t n = run_end(dst_row, e) - begin;
            const uint32_t lo = slice_begin(begin, n, split_idx, split_count);
            const uint32_t hi = slice_begin(begin, n, split_idx + 1, split_count);
            for (uint32_t p = lo; p < hi; p++) {
                stream_token(p, direct, dst_chip_id, chunk_page);
                if (!direct) {
                    chunk_page++;
                }
            }
        }

        if (!direct) {
            // Sentinel: terminates this chunk for the downstream reader. Carries no usable token, so no DRAM
            // read — only the tail matters. Chunk lengths are data-dependent now, which is exactly what the
            // sentinel exists for: nothing downstream assumes a chunk is full.
            //
            // It carries the chunk's destination in the otherwise-unused FINAL_ADDR word, because from phase
            // 10 a chunk can be EMPTY — a producer's slice of a run may hold no tokens at all. The relay
            // decides final-write-vs-re-forward from the first token it sees, and an empty chunk has none, so
            // without this the decision would be undecidable and the chunk count downstream would drift. Both
            // words travel with a forwarded packet (token + 16 bytes), so it propagates hop to hop for free.
            const uint32_t slot = claim_slot();
            volatile tt_l1_ptr uint64_t* tail =
                reinterpret_cast<volatile tt_l1_ptr uint64_t*>(ring_addr + slot * slot_stride + token_size_bytes);
            tail[TAIL_FINAL_ADDR] = (uint64_t)dst_chip_id;
            tail[TAIL_DST_CHIP] = SENTINEL_DST_CHIP;
            tail[TAIL_CMD] = CMD_FORWARD;
            tail[TAIL_THIS_ADDR] = dram_fwd.get_noc_addr(fwd_page(out_chunk, chunk_page));
            published++;
            pending_publish++;
            flush_publish();  // a chunk boundary: do not let it sit unpublished
            out_chunk++;
        }
    };

    // ---- One arrived chunk, pushed exactly one more hop.
    uint32_t consumed = 0;  // forwarded tokens (sentinels included) taken out of our quarter
    auto do_forward_chunk = [&](uint32_t chunk) {
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
                // An EMPTY chunk reaches its sentinel with nothing having decided for it, so the sentinel's
                // own destination word decides. The chunk still has to be passed on when it is not for our
                // neighbour: the downstream reader counts chunks positionally, so silently dropping one would
                // shift every chunk after it.
                if (!decided) {
                    reforward = (tail[TAIL_FINAL_ADDR] != (uint64_t)nbr_chip_id);
                    decided = true;
                }
                if (reforward) {
                    // Pass the terminator on, at the end of the chunk we have been writing downstream. Its
                    // destination word is already in place — it came in with the sentinel.
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
    };

    // ---- Walk the schedule. Forwarding chunks always appear in increasing order, so `consumed` stays a
    // valid watermark into our quarter however the own assignments are interleaved around them.
    for (uint32_t si = 0; si < schedule_len; si++) {
        const uint32_t entry = kernel_compile_time_args[SCHED_BASE + si];
        if (entry & SCHED_FWD) {
            do_forward_chunk(entry & ~SCHED_FWD);
        } else {
            do_own_assignment(entry);
        }
    }
    flush_publish();

    // ---- Same-chip tokens. An expert here also holds tokens that started here, and those need no cable at
    // all — they are a DRAM-to-DRAM copy on this chip. So they are not a producer's work and never appear in
    // its packet count; they go straight out over our own NoC.
    //
    // Done LAST and split `local_split_count` ways (one slice per core) for the same reason: it is pure local
    // bandwidth, so doing it while the fabric is still busy would compete with the token reads feeding it,
    // and leaving it to one core would make it a serial tail. Whether last is actually the best place for it,
    // and whether it belongs on separate cores entirely, is a stage-3 question.
    {
        // One slot serves as the staging buffer for the whole phase, and is deliberately NEVER published:
        // the producer must not see these tokens, or it would send them over the fabric as well. Slot
        // `published % num_l1_slots` is the next one the producer has yet to be told about, so it is ours to
        // scribble in; claim_slot() is still what proves at least one slot is free before we take it, and
        // `published` does not move during this phase so it stays free. The CMD_END slot below reclaims it.
        const uint32_t slot = claim_slot();
        const uint32_t slot_addr = ring_addr + slot * slot_stride;
        volatile tt_l1_ptr uint32_t* meta = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(meta_scratch_addr);
        for (uint32_t le = 0; le < experts_per_chip; le++) {
            const uint32_t e = my_expert_base + le;
            const uint32_t begin = run_begin(my_row, e);
            const uint32_t n = run_end(my_row, e) - begin;
            const uint32_t lo = slice_begin(begin, n, local_split_idx, local_split_count);
            const uint32_t hi = slice_begin(begin, n, local_split_idx + 1, local_split_count);
            for (uint32_t p = lo; p < hi; p++) {
                noc_async_read(dram_in.get_noc_addr(p), slot_addr, token_size_bytes);
                noc_async_read(dram_meta.get_noc_addr(p), meta_scratch_addr, META_READ_BYTES);
                noc_async_read_barrier();
                const uint32_t out_page = meta[1] * num_experts_per_tok + meta[2];
                noc_async_write(slot_addr, dram_out.get_noc_addr(out_page), token_size_bytes);
                // Serialised on purpose: the slot IS the buffer, so it cannot be refilled until the write has
                // read it out. Overlapping these needs several buffers in flight, tracked separately from the
                // producer's ring accounting — a stage-3 optimisation.
                noc_async_write_barrier();
            }
        }
    }

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
