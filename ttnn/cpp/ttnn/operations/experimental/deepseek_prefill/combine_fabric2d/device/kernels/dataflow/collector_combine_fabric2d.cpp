// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Collector kernel. Turns the routed expert's per-writer reports into one `ready` count every combine core
// that reads routed-expert output can poll locally.
//
// Every routed-expert writer reports once per expert slot it walks, in every pass, by bumping word s of a
// per-step count array here (see hybrid_expert_done.hpp). Step s is complete once that word reaches the
// writer count, and then `ready = s + 1` is set on each waiting core. The waiting cores derive, for each
// expert, the step that makes it ready, so this core needs to know nothing about experts.
//
// The array is plain L1, zeroed here at launch before the writers are told `go`: every bump of the previous
// launch was counted before that launch's collector exited, so nothing stale can land after the zeroing.
//
// With CMBF2D_DYNAMIC_ORDER the steps are released in the order the routed expert actually finishes them: each
// release position takes the lowest unreleased step that is complete, or failing that one with a chunk out (a
// whole-grid expert, which runs alone, so nothing else can finish before it). Every other step already complete
// goes out at the following positions in the same release, so a wave finishing at once costs one publish of the
// order and one of `ready`. Which step went out at each position is published on every waiting core BEFORE
// `progress` or `ready` names that position, and the untilizers and readers walk positions; only the untilizers
// take `progress` and `ready`. Without it, position k is step k.
//
// Unicast rather than one multicast: the waiting cores are scattered over rows 0-1, and a rectangle over them
// would also write into cores that are not part of this program. The routed expert's cores ARE a rectangle,
// which is what lets `go` be one multicast.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "combine_fabric2d_collector_ct_args.hpp"

constexpr ::cmbf2d::CollectorCtArgs ct{};

#if defined(CMBF2D_DYNAMIC_ORDER) && !defined(CMBF2D_PROGRESS_SEM)
#error "the release order is picked from the per-chunk reports, which only the progress build zeroes"
#endif

// Copy this core's semaphore `addr` to the same semaphore on the first `n` waiting cores.
void publish_to_targets(uint32_t addr, uint32_t n = ct.num_gated) {
    for (uint32_t t = 0; t < n; t++) {
        const uint32_t noc_x = kernel_compile_time_args[ct.target_base + 2 * t];
        const uint32_t noc_y = kernel_compile_time_args[ct.target_base + 2 * t + 1];
        noc_semaphore_set_remote(addr, get_noc_addr(noc_x, noc_y, addr));
    }
}

#ifdef CMBF2D_DYNAMIC_ORDER
// The steps already released, one bit each; a step id is the routed expert's predicted position for it.
uint64_t released_steps = 0;

// The next step to release, blocking until one is complete or has a chunk out. Lowest step first among equals,
// so a correct prediction releases exactly as before.
uint32_t pick_next_step(
    volatile tt_l1_ptr uint32_t* counts, volatile tt_l1_ptr uint32_t* chunk_counts, uint32_t max_chunks) {
    while (true) {
        invalidate_l1_cache();
        for (uint32_t s = 0; s < ct.num_steps; s++) {
            if (!(released_steps & (1ull << s)) && counts[s] >= ct.num_writers) {
                return s;
            }
        }
        for (uint32_t s = 0; s < ct.num_steps; s++) {
            if (released_steps & (1ull << s)) {
                continue;
            }
            for (uint32_t c = 0; c < max_chunks; c++) {
                if (chunk_counts[s * max_chunks + c] >= ct.num_writers) {
                    return s;
                }
            }
        }
    }
}

uint32_t order_word_addr(uint32_t position) {
    return static_cast<uint32_t>(get_semaphore(CMBF2D_DYNAMIC_ORDER + position / 4));
}

// Record locally that release position `position` is step `step`.
void record_release(uint32_t position, uint32_t step) {
    released_steps |= 1ull << step;
    volatile tt_l1_ptr uint32_t* word = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(order_word_addr(position));
    *word = *word | ((step + 1) << (8 * (position % 4)));
}

// Release `s` at position `k`, and if it is complete, every other complete step at the positions after it.
// Returns how many positions went out. Landed before `progress` or `ready` can name them, so a waiting core
// that sees either finds the steps already there.
uint32_t publish_releases(uint32_t k, uint32_t s, volatile tt_l1_ptr uint32_t* counts) {
    record_release(k, s);
    uint32_t n = 1;
    if (counts[s] >= ct.num_writers) {
        for (uint32_t t = 0; t < ct.num_steps && k + n < ct.num_steps; t++) {
            if (!(released_steps & (1ull << t)) && counts[t] >= ct.num_writers) {
                record_release(k + n, t);
                n++;
            }
        }
    }
    for (uint32_t w = k / 4; w <= (k + n - 1) / 4; w++) {
        publish_to_targets(order_word_addr(4 * w), ct.num_targets);
    }
    noc_async_write_barrier();
    return n;
}
#endif

void kernel_main() {
    volatile tt_l1_ptr uint32_t* counts = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ct.counts_addr);
    for (uint32_t s = 0; s < ct.num_steps; s++) {
        counts[s] = 0;
    }
#ifdef CMBF2D_PROGRESS_SEM
    // Per-(step, chunk) report counts and chunk sizes from the routed expert's writers; the layout mirrors
    // hybrid_expert_done.hpp. A step's chunks are released before the whole step is, in whatever order they
    // finish, as `progress` = step << 24 | chunk size << 16 | done-chunk mask: the untilizers take a tile-row
    // once its chunk's bit or the whole step covers it.
    constexpr uint32_t CHUNK_AREA_OFFSET = 1024;
    constexpr uint32_t MAX_CHUNKS = 16;
    constexpr uint32_t MAX_STEPS = 64;
    volatile tt_l1_ptr uint32_t* chunk_counts =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ct.counts_addr + CHUNK_AREA_OFFSET);
    volatile tt_l1_ptr uint32_t* chunk_rows = chunk_counts + MAX_STEPS * MAX_CHUNKS;
    for (uint32_t i = 0; i < MAX_STEPS * MAX_CHUNKS; i++) {
        chunk_counts[i] = 0;
    }
    const uint32_t progress_addr = static_cast<uint32_t>(get_semaphore(CMBF2D_PROGRESS_SEM));
    volatile tt_l1_ptr uint32_t* progress = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(progress_addr);
#endif
    // The word past the array is the source of the `go` multicast.
    const uint32_t go_src_addr = ct.counts_addr + ct.num_steps * sizeof(uint32_t);
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(go_src_addr) = 1;
    noc_semaphore_set_multicast(
        go_src_addr, get_noc_multicast_addr(ct.re_x0, ct.re_y0, ct.re_x1, ct.re_y1, ct.go_addr), ct.re_cores);

    // This core's own copy of `ready` is the 4-byte source every remote set is written from.
    const uint32_t ready_addr = static_cast<uint32_t>(get_semaphore(ct.ready_sem));
    volatile tt_l1_ptr uint32_t* ready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ready_addr);

    for (uint32_t k = 0; k < ct.num_steps;) {
#ifdef CMBF2D_DYNAMIC_ORDER
        static_assert(ct.num_steps <= 64 && ct.num_steps <= MAX_STEPS, "one bit and one chunk row per step");
        const uint32_t s = pick_next_step(counts, chunk_counts, MAX_CHUNKS);
        // More than one only when `s` is complete, so the wait below does not run for a batch.
        const uint32_t n = publish_releases(k, s, counts);
#else
        const uint32_t s = k;
        constexpr uint32_t n = 1;
#endif
        invalidate_l1_cache();
#ifdef CMBF2D_PROGRESS_SEM
        uint32_t done_mask = 0;
        while (counts[s] < ct.num_writers) {
            uint32_t newly = 0;
            uint32_t rows = 0;
            for (uint32_t c = 0; s < MAX_STEPS && c < MAX_CHUNKS; c++) {
                if (!(done_mask & (1u << c)) && chunk_counts[s * MAX_CHUNKS + c] >= ct.num_writers) {
                    newly |= 1u << c;
                    rows = chunk_rows[s * MAX_CHUNKS + c];
                }
            }
            if (newly != 0) {
                done_mask |= newly;
                // Named by release position, which is what the untilizers walk.
                *progress = (k << 24) | (rows << 16) | done_mask;
                publish_to_targets(progress_addr);
                noc_async_write_barrier();
            }
            invalidate_l1_cache();
        }
#else
        while (counts[s] < ct.num_writers) {
            invalidate_l1_cache();
        }
#endif
        *ready = k + n;
        publish_to_targets(ready_addr);
        // Landed before the source word changes, and so a later, larger value can never be overtaken.
        noc_async_write_barrier();
        k += n;
    }
}
