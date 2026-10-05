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
// Unicast rather than one multicast: the waiting cores are scattered over rows 0-1, and a rectangle over them
// would also write into cores that are not part of this program. The routed expert's cores ARE a rectangle,
// which is what lets `go` be one multicast.
//
// Compile-time args:
//   [0] routed-expert writers      [1] steps (slots walked per pass x passes)
//   [2] count array address        [3] ready semaphore id
//   [4] go address                 [5..8] routed-expert rectangle (noc x0, y0, x1, y1)
//   [9] routed-expert cores        [10] waiting cores N     [11 ..) N x (noc_x, noc_y)

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t num_writers = get_compile_time_arg_val(0);
    constexpr uint32_t num_steps = get_compile_time_arg_val(1);
    constexpr uint32_t counts_addr = get_compile_time_arg_val(2);
    constexpr uint32_t ready_sem = get_compile_time_arg_val(3);
    constexpr uint32_t go_addr = get_compile_time_arg_val(4);
    constexpr uint32_t re_x0 = get_compile_time_arg_val(5);
    constexpr uint32_t re_y0 = get_compile_time_arg_val(6);
    constexpr uint32_t re_x1 = get_compile_time_arg_val(7);
    constexpr uint32_t re_y1 = get_compile_time_arg_val(8);
    constexpr uint32_t re_cores = get_compile_time_arg_val(9);
    constexpr uint32_t num_targets = get_compile_time_arg_val(10);
    constexpr uint32_t targets_base = 11;

    volatile tt_l1_ptr uint32_t* counts = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(counts_addr);
    for (uint32_t s = 0; s < num_steps; s++) {
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
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(counts_addr + CHUNK_AREA_OFFSET);
    volatile tt_l1_ptr uint32_t* chunk_rows = chunk_counts + MAX_STEPS * MAX_CHUNKS;
    for (uint32_t i = 0; i < MAX_STEPS * MAX_CHUNKS; i++) {
        chunk_counts[i] = 0;
    }
    const uint32_t progress_addr = static_cast<uint32_t>(get_semaphore(CMBF2D_PROGRESS_SEM));
    volatile tt_l1_ptr uint32_t* progress = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(progress_addr);
#endif
    // The word past the array is the source of the `go` multicast.
    const uint32_t go_src_addr = counts_addr + num_steps * sizeof(uint32_t);
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(go_src_addr) = 1;
    noc_semaphore_set_multicast(go_src_addr, get_noc_multicast_addr(re_x0, re_y0, re_x1, re_y1, go_addr), re_cores);

    // This core's own copy of `ready` is the 4-byte source every remote set is written from.
    const uint32_t ready_addr = static_cast<uint32_t>(get_semaphore(ready_sem));
    volatile tt_l1_ptr uint32_t* ready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ready_addr);

    for (uint32_t s = 0; s < num_steps; s++) {
        invalidate_l1_cache();
#ifdef CMBF2D_PROGRESS_SEM
        uint32_t done_mask = 0;
        while (counts[s] < num_writers) {
            uint32_t newly = 0;
            uint32_t rows = 0;
            for (uint32_t c = 0; s < MAX_STEPS && c < MAX_CHUNKS; c++) {
                if (!(done_mask & (1u << c)) && chunk_counts[s * MAX_CHUNKS + c] >= num_writers) {
                    newly |= 1u << c;
                    rows = chunk_rows[s * MAX_CHUNKS + c];
                }
            }
            if (newly != 0) {
                done_mask |= newly;
                *progress = (s << 24) | (rows << 16) | done_mask;
                for (uint32_t t = 0; t < num_targets; t++) {
                    const uint32_t noc_x = kernel_compile_time_args[targets_base + 2 * t];
                    const uint32_t noc_y = kernel_compile_time_args[targets_base + 2 * t + 1];
                    noc_semaphore_set_remote(progress_addr, get_noc_addr(noc_x, noc_y, progress_addr));
                }
                noc_async_write_barrier();
            }
            invalidate_l1_cache();
        }
#else
        while (counts[s] < num_writers) {
            invalidate_l1_cache();
        }
#endif
        *ready = s + 1;
        for (uint32_t t = 0; t < num_targets; t++) {
            const uint32_t noc_x = kernel_compile_time_args[targets_base + 2 * t];
            const uint32_t noc_y = kernel_compile_time_args[targets_base + 2 * t + 1];
            noc_semaphore_set_remote(ready_addr, get_noc_addr(noc_x, noc_y, ready_addr));
        }
        // Landed before the source word changes, and so a later, larger value can never be overtaken.
        noc_async_write_barrier();
    }
}
