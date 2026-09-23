// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// scatter_offload experiment (perf_experiments/scatter_offload): the bank_coalesced load_block
// geometry shared by NCRISC (tilize_reader.cpp) and BRISC (tilize_writer.cpp), so either RISC-V
// can issue the per-bank DRAM reads and / or the NoC-loopback scatter of any subset of a run's
// banks. Same staging layout and destination positions as the op's read_bank_coalesced.
//
// Mode (define SO_MODE, injected by the test harness only when the host took bank_coalesced):
//   0  baseline: the op's read_bank_coalesced, BRISC only stores.
//   1  NCRISC reads every bank; the scatter is split by rotated bank ordinal: NCRISC scatters
//      ordinals [0, SO_NC_NUM / SO_DEN * banks), BRISC the rest (NoC1 loopback reads).
//   2  as 1 with SO_NC_NUM = 0: BRISC scatters everything, NCRISC only reads DRAM + pushes.
//   3  split coalesced reader: BRISC also READS its bank ordinals (NoC1 DRAM reads into the same
//      staging slot) and scatters exactly those; NCRISC reads + scatters the rest.
// Handshake (two program semaphores, local L1 words, monotonic unit counters):
//   SO_SEM_STAGED  NCRISC -> BRISC: units whose cb_input_sticks slot is reserved (and, modes 1/2,
//                  whose staging has landed).
//   SO_SEM_DONE    BRISC -> NCRISC: units whose BRISC share of the scatter has landed.
// NCRISC stays cb_input_sticks' only producer (reserve / push). BRISC never blocks on a CB: it
// polls for staged units and for output quanta, so draining cb_output_tiles can never be starved
// by the scatter wait (deadlock-free).

#pragma once

#include <cstdint>

#include "tilize_stick_reads.hpp"

#ifndef SO_MODE
#define SO_MODE 0
#endif
#ifndef SO_NC_NUM
#define SO_NC_NUM 1
#endif
#ifndef SO_DEN
#define SO_DEN 2
#endif
#ifndef SO_SEM_STAGED
#define SO_SEM_STAGED 0
#endif
#ifndef SO_SEM_DONE
#define SO_SEM_DONE 1
#endif

namespace so {

constexpr uint32_t nc_num = SO_MODE == 2 ? 0 : SO_NC_NUM;
constexpr uint32_t den = SO_DEN;

// Bank j of a run of `count` sticks: count / NB sticks, one more for j < count % NB; its sticks
// sit in staging after those of banks 0 .. j - 1.
template <uint32_t num_banks>
struct Run {
    uint32_t banks, q, rem;
    explicit Run(uint32_t count) :
        banks(count < num_banks ? count : num_banks), q(count / num_banks), rem(count % num_banks) {}
    uint32_t sticks(uint32_t j) const { return q + (j < rem ? 1 : 0); }
    uint32_t offset(uint32_t j) const { return j * q + (j < rem ? j : rem); }
};

// fn(first tile-row, tile-rows in run, unit position of the run's first tile-row) for the runs of
// consecutive tile-rows of the next `n` walk positions.
template <uint32_t block_width, typename Fn>
FORCE_INLINE void for_each_run(tilize_dataflow::Walker<block_width>& w, uint32_t n, Fn&& fn) {
    uint32_t p = 0;
    while (p < n) {
        const uint32_t run_row = w.row();
        uint32_t run_len = 1;
        w.advance();
        while (p + run_len < n && w.row() == run_row + run_len) {
            ++run_len;
            w.advance();
        }
        fn(run_row, run_len, p);
        p += run_len;
    }
}

template <
    uint32_t block_width,
    uint32_t tile_h,
    uint32_t stick_bytes,
    uint32_t stick_page_bytes,
    uint32_t num_banks,
    uint32_t lo,  // bank ordinals [banks * lo / den, banks * hi / den) of every run
    uint32_t hi>
struct Share {
    static_assert(lo <= hi && hi <= den, "ordinal share");
    static FORCE_INLINE uint32_t first(uint32_t banks) { return banks * lo / den; }
    static FORCE_INLINE uint32_t last(uint32_t banks) { return banks * hi / den; }

    // Per-bank DRAM reads of this share of unit (next n positions of w) into staging `stage`.
    template <typename Accessor>
    static FORCE_INLINE void issue(
        const Accessor& accessor, tilize_dataflow::Walker<block_width>& w, uint32_t n, uint32_t stage, uint32_t rot) {
        for_each_run(w, n, [&](uint32_t run_row, uint32_t run_len, uint32_t p) {
            if constexpr (lo == hi) {
                return;
            }
            const uint32_t first_stick = run_row * tile_h;
            const Run<num_banks> run(run_len * tile_h);
            const uint32_t run_stage = stage + p * tile_h * stick_page_bytes;
            const uint32_t b0 = first(run.banks), b1 = last(run.banks);
            uint32_t j = (rot % run.banks + b0) % run.banks;
            for (uint32_t b = b0; b < b1; ++b) {
                noc_async_read(
                    accessor.get_noc_addr(first_stick + j),
                    run_stage + run.offset(j) * stick_page_bytes,
                    run.sticks(j) * stick_page_bytes);
                if (++j == run.banks) {
                    j = 0;
                }
            }
        });
    }

    // Loopback scatter of this share: one-packet reads staging -> tilize position in `slot`.
    // The caller has set the read command buffer's one-packet state (this core, stick_bytes).
    static FORCE_INLINE void scatter(
        tilize_dataflow::Walker<block_width>& w, uint32_t n, uint32_t stage, uint32_t slot, uint32_t rot) {
        for_each_run(w, n, [&](uint32_t, uint32_t run_len, uint32_t p) {
            if constexpr (lo == hi) {
                return;
            }
            const Run<num_banks> run(run_len * tile_h);
            const uint32_t run_stage = stage + p * tile_h * stick_page_bytes;
            const uint32_t b0 = first(run.banks), b1 = last(run.banks);
            uint32_t j = (rot % run.banks + b0) % run.banks;
            for (uint32_t b = b0; b < b1; ++b) {
                uint32_t src = run_stage + run.offset(j) * stick_page_bytes;
                uint32_t dst = slot + (p * tile_h + j) * stick_bytes;
                for (uint32_t i = run.sticks(j); i > 0; --i) {
                    noc_async_read_one_packet_with_state(src, dst);
                    src += stick_page_bytes;
                    dst += num_banks * stick_bytes;
                }
                if (++j == run.banks) {
                    j = 0;
                }
            }
        });
    }
};

FORCE_INLINE volatile tt_l1_ptr uint32_t* sem_ptr(uint32_t id) {
    return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(id));
}
FORCE_INLINE uint32_t sem_read(volatile tt_l1_ptr uint32_t* p) {
    invalidate_l1_cache();
    return *p;
}

}  // namespace so
