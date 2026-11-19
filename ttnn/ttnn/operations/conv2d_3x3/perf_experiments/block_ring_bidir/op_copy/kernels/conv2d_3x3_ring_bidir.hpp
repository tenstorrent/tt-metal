// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// block_ring_bidir (perf experiment): the N-slice ring reduce-scatter in two directions.
//
// Every owner's K sum is split into two chains that both end at the owner. Chain A moves one column on per hop along
// +x (the reader, NoC0) through the n_a columns left of the owner; chain B moves one column on per hop along -x (the
// writer, NoC1) through the n_b columns right of it (n_a + n_b = k_cols - 1). A core's steps alternate the two chains,
// so a step's incoming partial was sent two steps earlier by its neighbour (one step on the one-direction ring), and
// each chain is about k_cols / 2 hops. Step j of column c:
//   j == k_cols - 1: the own slice, packed with L1 accumulation onto chain B's final partial (slot 0);
//   else j % 2 == k_cols % 2: chain A, position j / 2, owner c + n_a - j / 2;
//   else: chain B, position j / 2, owner c - n_b + j / 2.
// Chain A's final partial (sent at the neighbour's step k_cols - 2) lands in slot k_cols, and the combine adds it to
// slot 0 (the late chain is A on every k_cols, so the owner's own step always waits on a partial sent two steps
// earlier). Step j packs into slot k_cols - 1 - j, onto the incoming partial when its chain position is >= 1, and its
// partial goes to the chain neighbour's slot of step j + 2 (slot k_cols - 3 - j), or the owner's slot 0 / k_cols.
// Owners past the last C_out tile are skipped by every core of the row (whole chains), so a real step's incoming
// partial always comes from the neighbour's real step j - 2.

#pragma once

#include <cstdint>

#include "conv2d_3x3_nslice_order.hpp"

namespace conv2d_3x3 {

template <uint32_t k_cols, uint32_t num_owning_cols>
struct Bidir {
    static constexpr uint32_t CH_A = 0;
    static constexpr uint32_t CH_B = 1;
    static constexpr uint32_t CH_OWN = 2;
    static constexpr uint32_t n_a = k_cols / 2;
    static constexpr uint32_t n_b = (k_cols - 1) / 2;
    static_assert(k_cols < 2 || (n_a + n_b + 1 == k_cols && n_a >= 1));
    static constexpr uint32_t slots = k_cols + 1;
    static constexpr uint32_t late_slot = k_cols;

    static constexpr uint32_t chain(uint32_t j) {
        return j + 1 == k_cols ? CH_OWN : ((j % 2) == (k_cols % 2) ? CH_A : CH_B);
    }
    static constexpr uint32_t pos(uint32_t j) { return j / 2; }
    // Row-group order key (ring_group_row0's `owner`: only its parity is read): constant per chain, so the two
    // columns a partial moves between walk its groups alike, and a core's consecutive steps alternate it (A, B, A, ...,
    // and the own step, which takes chain B's final partial, after the last A step), so each step starts at the
    // subblock height the previous one ended on.
    static constexpr uint32_t order(uint32_t j) { return chain(j) == CH_A ? 0 : 1; }
    static uint32_t owner(uint32_t my_col, uint32_t j) {
        const uint32_t ch = chain(j);
        if (ch == CH_OWN) {
            return my_col;
        }
        if (ch == CH_A) {
            return (my_col + n_a - pos(j)) % k_cols;
        }
        return (my_col + k_cols - (n_b - pos(j))) % k_cols;
    }
    static bool real(uint32_t my_col, uint32_t j) { return owner(my_col, j) < num_owning_cols; }
    // Step j packs onto an incoming partial (A / B: chain position >= 1; own: chain B's final, when B has a column).
    static constexpr bool incoming(uint32_t j) { return chain(j) == CH_OWN ? n_b >= 1 : pos(j) >= 1; }
    static constexpr uint32_t slot(uint32_t j) { return k_cols - 1 - j; }
    // The slot the chain neighbour's step j + 2 (or the owner) holds step j's partial in.
    static constexpr uint32_t dst_slot(uint32_t j) { return j + 3 <= k_cols ? k_cols - 3 - j : late_slot; }
    static constexpr bool on_chain(uint32_t j, bool chain_a) { return chain_a ? chain(j) == CH_A : chain(j) != CH_A; }
    // The first real step at or after j on one token chain (A: the reader's; B: the writer's, own step included);
    // k_cols: none.
    static uint32_t next_real(uint32_t my_col, uint32_t j, bool chain_a) {
        for (; j < k_cols; ++j) {
            if (on_chain(j, chain_a) && real(my_col, j)) {
                return j;
            }
        }
        return k_cols;
    }
};

}  // namespace conv2d_3x3

#ifndef COMPILE_FOR_TRISC
#include "api/dataflow/dataflow_api.h"
#include "api/semaphore.h"

namespace conv2d_3x3 {

// One token chain's data-movement half (reader: A, NoC0, to column + 1; writer: B, NoC1, to column - 1), as a
// non-blocking state machine: poll() makes every step it can without waiting, so the writer can run it from inside
// its own waits. Units, in compute's push order (one token each): [lead], the groups of every real step on this chain,
// [late] (A, owning column: the combine's gate on chain A's final partial). Each unit's token is popped once the next
// unit's incoming partial has landed (the token CB holds one page, so that gates the packer); a sending unit's group
// goes to the neighbour's slot first, with up to two writes in flight, one transaction id each, and bumps the
// neighbour's arrival count once landed.
template <
    uint32_t k_cols,
    uint32_t num_owning_cols,
    bool chain_a,
    uint32_t block_m_tiles,
    uint32_t ring_group_rows,
    bool rt_valid>
struct BidirChain {
    using BD = Bidir<k_cols, num_owning_cols>;
    static constexpr uint32_t END = k_cols + 1;
    static constexpr uint32_t LATE = k_cols;
    static constexpr uint32_t inflight = 2;
    enum State : uint8_t { IDLE, LEAD, TOKEN, RELEASE, DONE };

    uint32_t cb_token;
    uint32_t my_col;
    uint32_t nx, ny;
    uint32_t credit_sem_id;
    uint32_t arrival_sem_id;
    uint32_t trid_base;
    uint32_t base;
    uint32_t slot_bytes;
    uint32_t row_bytes;
    bool own_real;
    // Per M-block.
    State state = IDLE;
    uint32_t m_block_idx = 0;
    uint32_t walk_rows = 0;
    uint32_t groups = 0;
    uint32_t j = END;
    uint32_t g = 0;
    bool credited = false;
    bool sent = false;
    // Whole kernel.
    uint32_t need = 0;
    uint32_t issued = 0;
    uint32_t retired = 0;

    FORCE_INLINE uint32_t inc(uint32_t unit_j) const {
        if (unit_j == END) {
            return 0;
        }
        if (unit_j == LATE) {
            return groups;
        }
        return BD::incoming(unit_j) ? 1 : 0;
    }
    FORCE_INLINE uint32_t first_unit() const {
        const uint32_t s = BD::next_real(my_col, 0, chain_a);
        if (s < k_cols) {
            return s;
        }
        return (chain_a && own_real) ? LATE : END;
    }
    // Advances (j, g) to the next unit.
    FORCE_INLINE void advance() {
        if (j < k_cols && g + 1 < groups) {
            ++g;
            return;
        }
        g = 0;
        if (j < k_cols) {
            const uint32_t s = BD::next_real(my_col, j + 1, chain_a);
            j = s < k_cols ? s : ((chain_a && own_real) ? LATE : END);
        } else {
            j = END;
        }
    }

    void begin(uint32_t m, uint32_t rows) {
        m_block_idx = m;
        walk_rows = rows;
        groups = (rows + ring_group_rows - 1) / ring_group_rows;
        j = first_unit();
        g = 0;
        credited = false;
        sent = false;
        if (j == END) {
            state = DONE;
        } else if (inc(j) > 0) {
            // Compute pushes a lead token, freed once the first unit's incoming partial has landed.
            state = LEAD;
        } else {
            state = TOKEN;
        }
    }

    FORCE_INLINE void try_retire() {
        if (retired == issued || !noc_cmd_buf_ready(noc_index, write_cmd_buf)) {
            return;
        }
        if (ncrisc_noc_nonposted_write_with_transaction_id_flushed(noc_index, trid_base + retired % inflight)) {
            Noc noc;
            Semaphore<>(arrival_sem_id).up(noc, nx, ny, 1);
            ++retired;
        }
    }

    FORCE_INLINE bool arrived() const {
        invalidate_l1_cache();
        return *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(arrival_sem_id)) >= need;
    }

    // Issues the current unit's group to the neighbour; false when it must wait (credit, in-flight writes).
    FORCE_INLINE bool try_send() {
        if (!credited) {
            if (Semaphore<>(credit_sem_id).value() < m_block_idx + 1) {
                return false;
            }
            credited = true;
        }
        if (issued - retired >= inflight) {
            return false;
        }
        const uint32_t trid = trid_base + issued % inflight;
        const uint32_t key = BD::order(j);
        const uint32_t r0 = rt_valid ? ring_group_row0_rt<ring_group_rows>(walk_rows, key, g)
                                     : ring_group_row0<block_m_tiles, ring_group_rows>(key, g);
        const uint32_t rows = walk_rows - r0 < ring_group_rows ? walk_rows - r0 : ring_group_rows;
        const uint32_t group_bytes = rows * row_bytes;
        const uint32_t src = base + BD::slot(j) * slot_bytes + r0 * row_bytes;
        const uint32_t dst = base + BD::dst_slot(j) * slot_bytes + r0 * row_bytes;
        for (uint32_t off = 0; off < group_bytes; off += NOC_MAX_BURST_SIZE) {
            const uint32_t bytes = group_bytes - off < NOC_MAX_BURST_SIZE ? group_bytes - off : NOC_MAX_BURST_SIZE;
            noc_async_write_one_packet_with_trid(src + off, get_noc_addr(nx, ny, dst + off), bytes, trid);
        }
        noc_async_write_set_trid(0);
        ++issued;
        return true;
    }

    // Every step it can make now.
    void poll() {
        while (true) {
            try_retire();
            switch (state) {
                case IDLE:
                case DONE: return;
                case LEAD:
                    if (!cb_pages_available_at_front(cb_token, 1)) {
                        return;
                    }
                    need += inc(j);
                    state = RELEASE;
                    break;
                case TOKEN: {
                    if (!sent) {
                        if (!cb_pages_available_at_front(cb_token, 1)) {
                            return;
                        }
                        if (j < k_cols && BD::chain(j) != BD::CH_OWN) {
                            if (!try_send()) {
                                return;
                            }
                        }
                        sent = true;
                    }
                    sent = false;
                    advance();
                    need += inc(j);
                    state = RELEASE;
                    break;
                }
                case RELEASE:
                    if (!arrived()) {
                        return;
                    }
                    cb_pop_front(cb_token, 1);
                    state = j == END ? DONE : TOKEN;
                    break;
            }
        }
    }
    bool done() const { return state == DONE && retired == issued; }
    void finish() {
        while (!done()) {
            poll();
        }
    }
};

}  // namespace conv2d_3x3
#endif
