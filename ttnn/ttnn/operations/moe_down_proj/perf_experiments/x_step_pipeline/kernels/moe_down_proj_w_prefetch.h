// SPDX-FileCopyrightText: 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// moe_down_proj w prefetch, run by whichever dataflow RISC the host assigns (w_reads_on_reader): the DRAM reads of an
// n-line sender's next owned w step into its staging buffer, issued a quota at a time so the RISC keeps serving the
// channel rounds in between. Staging holds NUM_EXPERTS chunk slots; expert e's k_block_tiles x core_n_tiles chunk
// lands at slot e in [k][n] order, so the sender multicasts the wave's expert range as one contiguous transfer.

#pragma once

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "moe_down_proj_plan.h"

template <
    uint32_t k_block_tiles,
    uint32_t core_n_tiles,
    uint32_t tensor_k_tiles,
    uint32_t num_k_rounds,
    uint32_t w_tile_bytes,
    uint32_t share_den,
    typename Accessor>
struct MoeWPrefetch {
    static constexpr uint32_t chunk_tiles = k_block_tiles * core_n_tiles;
    static constexpr uint32_t chunk_bytes = chunk_tiles * w_tile_bytes;
    static constexpr uint32_t row_pages = MOE_NUM_EXPERTS * tensor_k_tiles;
    static constexpr uint32_t NONE = UINT32_MAX;

    const MoePlan& plan;
    const Accessor& accessor;
    uint32_t n_first;
    // Owned step being prefetched, NONE when this core sends no later step.
    uint32_t step = NONE;
    uint32_t experts[MOE_NUM_EXPERTS];
    uint32_t k_first = 0;
    uint32_t next = 0;
    uint32_t total = 0;
    // This RISC reads the tiles t with share_lo <= t % share_den < share_hi of each step (the host's w_reader_share
    // splits [0, share_den) between the reader and writer RISCs).
    uint32_t share_lo;
    uint32_t share_hi;

    MoeWPrefetch(
        const MoePlan& plan_, const Accessor& accessor_, uint32_t n_first_, uint32_t share_lo_, uint32_t share_hi_) :
        plan(plan_), accessor(accessor_), n_first(n_first_), share_lo(share_lo_), share_hi(share_hi_) {}

    // Target the first owned step >= from.
    template <typename Owns>
    void start(uint32_t from, uint32_t num_steps, Owns owns) {
        next = 0;
        total = 0;
        step = NONE;
        while (from < num_steps && !owns(from)) {
            ++from;
        }
        if (from >= num_steps) {
            return;
        }
        step = from;
        const uint32_t wave = from / num_k_rounds;
        k_first = (from % num_k_rounds) * k_block_tiles;
        for (uint32_t e = 0; e < MOE_NUM_EXPERTS; ++e) {
            uint32_t row;
            if (plan.expert_rows_in(e, plan.wave_start(wave), plan.wave_end(wave), &row) != 0) {
                experts[total / chunk_tiles] = e;
                total += chunk_tiles;
            }
        }
    }

    // Issue up to max_tiles more of this RISC's reads of the targeted step into stage_l1. Tile t: expert
    // experts[t / chunk_tiles], then n, then kk.
    void issue(uint32_t stage_l1, uint32_t max_tiles) {
        for (; next < total && max_tiles > 0; ++next) {
            const uint32_t slot = next % share_den;
            if (slot < share_lo || slot >= share_hi) {
                continue;
            }
            --max_tiles;
            const uint32_t e = experts[next / chunk_tiles];
            const uint32_t n = (next / k_block_tiles) % core_n_tiles;
            const uint32_t kk = next % k_block_tiles;
            const uint32_t page = (n_first + n) * row_pages + e * tensor_k_tiles + k_first + kk;
            noc_async_read_page(page, accessor, stage_l1 + e * chunk_bytes + (kk * core_n_tiles + n) * w_tile_bytes);
        }
    }
};

// The rotating w sender owns steps first, first + period, ... The writer RISC (which runs the w pipes) publishes them
// for the reader RISC's prefetch through a one-page CB. MOE_W_OWNER_SEARCH bounds the rotation period.
constexpr uint32_t MOE_W_OWNER_SEARCH = 256;
constexpr uint32_t MOE_W_OWNER_NONE = UINT32_MAX;

template <uint32_t cb_w_owner, typename Owns>
void moe_publish_w_owner(Owns owns) {
    uint32_t first = 0;
    while (first < MOE_W_OWNER_SEARCH && !owns(first)) {
        ++first;
    }
    uint32_t period = 1;
    while (period < MOE_W_OWNER_SEARCH && !owns(first + period)) {
        ++period;
    }
    cb_reserve_back(cb_w_owner, 1);
    volatile tt_l1_ptr uint32_t* page = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_w_owner));
    page[0] = first < MOE_W_OWNER_SEARCH ? first : MOE_W_OWNER_NONE;
    page[1] = period;
    cb_push_back(cb_w_owner, 1);
}

struct MoeWOwner {
    uint32_t first;
    uint32_t period;
    bool operator()(uint32_t step) const {
        return first != MOE_W_OWNER_NONE && step >= first && (step - first) % period == 0;
    }
};

template <uint32_t cb_w_owner>
MoeWOwner moe_receive_w_owner() {
    cb_wait_front(cb_w_owner, 1);
    volatile tt_l1_ptr uint32_t* page = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_w_owner));
    const MoeWOwner owner{page[0], page[1]};
    cb_pop_front(cb_w_owner, 1);
    return owner;
}
