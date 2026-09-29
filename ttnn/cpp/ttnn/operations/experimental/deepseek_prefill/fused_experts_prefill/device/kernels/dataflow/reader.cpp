// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/tensor_accessor.h"
#include "api/compile_time_args.h"
#include "tools/profiler/kernel_profiler.hpp"

// Reader (DRAM -> L1) for fused_experts_prefill. One instance per worker core; the 8 cores of a group
// walk the same experts in lock-step (expert e is owned by group e % num_groups and every core derives
// the same routing, so the skip decisions agree).
//
// ---- 1. Routing prologue (EVERY core, all T tokens) ----
// Per band of 32 tokens the routing tiles are read into a dedicated scratch CB and, per
// token: the k selected experts are taken from the ids tile, or found by an ascending top-k scan of the
// ranking row; ids >= E and repeats are dropped; the weights are scores / (sum + eps) * scaling rounded
// to bf16 (exactly the decode `compute_expert_ids` arithmetic). For every selected expert owned by this
// group the entry  token | slot << 9 | bf16(weight) << 16  is appended to that expert's list in cb_meta:
//
//   cb_meta (one page, pushed once, never popped):
//     [0, hdr_words)                       counts[jl]   (jl = local expert index, expert g + jl * groups)
//     [hdr_words + jl * T, + counts[jl])   entries of expert jl, in token order
//
// ---- 2. Per owned expert with a non-zero count, per chunk of m_chunk tile rows ----
//   cb_w  <- gate_up slot: this core's `it_pc` gate_up shards (shard ids c*it_pc .. +it_pc).
//   per M-block of m_block tile rows: for each row, the 32 tokens' x rows are gathered from the
//   row-major DRAM tensor in kt / rm_chunk 1 KB-per-token segments into cb_rm (compute tilizes them),
//   then the block's per-row routing weights are pushed as one scalar tile per row (cb_rscal).
//   cb_w  <- down slot, prefetched after the first block (consumed by compute after all gate_up work).
//
// Compile-time args:
//   0 cb_meta  1 cb_rm  2 cb_w  3 cb_rscal
//   4 T  5 E  6 top_k  7 index_is_bf16  8 rank_from_scores  9 scaling_bits  10 eps_bits
//   11 num_groups  12 kt  13 it_pc  14 dn_shards  15 gu_shard_tiles  16 dn_shard_tiles
//   17 w_tile_bytes  18 slot_tiles  19 m_chunk  20 m_block  21 hdr_words  22 rm_chunk_tiles
//   23 row_bytes  24 score_tiles (E / 32 rounded up)  25 tile_bytes (bf16)
//   26 cb_scratch (routing scratch CB)
//   27 tpc  28 ks  29 sem_route  30 grid_x  31 grid_y
//   32 rm_group (row-major segments gathered per barrier = cb_rm capacity in segments)
//   33+ TensorAccessorArgs: x, ids, scores, ranking, gate_up[0], down[0]
// Runtime args:
//   0 x  1 ids  2 scores  3 ranking  (buffer addresses)
//   4 c (core index in group)  5 g (group)  6 n (experts owned by this group)
//   7 .. 7+n-1 gate_up addresses, then n down addresses (owned experts in ascending id order)

namespace {

constexpr uint32_t kFaceBytes = 512;
constexpr uint32_t kFaceRowBytes = 32;

// Byte offset of element (row, col) inside a 32x32 tile of 2-byte elements (4 16x16 faces).
FORCE_INLINE uint32_t tile_elem_offset(uint32_t row, uint32_t col) {
    const uint32_t face = ((row >> 4) << 1) + (col >> 4);
    return (face * kFaceBytes) + ((row & 15u) * kFaceRowBytes) + ((col & 15u) * 2u);
}

FORCE_INLINE float bf16_to_f32(uint16_t v) {
    const uint32_t bits = static_cast<uint32_t>(v) << 16;
    float out;
    __builtin_memcpy(&out, &bits, sizeof(out));
    return out;
}

// Round fp32 to nearest-even bf16 (returned as the 16 bits).
FORCE_INLINE uint32_t f32_to_bf16_rne(float v) {
    uint32_t bits;
    __builtin_memcpy(&bits, &v, sizeof(bits));
    return (bits + 0x7FFFu + ((bits >> 16) & 1u)) >> 16;
}

}  // namespace

void kernel_main() {
    constexpr uint32_t cb_meta_id = get_compile_time_arg_val(0);
    constexpr uint32_t cb_rm_id = get_compile_time_arg_val(1);
    constexpr uint32_t cb_w_id = get_compile_time_arg_val(2);
    constexpr uint32_t cb_rscal_id = get_compile_time_arg_val(3);
    constexpr uint32_t T = get_compile_time_arg_val(4);
    constexpr uint32_t E = get_compile_time_arg_val(5);
    constexpr uint32_t top_k = get_compile_time_arg_val(6);
    constexpr bool index_is_bf16 = get_compile_time_arg_val(7) == 1;
    constexpr bool rank_from_scores = get_compile_time_arg_val(8) == 1;
    constexpr uint32_t scaling_bits = get_compile_time_arg_val(9);
    constexpr uint32_t eps_bits = get_compile_time_arg_val(10);
    constexpr uint32_t num_groups = get_compile_time_arg_val(11);
    constexpr uint32_t kt = get_compile_time_arg_val(12);
    constexpr uint32_t it_pc = get_compile_time_arg_val(13);
    constexpr uint32_t dn_shards = get_compile_time_arg_val(14);
    constexpr uint32_t gu_shard_tiles = get_compile_time_arg_val(15);
    constexpr uint32_t dn_shard_tiles = get_compile_time_arg_val(16);
    constexpr uint32_t w_tile_bytes = get_compile_time_arg_val(17);
    constexpr uint32_t slot_tiles = get_compile_time_arg_val(18);
    constexpr uint32_t m_chunk = get_compile_time_arg_val(19);
    constexpr uint32_t m_block = get_compile_time_arg_val(20);
    constexpr uint32_t hdr_words = get_compile_time_arg_val(21);
    constexpr uint32_t rm_chunk_tiles = get_compile_time_arg_val(22);
    constexpr uint32_t row_bytes = get_compile_time_arg_val(23);
    constexpr uint32_t score_tiles = get_compile_time_arg_val(24);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(25);
    constexpr uint32_t cb_scratch_id = get_compile_time_arg_val(26);
    constexpr uint32_t tpc = get_compile_time_arg_val(27);
    constexpr uint32_t ks = get_compile_time_arg_val(28);
    constexpr uint32_t sem_route_id = get_compile_time_arg_val(29);
    constexpr uint32_t grid_x = get_compile_time_arg_val(30);
    constexpr uint32_t grid_y = get_compile_time_arg_val(31);

    constexpr uint32_t rm_group = get_compile_time_arg_val(32);

    constexpr auto x_args = TensorAccessorArgs<33>();
    constexpr auto ids_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto scores_args = TensorAccessorArgs<ids_args.next_compile_time_args_offset()>();
    constexpr auto rank_args = TensorAccessorArgs<scores_args.next_compile_time_args_offset()>();
    constexpr auto gu_args = TensorAccessorArgs<rank_args.next_compile_time_args_offset()>();
    constexpr auto dn_args = TensorAccessorArgs<gu_args.next_compile_time_args_offset()>();

    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t ids_addr = get_arg_val<uint32_t>(1);
    const uint32_t scores_addr = get_arg_val<uint32_t>(2);
    const uint32_t rank_addr = get_arg_val<uint32_t>(3);
    const uint32_t c = get_arg_val<uint32_t>(4);
    const uint32_t g = get_arg_val<uint32_t>(5);
    const uint32_t n_owned = get_arg_val<uint32_t>(6);
    const uint32_t cid = get_arg_val<uint32_t>(7);  // flat core index over the whole grid
    constexpr uint32_t kNocXRt = 8;                 // grid_x NoC x coordinates, then grid_y NoC y coordinates
    constexpr uint32_t kAddrRt = kNocXRt + grid_x + grid_y;

    constexpr uint32_t seg_bytes = rm_chunk_tiles * 32 * 2;  // one row-major segment of one token row
    constexpr uint32_t rm_chunks = kt / rm_chunk_tiles;
    static_assert(top_k >= 1 && top_k <= 16, "top_k must be in [1, 16]");

    Noc noc;
    CircularBuffer cb_meta(cb_meta_id);
    CircularBuffer cb_rm(cb_rm_id);
    CircularBuffer cb_w(cb_w_id);
    CircularBuffer cb_rscal(cb_rscal_id);
    CircularBuffer cb_scratch(cb_scratch_id);

    const auto x_acc = TensorAccessor(x_args, x_addr, row_bytes);

    // ======================= 1. routing prologue =======================
    cb_meta.reserve_back(1);
    const uint32_t meta_l1 = cb_meta.get_write_ptr();
    CoreLocalMem<volatile uint32_t> counts(meta_l1);
    CoreLocalMem<volatile uint32_t> entries(meta_l1 + hdr_words * 4);
    for (uint32_t jl = 0; jl < n_owned; ++jl) {
        counts[jl] = 0;
    }

    constexpr uint32_t gu_shard_bytes = gu_shard_tiles * w_tile_bytes;
    constexpr uint32_t dn_shard_bytes = dn_shard_tiles * w_tile_bytes;

    auto read_gate_up = [&](uint32_t jl) {
        const auto gu = TensorAccessor(gu_args, get_arg_val<uint32_t>(kAddrRt + jl));
        ShardView gu_shard(gu);
        for (uint32_t s = 0; s < it_pc; ++s) {
            noc.async_read(
                gu_shard, cb_w, gu_shard_bytes, {.shard_id = c * it_pc + s}, {.offset_bytes = s * gu_shard_bytes});
        }
    };

    // The first owned expert's gate_up slot is prefetched while the routing runs. It is used if that expert
    // turns out to have tokens. The routing's read barriers after the issue guarantee it has landed by
    // the time the prologue is done (the main loop barriers on it explicitly).
    bool gu_prefetched = false;
    auto issue_prefetch = [&]() {
        if (n_owned > 0 && !gu_prefetched) {
            cb_w.reserve_back(slot_tiles);
            read_gate_up(0);
            gu_prefetched = true;
        }
    };

    {
        DeviceZoneScopedN("ROUTING");
        // ---- Scratch layout (dedicated CB, identical on every core) ----
        //   [route: T * ks words (the routing of every token)] [send: this core's tokens] [ids tile]
        //   [score tiles] [ranking tiles] [owned-expert byte table (E bytes)]
        constexpr uint32_t route_bytes = (T * ks * 4 + 63u) / 64u * 64u;
        constexpr uint32_t send_bytes_max = (tpc * ks * 4 + 63u) / 64u * 64u;
        const uint32_t scratch = cb_scratch.get_write_ptr();
        const uint32_t route_l1 = scratch;
        const uint32_t send_l1 = route_l1 + route_bytes;
        const uint32_t ids_l1 = send_l1 + send_bytes_max;
        const uint32_t sc_l1 = ids_l1 + tile_bytes;
        const uint32_t rk_l1 = sc_l1 + score_tiles * tile_bytes;
        const uint32_t tab_l1 = rk_l1 + score_tiles * tile_bytes;
        CoreLocalMem<volatile uint32_t> route(route_l1);
        CoreLocalMem<volatile uint32_t> send(send_l1);
        CoreLocalMem<volatile uint16_t> ids_mem(ids_l1);
        CoreLocalMem<volatile uint16_t> sc_mem(sc_l1);
        CoreLocalMem<volatile uint32_t> rk32(rk_l1);
        CoreLocalMem<volatile uint8_t> owned(tab_l1);
        (void)ids_mem;
        (void)rk32;

        // owned[e] = local index jl of expert e if this group owns it (e % num_groups == g), else 0xFF;
        // built with a running counter (no integer division on the RISC-V).
        {
            uint32_t gi = 0;
            uint32_t jli = 0;
            for (uint32_t e = 0; e < E; ++e) {
                owned[e] = (gi == g) ? static_cast<uint8_t>(jli) : static_cast<uint8_t>(0xFF);
                if (++gi == num_groups) {
                    gi = 0;
                    ++jli;
                }
            }
        }

        const auto ids_acc = TensorAccessor(ids_args, ids_addr, tile_bytes);
        const auto scores_acc = TensorAccessor(scores_args, scores_addr, tile_bytes);
        const auto rank_acc = TensorAccessor(rank_args, rank_addr, tile_bytes);
        (void)ids_acc;
        (void)rank_acc;

        constexpr float scaling = __builtin_bit_cast(float, scaling_bits);
        constexpr float eps = __builtin_bit_cast(float, eps_bits);

        // ---- A. Route this core's tokens [t0, t1): tpc tokens per core, spread over all cores. ----
        const uint32_t t0 = (cid * tpc) < T ? cid * tpc : T;
        const uint32_t t1 = (t0 + tpc) < T ? t0 + tpc : T;
        {
            DeviceZoneScopedN("ROUTE_LOCAL");
            uint32_t cur_band = 0xFFFFFFFFu;
            for (uint32_t token = t0; token < t1; ++token) {
                const uint32_t band = token >> 5;
                const uint32_t b = token & 31u;
                if (band != cur_band) {
                    cur_band = band;
                    if constexpr (!rank_from_scores) {
                        noc.async_read(ids_acc, CoreLocalMem<uint32_t>(ids_l1), tile_bytes, {.page_id = band}, {});
                    }
                    for (uint32_t p = 0; p < score_tiles; ++p) {
                        noc.async_read(
                            scores_acc,
                            CoreLocalMem<uint32_t>(sc_l1 + p * tile_bytes),
                            tile_bytes,
                            {.page_id = band * score_tiles + p},
                            {});
                        if constexpr (rank_from_scores) {
                            noc.async_read(
                                rank_acc,
                                CoreLocalMem<uint32_t>(rk_l1 + p * tile_bytes),
                                tile_bytes,
                                {.page_id = band * score_tiles + p},
                                {});
                        }
                    }
                    noc.async_read_barrier();
                    // Overlap the first weight slot with the (long) ranking below.
                    issue_prefetch();
                }

                uint32_t sel[top_k];

                if constexpr (rank_from_scores) {
                    // Ascending scan keeping a sorted running top-k (strict >: of equal scores the
                    // lower expert id is kept). Scores are compared as order-preserving integer keys
                    // (always > 0 for a real bf16), so the zero-initialised running list fills with
                    // the first top_k experts and afterwards `thr` (the k-th best key) is the bar.
                    // Row b of a tile is two 16-element face rows (32 B each) read as 32-bit words.
                    uint32_t best_key[top_k];
                    uint32_t best_id[top_k];
                    for (uint32_t j = 0; j < top_k; ++j) {
                        best_key[j] = 0;
                        best_id[j] = 0;
                    }
                    uint32_t thr = 0;
                    auto consider = [&](uint32_t e, uint32_t v) {
                        const uint32_t key = v ^ (0x8000u | ((0u - (v >> 15)) & 0x7FFFu));
                        if (key > thr) {
                            uint32_t p = 0;
                            while (p + 1u < top_k && best_key[p] >= key) {
                                ++p;
                            }
                            for (uint32_t j = top_k - 1u; j > p; --j) {
                                best_key[j] = best_key[j - 1u];
                                best_id[j] = best_id[j - 1u];
                            }
                            best_key[p] = key;
                            best_id[p] = e;
                            thr = best_key[top_k - 1u];
                        }
                    };
                    const uint32_t row_off = ((b >> 4) * 1024u) + ((b & 15u) * 32u);
                    for (uint32_t p = 0; p < score_tiles; ++p) {
                        for (uint32_t f = 0; f < 2; ++f) {
                            const uint32_t base_e = p * 32u + f * 16u;
                            if (base_e >= E) {
                                break;
                            }
                            const uint32_t w0 = (p * tile_bytes + f * kFaceBytes + row_off) >> 2;
                            if (base_e + 16u <= E) {
                                for (uint32_t w = 0; w < 8; ++w) {
                                    const uint32_t word = rk32[w0 + w];
                                    consider(base_e + 2u * w, word & 0xFFFFu);
                                    consider(base_e + 2u * w + 1u, word >> 16);
                                }
                            } else {
                                for (uint32_t w = 0; w < 8; ++w) {
                                    const uint32_t e = base_e + 2u * w;
                                    if (e >= E) {
                                        break;
                                    }
                                    const uint32_t word = rk32[w0 + w];
                                    consider(e, word & 0xFFFFu);
                                    if (e + 1u < E) {
                                        consider(e + 1u, word >> 16);
                                    }
                                }
                            }
                        }
                    }
                    for (uint32_t j = 0; j < top_k; ++j) {
                        sel[j] = best_id[j];
                    }
                } else {
                    for (uint32_t j = 0; j < top_k; ++j) {
                        const uint16_t raw = ids_mem[tile_elem_offset(b, j) >> 1];
                        uint32_t e;
                        if constexpr (index_is_bf16) {
                            e = static_cast<uint32_t>(bf16_to_f32(raw));
                        } else {
                            e = raw;
                        }
                        // Out-of-range ids and repeats of an id this token already picked are dropped
                        // (parked on the sentinel E), like the decode kernel.
                        bool drop = e >= E;
                        for (uint32_t p = 0; p < j && !drop; ++p) {
                            drop = sel[p] == e;
                        }
                        sel[j] = drop ? E : e;
                    }
                }

                // Weights: the selected UNBIASED scores renormalised to sum to 1 and scaled.
                float w[top_k];
                float sum = 0.0f;
                for (uint32_t j = 0; j < top_k; ++j) {
                    if (sel[j] >= E) {
                        w[j] = 0.0f;
                        continue;
                    }
                    const uint32_t e = sel[j];
                    const uint32_t off = ((e >> 5) * tile_bytes) + tile_elem_offset(b, e & 31u);
                    w[j] = bf16_to_f32(sc_mem[off >> 1]);
                    sum += w[j];
                }
                const float inv = scaling / (sum + eps);

                // Routing word: expert id (0xFFFF = dropped) | bf16 weight << 16.
                for (uint32_t j = 0; j < top_k; ++j) {
                    const uint32_t e = sel[j];
                    send[(token - t0) * ks + j] = (e >= E) ? 0xFFFFu : (e | (f32_to_bf16_rne(w[j] * inv) << 16));
                }
            }
        }
        issue_prefetch();  // cores without tokens

        // ---- B. Exchange: every core gets every token's routing. ----
        {
            DeviceZoneScopedN("ROUTE_EXCHANGE");
            Semaphore<> sem_route(sem_route_id);
            if (t1 > t0) {
                const uint32_t bytes = (t1 - t0) * ks * 4;
                for (uint32_t dy = 0; dy < grid_y; ++dy) {
                    for (uint32_t dx = 0; dx < grid_x; ++dx) {
                        noc.async_write(
                            CoreLocalMem<uint32_t>(send_l1),
                            UnicastEndpoint{},
                            bytes,
                            {.offset_bytes = 0},
                            {.noc_x = get_arg_val<uint32_t>(kNocXRt + dx),
                             .noc_y = get_arg_val<uint32_t>(kNocXRt + grid_x + dy),
                             .addr = route_l1 + t0 * ks * 4});
                    }
                }
                noc.async_write_barrier();
            }
            for (uint32_t dy = 0; dy < grid_y; ++dy) {
                for (uint32_t dx = 0; dx < grid_x; ++dx) {
                    sem_route.up(
                        noc, get_arg_val<uint32_t>(kNocXRt + dx), get_arg_val<uint32_t>(kNocXRt + grid_x + dy), 1);
                }
            }
            sem_route.wait_min(grid_x * grid_y);
        }

        // ---- C. Build this group's per-expert token lists from the full routing. ----
        if (n_owned > 0) {
            DeviceZoneScopedN("ROUTE_LISTS");
            for (uint32_t token = 0; token < T; ++token) {
                for (uint32_t j = 0; j < top_k; ++j) {
                    const uint32_t word = route[token * ks + j];
                    const uint32_t e = word & 0xFFFFu;
                    if (e >= E) {
                        continue;
                    }
                    const uint32_t jl = owned[e];
                    if (jl == 0xFFu) {
                        continue;
                    }
                    const uint32_t n = counts[jl];
                    entries[jl * T + n] = token | (j << 9) | (word & 0xFFFF0000u);
                    counts[jl] = n + 1;
                }
            }
        }
    }
    cb_meta.push_back(1);

    // ======================= 2. per-expert streams =======================

    // Gather the (up to 32) token rows of tile row `row_tile` of expert jl and hand them to compute as
    // rm_chunks segments; then push this tile row's per-token routing weights as one scalar tile.
    // The x rows are gathered rm_group segments per barrier: the segments of a group are reserved
    // cumulatively (segment j waits only for j + 1 free segments, so the reader refills the ring while
    // compute is still tilizing the previous group), all their reads are issued, then one barrier and one
    // push per segment. cb_rm holds exactly rm_group segments, and rm_chunks is a multiple of rm_group, so
    // every group starts at the ring base. Many more reads are in flight than with a barrier per segment,
    // which hides the DRAM latency.
    // The zones are recorded for one sampled call only (the profiler keeps ~125 zones per RISC):
    // R_SEG_RESERVE = waiting for cb_rm space (compute has not tilized yet), R_SEG_ISSUE = issuing the
    // row reads, R_SEG_BARRIER = waiting for a group's reads to land.
    static_assert(rm_chunks % rm_group == 0, "cb_rm group must divide the segments of a row");
    uint32_t rows_calls = 0;
    constexpr uint32_t kSampledRowsCall = 6;
    auto read_rows = [&](uint32_t jl, uint32_t count, uint32_t row_tile) {
        const bool sampled = (rows_calls++ == kSampledRowsCall);
        const uint32_t first = row_tile * 32;
        const uint32_t nvalid = (count - first) < 32 ? (count - first) : 32;
        for (uint32_t kc0 = 0; kc0 < rm_chunks; kc0 += rm_group) {
            uint32_t group_base = 0;
            for (uint32_t j = 0; j < rm_group; ++j) {
                if (sampled) {
                    DeviceZoneScopedN("R_SEG_RESERVE");
                    cb_rm.reserve_back((j + 1) * rm_chunk_tiles);
                } else {
                    cb_rm.reserve_back((j + 1) * rm_chunk_tiles);
                }
                if (j == 0) {
                    group_base = cb_rm.get_write_ptr();
                }
                const uint32_t base = group_base + j * (rm_chunk_tiles * tile_bytes);
                const uint32_t kc = kc0 + j;
                auto issue = [&]() {
                    for (uint32_t rr = 0; rr < nvalid; ++rr) {
                        const uint32_t token = entries[jl * T + first + rr] & 0x1FFu;
                        noc.async_read(
                            x_acc,
                            CoreLocalMem<uint32_t>(base + rr * seg_bytes),
                            seg_bytes,
                            {.page_id = token, .offset_bytes = kc * seg_bytes},
                            {});
                    }
                };
                if (sampled) {
                    DeviceZoneScopedN("R_SEG_ISSUE");
                    issue();
                } else {
                    issue();
                }
            }
            if (sampled) {
                DeviceZoneScopedN("R_SEG_BARRIER");
                noc.async_read_barrier();
            } else {
                noc.async_read_barrier();
            }
            for (uint32_t j = 0; j < rm_group; ++j) {
                cb_rm.push_back(rm_chunk_tiles);
            }
        }
    };
    // Scalar tiles are pushed per M-block (m_block tiles, constant size): row r of tile `m` holds the
    // routing weight of token first + r in all 32 columns (0 for padding rows).
    auto push_scalars = [&](uint32_t jl, uint32_t count, uint32_t row_tile0, uint32_t mb) {
        DeviceZoneScopedN("R_SCALARS");
        cb_rscal.reserve_back(m_block);
        CoreLocalMem<volatile uint16_t> rs(cb_rscal.get_write_ptr());
        for (uint32_t m = 0; m < mb; ++m) {
            const uint32_t first = (row_tile0 + m) * 32;
            for (uint32_t rr = 0; rr < 32; ++rr) {
                const uint32_t idx = first + rr;
                const uint16_t wb = idx < count ? static_cast<uint16_t>(entries[jl * T + idx] >> 16) : 0;
                for (uint32_t h = 0; h < 2; ++h) {
                    const uint32_t face = ((rr >> 4) << 1) + h;
                    const uint32_t base = m * (tile_bytes / 2) + face * 256 + (rr & 15u) * 16;
                    for (uint32_t col = 0; col < 16; ++col) {
                        rs[base + col] = wb;
                    }
                }
            }
        }
        cb_rscal.push_back(m_block);
    };

    for (uint32_t jl = 0; jl < n_owned; ++jl) {
        const uint32_t count = counts[jl];
        if (count == 0) {
            continue;
        }
        const uint32_t m = (count + 31) / 32;

        for (uint32_t r0 = 0; r0 < m; r0 += m_chunk) {
            const uint32_t mc = (m - r0) < m_chunk ? (m - r0) : m_chunk;

            // ---- gate_up slot (already in flight / landed for the first owned expert's first chunk) ----
            {
                DeviceZoneScopedN("R_GU");
                if (gu_prefetched && jl == 0 && r0 == 0) {
                    noc.async_read_barrier();
                } else {
                    cb_w.reserve_back(slot_tiles);
                    read_gate_up(jl);
                    noc.async_read_barrier();
                }
                cb_w.push_back(slot_tiles);
            }

            for (uint32_t b0 = 0; b0 < mc; b0 += m_block) {
                const uint32_t mb = (mc - b0) < m_block ? (mc - b0) : m_block;
                {
                    DeviceZoneScopedN("R_ROWS");
                    for (uint32_t mm = 0; mm < mb; ++mm) {
                        read_rows(jl, count, r0 + b0 + mm);
                    }
                }
                push_scalars(jl, count, r0 + b0, mb);

                // ---- down slot: prefetched while compute works on the gate_up blocks. ----
                if (b0 == 0) {
                    const auto dn = TensorAccessor(dn_args, get_arg_val<uint32_t>(kAddrRt + n_owned + jl));
                    ShardView dn_shard(dn);
                    DeviceZoneScopedN("R_DN");
                    cb_w.reserve_back(slot_tiles);
                    for (uint32_t s = 0; s < dn_shards; ++s) {
                        noc.async_read(
                            dn_shard,
                            cb_w,
                            dn_shard_bytes,
                            {.shard_id = c * dn_shards + s},
                            {.offset_bytes = s * dn_shard_bytes});
                    }
                    noc.async_read_barrier();
                    cb_w.push_back(slot_tiles);
                }
            }
        }
    }
}
