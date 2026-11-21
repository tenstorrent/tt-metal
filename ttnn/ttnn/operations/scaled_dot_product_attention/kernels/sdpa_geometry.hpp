// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// The work geometry shared by the reader, the compute kernel and the writer of scaled_dot_product_attention,
// so that all three derive rounds, units, rows and key chunks from the same compile-time arguments in the
// same way (op_design.md, Work Distribution and Block schedule).
//
//   round      C consecutive NEO units of one (sequence b, key/value head c);
//              round = (b * Hkv + c) * rounds_per_head + head_round.
//   unit       q_chunk_tiles query tile-rows x Dt of one query head, units of a head q-chunk-major:
//              u -> q_chunk = u / group, member = u % group, query head h = c * group + member.
//              NEO t of a round owns unit head_round * C + t; past the head's last unit it recomputes
//              NEO 0's unit of the round (padding: every NEO takes the same entries on every ring).
//   row clamp  unit row i is query tile-row min(q_chunk * q_chunk_tiles + i, Sqt - 1).
//   key chunk  k_chunk_tiles key tile-rows. Causal: a round needs the chunks up to its last query tile,
//              a NEO only those up to its own; it pops the rest of the round's chunks unread.

#pragma once

#include <cstdint>

// Entries of the constants ring, per NEO, in the order the reader pushes them (the host sizes the ring with
// const_tiles(): 1, plus 1 when causal).
constexpr uint32_t SCALER_TILE = 0;     // the reduce scaler (1.0 on row 0 of every face; MAX and SUM alike)
constexpr uint32_t DIAG_MASK_TILE = 1;  // causal only: the diagonal mask (0 where column <= row, -inf above)

template <
    uint32_t C,
    uint32_t Hq,
    uint32_t Hkv,
    uint32_t group,
    uint32_t Sqt,
    uint32_t Skt,
    uint32_t Dt,
    uint32_t qc,
    uint32_t kc,
    uint32_t units_per_head,
    uint32_t rounds_per_head,
    bool causal>
struct SdpaGeometry {
    static_assert(C > 0 && qc > 0 && kc > 0 && Dt > 0, "bad geometry");

    struct Round {
        uint32_t b;
        uint32_t c;
        uint32_t head_round;
    };

    static Round round_coords(uint32_t round) {
        const uint32_t head_round = round % rounds_per_head;
        const uint32_t bc = round / rounds_per_head;
        return {bc / Hkv, bc % Hkv, head_round};
    }

    // The unit NEO t computes in a round (padded NEOs recompute NEO 0's unit).
    static uint32_t unit_of(uint32_t head_round, uint32_t t) {
        const uint32_t u = head_round * C + t;
        return u < units_per_head ? u : head_round * C;
    }

    static uint32_t q_head(uint32_t c, uint32_t u) { return c * group + u % group; }
    static uint32_t q_tile_start(uint32_t u) { return (u / group) * qc; }

    // Absolute query tile-row of unit row i (clamped).
    static uint32_t q_row(uint32_t u, uint32_t i) {
        const uint32_t row = q_tile_start(u) + i;
        return row < Sqt ? row : Sqt - 1;
    }
    static uint32_t last_q_tile(uint32_t u) { return q_row(u, qc - 1); }

    // Key tile-rows a round needs (causal: up to its last query tile) and how many chunks that is.
    static uint32_t round_key_limit(uint32_t head_round) {
        if constexpr (!causal) {
            return Skt;
        } else {
            const uint32_t last_unit_end = head_round * C + C;
            const uint32_t last_unit = (last_unit_end < units_per_head ? last_unit_end : units_per_head) - 1;
            const uint32_t limit = last_q_tile(last_unit) + 1;
            return limit < Skt ? limit : Skt;
        }
    }
    static uint32_t round_chunks(uint32_t head_round) { return (round_key_limit(head_round) + kc - 1) / kc; }

    // Key tile-rows a unit attends.
    static uint32_t unit_key_limit(uint32_t u) {
        if constexpr (!causal) {
            return Skt;
        } else {
            const uint32_t limit = last_q_tile(u) + 1;
            return limit < Skt ? limit : Skt;
        }
    }

    // Key tiles of chunk `chunk` below `limit` (0 when the chunk lies past it).
    static uint32_t chunk_extent(uint32_t chunk, uint32_t limit) {
        const uint32_t first = chunk * kc;
        if (first >= limit) {
            return 0;
        }
        return limit - first < kc ? limit - first : kc;
    }

    // Pages of the interleaved tensors.
    static uint32_t q_page(uint32_t b, uint32_t h, uint32_t row, uint32_t d) { return ((b * Hq + h) * Sqt + row) * Dt + d; }
    static uint32_t kv_page(uint32_t b, uint32_t c, uint32_t row, uint32_t d) {
        return ((b * Hkv + c) * Skt + row) * Dt + d;
    }
};

// The geometry of the calling kernel, from its compile-time arguments (identical in all three kernels).
#define SDPA_GEOMETRY_FROM_ARGS                                                                                    \
    SdpaGeometry<                                                                                                  \
        get_arg(args::compute_threads),                                                                            \
        get_arg(args::Hq),                                                                                         \
        get_arg(args::Hkv),                                                                                        \
        get_arg(args::group),                                                                                      \
        get_arg(args::Sqt),                                                                                        \
        get_arg(args::Skt),                                                                                        \
        get_arg(args::Dt),                                                                                         \
        get_arg(args::q_chunk_tiles),                                                                              \
        get_arg(args::k_chunk_tiles),                                                                              \
        get_arg(args::units_per_head),                                                                             \
        get_arg(args::rounds_per_head),                                                                            \
        get_arg(args::causal) != 0>
