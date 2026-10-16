// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Blackhole gathers of the grouped top-k gate, shared by the writer and the reader: the reader takes half of each
// gather (every other winning-group slot, the second half of the tokens) from a job the writer posts.

#pragma once

#include "ttnn/operations/experimental/deepseek_prefill/moe_grouped_topk/device/kernels/dataflow/moe_gate_common_dataflow.hpp"

namespace gate_gather {

constexpr uint32_t tile_height_max = 32;

// A job page: addresses and counts, one word each.
enum JobWord : uint32_t { TOKENS = 0, SRC = 1, DST = 2, OUT = 3, FIRST_TOKEN = 4, ORDER = 5 };

// Orders the RISC's earlier L1 stores before what it does next: a load of the last stored word returns only once the
// stores reached L1, and the next instruction waits for it.
FORCE_INLINE void settle_l1_stores(volatile tt_l1_ptr uint32_t* last_stored) {
    const uint32_t v = *last_stored;
    asm volatile("add zero, zero, %0" : : "r"(v) : "memory");
}

// Each token's group in each slot (slot_groups[k * 32 + t]) from the group sort's order tile (uint16, rank k of token t
// at rank-major face offsets): with by_id the winning groups in ascending id, else in rank order.
template <uint32_t topk_groups, bool by_id>
FORCE_INLINE void slot_groups_from_order(const uint16_t* order, uint32_t tokens, uint16_t* slot_groups) {
    for (uint32_t t = 0; t < tokens; t++) {
        const uint32_t token_offset = (t < rows_per_face) ? t : elements_per_face + (t - rows_per_face);
        if constexpr (by_id && topk_groups == 4) {
            uint32_t g0 = order[token_offset];
            uint32_t g1 = order[rows_per_face + token_offset];
            uint32_t g2 = order[2 * rows_per_face + token_offset];
            uint32_t g3 = order[3 * rows_per_face + token_offset];
            auto sort2 = [](uint32_t& a, uint32_t& b) {
                const uint32_t lo = a < b ? a : b;
                b = a < b ? b : a;
                a = lo;
            };
            sort2(g0, g1);
            sort2(g2, g3);
            sort2(g0, g2);
            sort2(g1, g3);
            sort2(g1, g2);
            slot_groups[t] = g0;
            slot_groups[tile_height_max + t] = g1;
            slot_groups[2 * tile_height_max + t] = g2;
            slot_groups[3 * tile_height_max + t] = g3;
        } else {
            for (uint32_t k = 0; k < topk_groups; k++) {
                const uint32_t rank_offset = (k < rows_per_face)
                                                 ? k * rows_per_face
                                                 : 2 * elements_per_face + (k - rows_per_face) * rows_per_face;
                const uint16_t group = order[rank_offset + token_offset];
                uint32_t i = k;
                if constexpr (by_id) {
                    for (; i > 0 && slot_groups[(i - 1) * tile_height_max + t] > group; i--) {
                        slot_groups[i * tile_height_max + t] = slot_groups[(i - 1) * tile_height_max + t];
                    }
                }
                slot_groups[i * tile_height_max + t] = group;
            }
        }
    }
}

// Token t's first element in a 32x32 tile.
FORCE_INLINE uint32_t row_elements(uint32_t t) {
    return (t < rows_per_face) ? t * columns_per_face : 2 * elements_per_face + (t - rows_per_face) * columns_per_face;
}

// Slot k's score rows from each token's group tile: two face-row reads a token. Needs
// noc_async_read_one_packet_set_state(local core, score_tile::face_line_bytes).
FORCE_INLINE void read_slot_rows(uint32_t src_base, uint32_t dst_slot, const uint16_t* groups, uint32_t tokens) {
    for (uint32_t t = 0; t < tokens; t++) {
        const uint32_t row = row_elements(t) * score_tile::bytes_per_element;
        const uint32_t src = src_base + groups[t] * score_tile::tile_size_bytes + row;
        noc_async_read_one_packet_with_state<true>(src, dst_slot + row);
        noc_async_read_one_packet_with_state<true>(
            src + score_tile::face_size_bytes, dst_slot + row + score_tile::face_size_bytes);
    }
}

// Tokens [t_begin, t_end): each selected expert's unbiased score. With remap the indices are slot positions
// (slot * 32 + column) and become expert ids in place first. Loads go out before their uses so the L1 load latency
// overlaps; the stores are settled, as another RISC or the NoC reads them next.
template <uint32_t n_activated_experts, bool remap>
FORCE_INLINE void gather_rows(
    uint32_t t_begin, uint32_t t_end, uint32_t* idx, const uint32_t* sig, uint32_t* out, const uint16_t* slot_groups) {
    // Index pairs are read as one word; an odd count's last word carries one unused neighbor.
    constexpr uint32_t pairs = (n_activated_experts + 1) / 2;
    auto pair_word = [](uint32_t p) {
        return p < columns_per_face / 2 ? p : elements_per_face / 2 + p - columns_per_face / 2;
    };
    auto column = [](uint32_t e) { return e < columns_per_face ? e : elements_per_face + e - columns_per_face; };
    for (uint32_t t = t_begin; t < t_end; t++) {
        const uint32_t row = row_elements(t);
        uint32_t* irow = idx + row / 2;
        uint32_t* orow = out + row;
        const uint32_t* srow = sig + row;
        uint32_t w[pairs];
#pragma GCC unroll 16
        for (uint32_t p = 0; p < pairs; p++) {
            w[p] = irow[pair_word(p)];
        }
        uint32_t x[n_activated_experts];
#pragma GCC unroll 32
        for (uint32_t e = 0; e < n_activated_experts; e++) {
            x[e] = (e % 2) ? w[e / 2] >> 16 : w[e / 2] & 0xFFFF;
            if constexpr (remap) {
                x[e] = (x[e] & (columns_per_tile - 1)) |
                       (static_cast<uint32_t>(slot_groups[(x[e] / columns_per_tile) * tile_height_max + t]) *
                        columns_per_tile);
            }
        }
        uint32_t v[n_activated_experts];
#pragma GCC unroll 32
        for (uint32_t e = 0; e < n_activated_experts; e++) {
            v[e] = srow
                [(x[e] / columns_per_tile) * elements_per_tile + ((x[e] & columns_per_face) ? elements_per_face : 0) +
                 (x[e] & (columns_per_face - 1))];
        }
#pragma GCC unroll 32
        for (uint32_t e = 0; e < n_activated_experts; e++) {
            orow[column(e)] = v[e];
        }
        if constexpr (remap) {
#pragma GCC unroll 16
            for (uint32_t p = 0; p < n_activated_experts / 2; p++) {
                irow[pair_word(p)] = x[2 * p] | (x[2 * p + 1] << 16);
            }
            if constexpr (n_activated_experts % 2 == 1) {
                reinterpret_cast<uint16_t*>(idx)[row + column(n_activated_experts - 1)] = x[n_activated_experts - 1];
            }
        }
    }
    if (t_end > t_begin) {
        const uint32_t row = row_elements(t_end - 1);
        if constexpr (remap) {
            settle_l1_stores(
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(idx) + (row + column(n_activated_experts - 1)) / 2);
        } else {
            settle_l1_stores(
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out) + row + column(n_activated_experts - 1));
        }
    }
}

}  // namespace gate_gather
