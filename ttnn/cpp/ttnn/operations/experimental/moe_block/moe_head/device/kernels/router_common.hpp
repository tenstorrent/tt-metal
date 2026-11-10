// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The router cores' kernels' shared pieces (router_input_feeder, router_compute, router_output_writer):
//   * RouterCommonArgs: the router kernels' common runtime args (the program hash then holds no address);
//   * routing_rows: the joint routing rows (each token's top-K expert ids and weights) from the top-K keys and scores;
//   * route_words (router_compute, ROUTER_ROUTE_WORDS): the dispatch prep's route words, one token at a time.
// The routing rows need ROUTER_FOLD, the route words the ROUTE_WORDS_* defines (head_router_program.cpp).
#pragma once

#include <cstdint>
#if defined(COMPILE_FOR_TRISC)
#include "api/compute/common.h"
#else
#include "api/dataflow/dataflow_api.h"
#endif
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/common/kernels/tile_bytes.hpp"

// The router kernels' common args (every router kernel - NCRISC, compute, BRISC - gets the same first two), read
// once by kernel_main from moe_block_args::common_pointer(0). The host defines the offset of the math-sync word in the
// arena: ROUTER_MATH_SYNC_OFFSET (router_compute's PACK -> MATH word: arena_base + ROUTER_MATH_SYNC_OFFSET).
struct RouterCommonArgs {
    uint32_t arena_base;                     // [0] the router cores' arena base (the CB range's address on them)
    uint32_t engine_done_semaphore_address;  // [1] the matmul cores count their partial logits tiles into it
};
FORCE_INLINE RouterCommonArgs read_router_common_args(const volatile tt_l1_ptr uint32_t* args) {
    return RouterCommonArgs{args[0], args[1]};
}

// The joint routing rows: per token column, its top_k expert ids (uint16) then their weights (bf16), from the best page
// (each column's top-K keys, sorted descending) and the scores (the fin's score tiles, transposed like the logits)
namespace routing_rows {
// (row, column) of a tile: face (row / 16) * 2 + column / 16, then row-major within the face
FORCE_INLINE uint32_t face_offset(uint32_t row, uint32_t column) {
    return ((row >> 4) * 2 + (column >> 4)) * 256 + (row & 15) * 16 + (column & 15);
}
// the index of value's top bit (value > 0): five steps (no Zbb on the RISCs: __builtin_clz is a library loop)
FORCE_INLINE int32_t top_bit_index(uint32_t value) {
    int32_t index = 0;
    if (value >> 16) {
        value >>= 16, index += 16;
    }
    if (value >> 8) {
        value >>= 8, index += 8;
    }
    if (value >> 4) {
        value >>= 4, index += 4;
    }
    if (value >> 2) {
        value >>= 2, index += 2;
    }
    if (value >> 1) {
        index += 1;
    }
    return index;
}
// value (> 0, < 2^31): a value scaled by 2^scale_exponent -> bf16 bits, round to nearest even
FORCE_INLINE uint32_t scaled_to_bf16(uint32_t value, int32_t scale_exponent) {
    if (!value) {
        return 0;
    }
    const int32_t top_bit = top_bit_index(value);
    uint32_t mantissa;
    int32_t exponent = top_bit;
    if (top_bit > 7) {  // (keep 8 bits: the hidden one + 7)
        const int32_t dropped_bits = top_bit - 7;
        const uint32_t half = 1u << (dropped_bits - 1), remainder = value & ((1u << dropped_bits) - 1);
        mantissa = value >> dropped_bits;
        if (remainder > half || (remainder == half && (mantissa & 1))) {
            ++mantissa;
        }
        if (mantissa >> 8) {
            mantissa >>= 1, ++exponent;
        }
    } else {
        mantissa = value << (7 - top_bit);
    }
    return ((uint32_t)(exponent - scale_exponent + 127) << 7) | (mantissa & 0x7F);
}
// the expert id bits of a top-K key (the low mantissa bits of its fp32 score)
constexpr uint32_t id_bits_mask_of(uint32_t candidate_tiles) {
    return candidate_tiles * 32 <= 256 ? 0xFF : candidate_tiles * 32 <= 512 ? 0x1FF : 0x3FF;
}
// the fp32 score of expert `expert` for token column `column` - the fin's score tiles at scores_l1, transposed like the
// logits: (expert % 32, column) of tile expert / 32
FORCE_INLINE uint32_t score_bits(uint32_t scores_l1, uint32_t expert, uint32_t column) {
    return reinterpret_cast<const uint32_t*>(
        scores_l1 + (expert >> 5) * moe_block_tiles::kFp32TileBytes)[face_offset(expert & 31, column)];
}
// token columns [first_column, end_column) of the rows at rows_l1 (4 top_k bytes a column)
template <uint32_t top_k, uint32_t expert_tiles>
inline void build(
    uint32_t best_page_l1,
    uint32_t scores_l1,
    uint32_t rows_l1,
    uint32_t scale_bits,
    uint32_t first_column,
    uint32_t end_column) {
    // weights = score / their sum x scale in integers (the RISCs have no FPU, no 64-bit divide): the top_k scores in
    // one block exponent (the largest's top bit at 25: a sum of up to 32 below 2^31), the sum normalized to bit 30, its
    // reciprocal by one 32-bit divide (16 bits: a bf16 weight needs 8 + rounding), x the scale's mantissa, then one
    // 32 x 32 -> 64 product a weight
    const uint32_t scale_mantissa = (scale_bits & 0x7FFFFF) | 0x800000;  // (scale = mantissa x 2^exponent)
    const int32_t scale_exponent = (int32_t)((scale_bits >> 23) & 0xFF) - 150;
    constexpr uint32_t id_mask = id_bits_mask_of(expert_tiles);
    for (uint32_t column = first_column; column < end_column; ++column) {
        // (weights: the scores' fp32 bits, then their block-exponent integers)
        uint32_t expert_ids[top_k], weights[top_k], sum = 0;
        int32_t max_exponent = 0;
        uint32_t bad_ids = 0;
        for (uint32_t pick = 0; pick < top_k; ++pick) {
            const uint32_t key = reinterpret_cast<const uint32_t*>(
                best_page_l1)[face_offset(pick, column)];  // (the top-K's key: id in the low bits)
            expert_ids[pick] = key & id_mask;
            bad_ids |= uint32_t(expert_ids[pick] >= expert_tiles * 32) << pick;
        }
        if (bad_ids) {  // (an id past the candidates - only a broken top-K: every consumer indexes by it, the planner's
                        // tables and the route words. Each such key takes the first expert the token does not have yet
                        // (no repeat: the sizing's one entry an expert a token holds) at weight bits 0x7FC0 (bf16
                        // NaN): the error marker in the routing rows - garbage routing, not a hang)
            for (uint32_t pick = 0; pick < top_k; ++pick) {
                if (!((bad_ids >> pick) & 1)) {
                    continue;
                }
                for (uint32_t expert = 0;; ++expert) {  // (top_k <= E experts: a free one exists)
                    bool used = false;
                    for (uint32_t other = 0; other < top_k; ++other) {
                        used |=
                            (other < pick || !((bad_ids >> other) & 1)) && expert_ids[other] / ROUTER_FOLD == expert;
                    }
                    if (!used) {
                        expert_ids[pick] = expert * ROUTER_FOLD;
                        break;
                    }
                }
            }
        }
        for (uint32_t pick = 0; pick < top_k; ++pick) {
            weights[pick] = (bad_ids >> pick) & 1 ? 0 : score_bits(scores_l1, expert_ids[pick], column);
            if (weights[pick] >> 31) {
                weights[pick] = 0;  // (sigmoid and sqrt(softplus) scores are >= 0)
            }
            const int32_t exponent = (weights[pick] >> 23) & 0xFF;
            if (exponent > max_exponent) {
                max_exponent = exponent;
            }
        }
        for (uint32_t pick = 0; pick < top_k; ++pick) {
            // (mantissa top bit 23 -> 25 for the largest)
            const int32_t exponent = (weights[pick] >> 23) & 0xFF, shift = exponent - max_exponent + 2;
            const uint32_t mantissa = (weights[pick] & 0x7FFFFF) | 0x800000;
            weights[pick] = exponent == 0 ? 0
                            : shift >= 0  ? (mantissa << shift)
                            : shift > -31 ? (mantissa >> -shift)
                                          : 0;
            sum += weights[pick];
        }
        uint16_t* row = reinterpret_cast<uint16_t*>(rows_l1 + column * 4 * top_k);
        const int32_t sum_shift = sum ? 30 - top_bit_index(sum) : 0;
        const uint32_t normalized_sum = sum_shift >= 0 ? sum << sum_shift : sum >> -sum_shift;
        // (2^32 / (normalized_sum / 2^14): 2^62 / normalized_sum = reciprocal x 2^16)
        const uint32_t reciprocal = normalized_sum ? 0xFFFFFFFFu / (normalized_sum >> 14) : 0;
        // (weight = normalized weight x reciprocal_scale / 2^32 x 2^(scale_exponent - 6))
        const uint32_t reciprocal_scale = (uint32_t)(((uint64_t)reciprocal * scale_mantissa) >> 8);
        for (uint32_t pick = 0; pick < top_k; ++pick) {
            // (candidate -> expert: RouterSpec.fold)
            row[pick] = static_cast<uint16_t>(expert_ids[pick] / ROUTER_FOLD);
            const uint32_t normalized_weight =
                sum_shift >= 0 ? (uint32_t)weights[pick] << sum_shift : (uint32_t)weights[pick] >> -sum_shift;
            row[top_k + pick] = static_cast<uint16_t>(
                (bad_ids >> pick) & 1
                    ? 0x7FC0u
                    : scaled_to_bf16(
                          (uint32_t)(((uint64_t)normalized_weight * reciprocal_scale) >> 32), 6 - scale_exponent));
        }
    }
}
}  // namespace routing_rows

#if defined(ROUTER_ROUTE_WORDS) && defined(COMPILE_FOR_TRISC)
// The dispatch prep's route words, one token at a time, parametrized by the ROUTE_WORDS_* defines: a token's mask of
// destination chips and whether the dispatch goes X or Y first (the fewer fabric links)
namespace route_words {
constexpr uint32_t sp_rows = ROUTE_WORDS_SP_ROWS, tp_columns = ROUTE_WORDS_TP_COLUMNS, top_k = ROUTE_WORDS_TOP_K,
                   words_per_token = ROUTE_WORDS_PER_TOKEN;
#ifdef ROUTER_CHIP_FROM_READY_PAGE
inline uint32_t my_chip = 0;  // (this chip's index: the router NCRISC's ready page word 1 - router_compute.cpp sets it)
#else
constexpr uint32_t my_chip = ROUTE_WORDS_MY_CHIP;
#endif
constexpr bool wrap_x = ROUTE_WORDS_WRAP_X, wrap_y = ROUTE_WORDS_WRAP_Y;
// (a ring's spans, forward | backward << 4 links: per (the positions' mask rotated to mine, tie))
constexpr uint8_t x_ring_spans[wrap_x ? (1u << tp_columns) * 2 : 1] = {ROUTE_WORDS_X_RING_INIT};
constexpr uint8_t y_ring_spans[wrap_y ? (1u << sp_rows) * 2 : 1] = {ROUTE_WORDS_Y_RING_INIT};
constexpr uint8_t chip_of_expert[ROUTE_WORDS_EXPERTS] = {ROUTE_WORDS_EXPERT_CHIPS_INIT};
inline uint32_t tie_break(uint32_t token) { return ((token * 2654435761u) >> 30) & 1; }
inline uint32_t top_bit(uint32_t mask) {
    uint32_t bit = 0;
    while (mask >>= 1) {
        ++bit;
    }
    return bit;
}
inline uint32_t bottom_bit(uint32_t mask) {
    uint32_t bit = 0;
    while (!(mask & 1)) {
        mask >>= 1, ++bit;
    }
    return bit;
}
// the span (forward links | backward links << 4) from `position` to the positions of `mask` on a line or ring
template <uint32_t positions, bool wrap>
inline uint32_t span_of(const uint8_t* ring_spans, uint32_t position, uint32_t mask, uint32_t token) {
    mask &= ~(1u << position) & ((1u << positions) - 1);
    if (mask == 0) {
        return 0;
    }
    if constexpr (wrap) {
        const uint32_t rotated = ((mask >> position) | (mask << (positions - position))) & ((1u << positions) - 1);
        return ring_spans[rotated * 2 + tie_break(token)];
    } else {
        const uint32_t highest = top_bit(mask), lowest = bottom_bit(mask);
        return (highest > position ? highest - position : 0) | ((lowest < position ? position - lowest : 0) << 4);
    }
}
inline uint32_t x_span(uint32_t position, uint32_t column_mask, uint32_t token) {
    return span_of<tp_columns, wrap_x>(x_ring_spans, position, column_mask, token);
}
inline uint32_t y_span(uint32_t position, uint32_t row_mask, uint32_t token) {
    return span_of<sp_rows, wrap_y>(y_ring_spans, position, row_mask, token);
}
inline uint32_t links(uint32_t span) { return (span & 15) + (span >> 4); }
inline uint32_t columns_of(uint32_t chip_mask) {
    uint32_t column_mask = 0;
    for (uint32_t row = 0; row < sp_rows; ++row) {
        column_mask |= (chip_mask >> (row * tp_columns)) & ((1u << tp_columns) - 1);
    }
    return column_mask;
}
inline uint32_t rows_of(uint32_t chip_mask) {
    uint32_t row_mask = 0;
    for (uint32_t row = 0; row < sp_rows; ++row) {
        row_mask |= (((chip_mask >> (row * tp_columns)) & ((1u << tp_columns) - 1)) != 0) << row;
    }
    return row_mask;
}
inline uint32_t rows_in_column(uint32_t chip_mask, uint32_t column) {
    uint32_t row_mask = 0;
    for (uint32_t row = 0; row < sp_rows; ++row) {
        row_mask |= ((chip_mask >> (row * tp_columns + column)) & 1) << row;
    }
    return row_mask;
}
inline uint32_t columns_in_row(uint32_t chip_mask, uint32_t row) {
    return (chip_mask >> (row * tp_columns)) & ((1u << tp_columns) - 1);
}
// token index's route word(s) at output[words_per_token index] (words_per_token 1: mask | order << 16 | link << 17;
// 2: mask, order | link << 1)
inline void put(volatile uint32_t* output, uint32_t index, uint32_t chip_mask, uint32_t flags) {
    if constexpr (words_per_token == 2) {
        output[2 * index] = chip_mask;
        output[2 * index + 1] = flags;
    } else {
        output[index] = chip_mask | (flags << 16);
    }
}
// token `token`'s route word(s) from its top_k expert ids, at output[words_per_token index]
inline void write(volatile uint32_t* output, uint32_t index, volatile const uint16_t* expert_ids, uint32_t token) {
    const uint32_t my_row = my_chip / tp_columns, my_column = my_chip % tp_columns;
    uint32_t chip_mask = 0;
    for (uint32_t pick = 0; pick < top_k; ++pick) {
        chip_mask |= 1u << chip_of_expert[expert_ids[pick]];
    }
    chip_mask &= ~(1u << my_chip);
    if (chip_mask == 0) {
        return put(output, index, 0, 0);
    }
    const uint32_t column_mask = columns_of(chip_mask), row_mask = rows_of(chip_mask);
    // (the links of X first, then Y in every column; and of Y first, then X in every row)
    uint32_t x_first_links = links(x_span(my_column, column_mask & ~(1u << my_column), token));
    uint32_t y_first_links = links(y_span(my_row, row_mask & ~(1u << my_row), token));
    for (uint32_t column = 0; column < tp_columns; ++column) {
        if (((column_mask | (1u << my_column)) >> column) & 1) {
            x_first_links += links(y_span(my_row, rows_in_column(chip_mask, column) & ~(1u << my_row), token));
        }
    }
    for (uint32_t row = 0; row < sp_rows; ++row) {
        if (((row_mask | (1u << my_row)) >> row) & 1) {
            y_first_links += links(x_span(my_column, columns_in_row(chip_mask, row) & ~(1u << my_column), token));
        }
    }
    const uint32_t token_hash = token * 2654435761u;
    const uint32_t order =
        x_first_links < y_first_links ? 0 : (y_first_links < x_first_links ? 1 : (token_hash >> 31) & 1);
    put(output, index, chip_mask, order | (((token_hash >> 29) & 1) << 1));
}
}  // namespace route_words
#endif
