// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Shared MoE-gate dataflow helpers: tile/face layout constants, the scalar generators, the
// unbiased-score gather() (keyed off cb_out_indices, regardless of how those indices were produced),
// and the padded-row sentinel patch. Included by both moe_grouped_topk and moe_hash_gate writers.

#pragma once

#include <cstdint>

#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc.h"

constexpr uint32_t rows_per_face = 16;
constexpr uint32_t columns_per_face = 16;
constexpr uint32_t rows_per_tile = 32;
constexpr uint32_t columns_per_tile = 32;
constexpr uint32_t elements_per_face = rows_per_face * columns_per_face;  // 256
constexpr uint32_t elements_per_tile = rows_per_tile * columns_per_tile;  // 1024

namespace score_tile {
constexpr uint32_t bytes_per_element = 4;                                    // float32
constexpr uint32_t face_line_bytes = columns_per_face * bytes_per_element;   // 64
constexpr uint32_t face_size_bytes = elements_per_face * bytes_per_element;  // 1024
constexpr uint32_t tile_size_bytes = elements_per_tile * bytes_per_element;  // 4096
}  // namespace score_tile

namespace index_tile {
constexpr uint32_t bytes_per_element = 2;                                    // uint16
constexpr uint32_t face_line_bytes = columns_per_face * bytes_per_element;   // 32
constexpr uint32_t face_size_bytes = elements_per_face * bytes_per_element;  // 512
constexpr uint32_t tile_size_bytes = elements_per_tile * bytes_per_element;  // 2048
}  // namespace index_tile

FORCE_INLINE void generate_reduce_scalar(
    const uint32_t cb_reduce_ones_scalar, const uint32_t packed_scalar, const uint32_t n_activated_experts) {
    Noc noc;
    CircularBuffer reduce_cb(cb_reduce_ones_scalar);
    reduce_cb.reserve_back(1);

    uint32_t write_addr = reduce_cb.get_write_ptr();
    tt_l1_ptr uint32_t* write_ptr = reinterpret_cast<tt_l1_ptr uint32_t*>(write_addr);
    uint32_t scalar = packed_scalar;
    for (uint32_t i = 0; i < n_activated_experts; i++) {
        write_ptr[i] = scalar;
        if (i > rows_per_face - 1) {
            write_ptr[i + elements_per_face - columns_per_face + 1] = scalar;
        }
    }
    for (uint32_t i = n_activated_experts; i < rows_per_tile; i++) {
        write_ptr[i] = 0;
        if (i == rows_per_face) {
            // Zero the second face's first line (offset face_size_bytes into the tile).
            noc.async_write_zeros(
                reduce_cb, score_tile::face_line_bytes, {.offset_bytes = score_tile::face_size_bytes});
        }
    }
    uint32_t face_3_write_addr = write_addr + 2 * score_tile::face_size_bytes;
    uint32_t face_4_write_addr = write_addr + 3 * score_tile::face_size_bytes;
    noc.write_zeros_l1_barrier();
    noc_async_read(get_noc_addr(write_addr), face_3_write_addr, score_tile::face_line_bytes);
    noc_async_read(
        get_noc_addr(write_addr + score_tile::face_size_bytes), face_4_write_addr, score_tile::face_line_bytes);
    noc_async_read_barrier();

    reduce_cb.push_back(1);
}

FORCE_INLINE void write_single_scalar(const uint32_t cb_scalar, const uint32_t packed_scalar) {
    CircularBuffer cb(cb_scalar);
    cb.reserve_back(1);
    uint32_t write_addr = cb.get_write_ptr();
    tt_l1_ptr uint32_t* write_ptr = reinterpret_cast<tt_l1_ptr uint32_t*>(write_addr);
    write_ptr[0] = packed_scalar;
    cb.push_back(1);
}

template <
    uint32_t cb_out_indices,
    uint32_t cb_sigmoid_scores,
    uint32_t cb_gathered_sigmoid,
    uint32_t width_tiles,
    uint32_t n_activated_experts,
    uint32_t n_activated_expert_tiles>
FORCE_INLINE void gather(uint32_t tokens_per_tile) {
    CircularBuffer sigmoid_cb(cb_sigmoid_scores);
    CircularBuffer out_indices_cb(cb_out_indices);
    CircularBuffer gathered_cb(cb_gathered_sigmoid);

    sigmoid_cb.wait_front(width_tiles);
    out_indices_cb.wait_front(1);

    gathered_cb.reserve_back(n_activated_expert_tiles);

    uint32_t sigmoid_base_addr = sigmoid_cb.get_read_ptr();
    uint32_t indices_addr = out_indices_cb.get_read_ptr();
    uint32_t gathered_addr = gathered_cb.get_write_ptr();

    volatile tt_l1_ptr uint16_t* indices_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(indices_addr);
    volatile tt_l1_ptr uint32_t* sigmoid_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sigmoid_base_addr);
    volatile tt_l1_ptr uint32_t* gathered_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(gathered_addr);

    for (uint32_t token = 0; token < tokens_per_tile; token++) {
        uint32_t token_face_row = token % rows_per_face;
        uint32_t token_face_base = (token < rows_per_face) ? 0 : 2;

        for (uint32_t expert = 0; expert < n_activated_experts; expert++) {
            uint32_t idx_col = expert;
            uint32_t idx_face_col = idx_col % columns_per_face;
            uint32_t idx_face = token_face_base + (idx_col < columns_per_face ? 0 : 1);
            uint32_t idx_offset =
                idx_face * (index_tile::face_size_bytes / 2) + token_face_row * columns_per_face + idx_face_col;

            uint16_t expert_idx = indices_ptr[idx_offset];

            uint32_t sigmoid_tile = expert_idx / columns_per_tile;
            uint32_t sigmoid_col = expert_idx % columns_per_tile;
            uint32_t sigmoid_face_col = sigmoid_col % columns_per_face;
            uint32_t sigmoid_face = token_face_base + (sigmoid_col < columns_per_face ? 0 : 1);
            uint32_t sigmoid_offset = sigmoid_tile * (score_tile::tile_size_bytes / 4) +
                                      sigmoid_face * (score_tile::face_size_bytes / 4) +
                                      token_face_row * columns_per_face + sigmoid_face_col;

            uint32_t sigmoid_val = sigmoid_ptr[sigmoid_offset];

            uint32_t gathered_face_col = idx_col % columns_per_face;
            uint32_t gathered_face = token_face_base + (idx_col < columns_per_face ? 0 : 1);
            uint32_t gathered_offset = gathered_face * (score_tile::face_size_bytes / 4) +
                                       token_face_row * columns_per_face + gathered_face_col;

            gathered_ptr[gathered_offset] = sigmoid_val;
        }
    }

    sigmoid_cb.pop_front(width_tiles);

    gathered_cb.push_back(n_activated_expert_tiles);
}

// Overwrite all n_activated_experts index entries for a single (tile-local) row with SENTINEL.
// Mirrors the face/column addressing used by gather() so it patches the exact output indices.
FORCE_INLINE void overwrite_index_row_with_sentinel(
    volatile tt_l1_ptr uint16_t* idx_ptr, uint32_t row, uint32_t n_activated_experts, uint16_t sentinel) {
    uint32_t face_row = row % rows_per_face;
    uint32_t face_base = (row < rows_per_face) ? 0 : 2;

    for (uint32_t expert = 0; expert < n_activated_experts; expert++) {
        uint32_t face_col = expert % columns_per_face;
        uint32_t face = face_base + (expert < columns_per_face ? 0 : 1);
        uint32_t offset = face * elements_per_face + face_row * columns_per_face + face_col;
        idx_ptr[offset] = sentinel;
    }
}
