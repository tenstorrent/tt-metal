// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/constants.hpp>

#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

// The compute kernel produces one [heads, v_dim] tiled block per token. The public result is
// [heads, sequence, v_dim], so each source face-row is scattered into the corresponding sequence row.
template <uint32_t vDHt, typename OutputAccessor>
inline void write_tiled_output_row(
    const Noc& noc,
    experimental::CB& out_cb,
    const OutputAccessor& out,
    uint32_t tok,
    uint32_t source_head,
    uint32_t output_head,
    uint32_t sequence_length,
    uint32_t element_bytes) {
    constexpr uint32_t face_width = tt::constants::FACE_WIDTH;
    constexpr uint32_t face_elements = tt::constants::FACE_HW;
    const uint32_t face_row_bytes = face_width * element_bytes;
    const uint32_t face_bytes = face_elements * element_bytes;
    const uint32_t tile_bytes = 4 * face_bytes;
    const uint32_t sequence_tile = tok / tt::constants::TILE_HEIGHT;
    const uint32_t sequence_row = tok % tt::constants::TILE_HEIGHT;
    const uint32_t sequence_tiles = (sequence_length + tt::constants::TILE_HEIGHT - 1) / tt::constants::TILE_HEIGHT;
    const uint32_t source_face_row = (source_head % tt::constants::TILE_HEIGHT) % tt::constants::FACE_WIDTH;
    const uint32_t source_face_base = (source_head % tt::constants::TILE_HEIGHT) / tt::constants::FACE_WIDTH;
    const uint32_t output_face_row = sequence_row % tt::constants::FACE_WIDTH;
    const uint32_t output_face_base = sequence_row / tt::constants::FACE_WIDTH;
    const uint32_t source_tile = (source_head / tt::constants::TILE_HEIGHT) * vDHt;
    const uint32_t output_tile = output_head * sequence_tiles * vDHt + sequence_tile * vDHt;

    for (uint32_t v_tile = 0; v_tile < vDHt; ++v_tile) {
        for (uint32_t face_column = 0; face_column < 2; ++face_column) {
            const uint32_t source_face = source_face_base * 2 + face_column;
            const uint32_t output_face = output_face_base * 2 + face_column;
            noc.async_write(
                out_cb,
                out,
                face_row_bytes,
                {.offset_bytes =
                     (source_tile + v_tile) * tile_bytes + source_face * face_bytes + source_face_row * face_row_bytes},
                {.page_id = output_tile + v_tile,
                 .offset_bytes = output_face * face_bytes + output_face_row * face_row_bytes});
        }
    }
}

template <uint32_t vDHt, typename OutputAccessor>
inline void zero_tiled_output_padding(
    const Noc& noc,
    const OutputAccessor& out,
    experimental::CB& zero_cb,
    uint32_t first_output_head,
    uint32_t head_count,
    uint32_t sequence_length,
    uint32_t element_bytes) {
    const uint32_t valid_rows = sequence_length % tt::constants::TILE_HEIGHT;
    if (valid_rows == 0) {
        return;
    }

    constexpr uint32_t face_width = tt::constants::FACE_WIDTH;
    constexpr uint32_t face_elements = tt::constants::FACE_HW;
    const uint32_t face_row_bytes = face_width * element_bytes;
    const uint32_t face_bytes = face_elements * element_bytes;
    const uint32_t final_sequence_tile = (sequence_length - 1) / tt::constants::TILE_HEIGHT;
    const uint32_t sequence_tiles = final_sequence_tile + 1;
    for (uint32_t head = 0; head < head_count; ++head) {
        const uint32_t output_tile = (first_output_head + head) * sequence_tiles * vDHt + final_sequence_tile * vDHt;
        for (uint32_t row = valid_rows; row < tt::constants::TILE_HEIGHT; ++row) {
            const uint32_t face_row = row % tt::constants::FACE_WIDTH;
            const uint32_t face_base = row / tt::constants::FACE_WIDTH;
            for (uint32_t v_tile = 0; v_tile < vDHt; ++v_tile) {
                for (uint32_t face_column = 0; face_column < 2; ++face_column) {
                    const uint32_t face = face_base * 2 + face_column;
                    noc.async_write_zeros(
                        out,
                        face_row_bytes,
                        {.page_id = output_tile + v_tile,
                         .offset_bytes = face * face_bytes + face_row * face_row_bytes},
                        zero_cb);
                }
            }
        }
    }
}
