// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "api/core_local_mem.h"
#include "api/dataflow/noc.h"

namespace routing_metadata {

constexpr uint32_t kBlockRows = 32;
constexpr uint32_t kWordBytes = sizeof(uint32_t);
constexpr uint32_t kFaceWords = 16 * 16;
constexpr uint32_t kFaceBytes = kFaceWords * kWordBytes;
// A tile is four 16 x 16 faces, each row-major; its columns 0-15 are face 0 (rows 0-15, at byte 0)
// and face 2 (rows 16-31, at byte 2048). Read back to back, they put token i's choice k at word
// 16 * i + k.
constexpr uint32_t kFaceTwoOffset = 2 * kFaceBytes;
constexpr uint32_t kFacePairBytes = 2 * kFaceBytes;
constexpr uint32_t kWordsPerToken = 16;

// Issues the reads of faces 0 and 2 of the (1, 1, tokens, top_k) tensor's tile pages
// [first_page, first_page + num_tile_rows), one face pair after another from destination. The
// caller waits for them with async_read_barrier.
template <typename Accessor>
void read_face_pairs(
    Noc& noc, const Accessor& tensor, uint32_t destination, uint32_t first_page, uint32_t num_tile_rows) {
    for (uint32_t tile_row = 0; tile_row < num_tile_rows; ++tile_row) {
        const uint32_t pair = destination + tile_row * kFacePairBytes;
        noc.async_read(tensor, CoreLocalMem<uint32_t>(pair), kFaceBytes, {.page_id = first_page + tile_row}, {});
        noc.async_read(
            tensor,
            CoreLocalMem<uint32_t>(pair + kFaceBytes),
            kFaceBytes,
            {.page_id = first_page + tile_row, .offset_bytes = kFaceTwoOffset},
            {});
    }
}

}  // namespace routing_metadata
