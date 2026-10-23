// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"

// Token-major gates. g and beta arrive as the model produces them, [B, T, HV] TILE in fp32 or bf16: source tile
// (b, rt, ht) holds tokens rt*32.. of heads ht*32... The compute consumes, per (head, chunk), Ct fp32 tiles whose
// column 0 is the head's gate; nothing reads the other columns (decay = tril @ g is column-wise and every consumer
// broadcasts column 0, chunk_gdn_math.hpp decay_all), so per chunk tile the reader fetches the source tile that holds
// the head's column into the CB page and, once it has landed, moves that column into column 0 widened to fp32 with
// the rows past T zeroed. The other columns keep what the fetch left there (other heads' gates, or stale bytes above
// a bf16 tile): finite or not, they never reach a result.
namespace gdn_gate {

struct Source {
    uint32_t T;           // tokens
    uint32_t HV;          // value heads per batch row
    uint32_t Tt;          // row tiles of the source, (T + 31) / 32
    uint32_t Ht;          // head tiles of the source, (HV + 31) / 32
    uint32_t tile_bytes;  // the source's tile: 4096 (fp32) or 2048 (bf16)
};

inline Source source(uint32_t T, uint32_t HV, uint32_t tile_bytes) {
    return {T, HV, (T + 31) / 32, (HV + 31) / 32, tile_bytes};
}

// Reserve Ct pages of `cb_id` (each page_bytes, the fp32 tile) and post the reads of head bh's chunk c: page r takes
// source tile (b, c*Ct + r, hv/32), or nothing when that row tile lies past the source (settle writes zeros).
// No barrier, no push. Out of line: the readers call it from two places and the prep program's binaries sit
// close to the kernel-config budget.
template <typename Acc>
__attribute__((noinline)) static void issue(
    const Noc& noc,
    const Acc& acc,
    uint32_t cb_id,
    uint32_t bh,
    uint32_t c,
    uint32_t Ct,
    const Source& s,
    uint32_t page_bytes) {
    const uint32_t b = bh / s.HV;
    const uint32_t hv = bh % s.HV;
    CircularBuffer cb(cb_id);
    cb.reserve_back(Ct);
    for (uint32_t r = 0; r < Ct; r++) {
        const uint32_t rt = c * Ct + r;
        if (rt < s.Tt) {
            noc.async_read(
                acc, cb, s.tile_bytes, {.page_id = (b * s.Tt + rt) * s.Ht + hv / 32}, {.offset_bytes = r * page_bytes});
        }
    }
}

// After the read barrier, before the push: column hv%32 of each page -> column 0 as fp32, rows past T zero. A 32x32
// tile is four 16x16 faces, (0) rows 0-15 cols 0-15, (1) rows 0-15 cols 16-31, (2) rows 16-31 cols 0-15, (3) the
// rest: row i of column j is element (i >> 4) * 512 + (i & 15) * 16 + (j >> 4) * 256 + (j & 15).
__attribute__((noinline)) static void settle(
    uint32_t cb_id, uint32_t bh, uint32_t c, uint32_t Ct, const Source& s, uint32_t page_bytes) {
    const uint32_t j = (bh % s.HV) % 32;
    const uint32_t col_j = (j >> 4) * 256 + (j & 15);
    const bool bf16 = s.tile_bytes < page_bytes;
    const uint32_t base = CircularBuffer(cb_id).get_write_ptr();
    invalidate_l1_cache();
    for (uint32_t r = 0; r < Ct; r++) {
        const uint32_t tok0 = (c * Ct + r) * 32;
        volatile tt_l1_ptr uint32_t* w32 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base + r * page_bytes);
        volatile tt_l1_ptr uint16_t* w16 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(base + r * page_bytes);
        uint32_t col[32];
        for (uint32_t i = 0; i < 32; i++) {
            const uint32_t idx = col_j + (i >> 4) * 512 + (i & 15) * 16;
            col[i] = (tok0 + i < s.T) ? (bf16 ? (static_cast<uint32_t>(w16[idx]) << 16) : w32[idx]) : 0u;
        }
        for (uint32_t i = 0; i < 32; i++) {
            w32[(i >> 4) * 512 + (i & 15) * 16] = col[i];
        }
    }
    asm volatile("fence");
}

}  // namespace gdn_gate
