// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// nomic_sdpa: key padding (mask_mode = key_padding), shared by the reader (cb_key_t's producer) and whichever RISC
// produces cb_value_aug.
//
// Keys j >= L_b of sequence b must take zero weight. The compute kernel is unchanged: it still scores every key of the
// resident range. Instead the producers zero, for every key j >= L_b, after the slot's data has landed and before it
// is pushed:
//   - its K row (the reader, cb_key_t's only producer), so every padded score is exactly 0;
//   - its row of the HD value tiles AND of the ones tile (the cb_value_aug producer). Then
//       O = sum_j P_j v_j   and   l = sum_j P_j * 1_j
//     both drop the padded keys exactly (P_j * 0 = 0), so out = O / l is the softmax over the valid keys.
// The padded scores (0) still enter the row max: m = max(valid max, 0). Softmax is invariant to m and the padded
// weights exp(0 - m) <= 1 cannot overflow; the valid weights underflow only where every valid score of a row is below
// the fast exp's range (~-88; the model's scores reach ~ +-21). Zeroing K matters: with the padded keys' own K left
// in, a padded score ~60 above a row's only valid one (golden peaky profile, L_b = 1) underflowed l to 0 (NaN row).
//
// Cost (measured 8x512 L1 with a length-sorted batch): the first form re-read every padded tile and
// zeroed it in 512-byte pieces, 52.7 -> 63.2 us; with every row one token short (boundary rows only) it was free. So:
//   - reads skip the key tile-rows that are wholly padded (st >= ceil(L_b / 32)), and the K / V multicasts carry only
//     the valid tile-rows: fewer bytes on the NoCs, the op's binding resource;
//   - a slot's padded tile-rows stay zero until a read writes them again, so each producer tracks, per ring slot, the
//     first tile-row from which the slot is known zero (`known`), and zeroes only [ceil(L_b / 32), known); with the
//     batch sorted longest first (embed()) a slot's padded range only grows, so each tile is zeroed about once. A
//     slot's first use (segment < kv_depth) knows nothing (known = N): it may hold another call's data, or a sender's
//     unread tile-rows after a multicast;
//   - the boundary tile-row (read whole every time) has its rows >= L_b mod 32 zeroed every time;
//   - a range is zeroed by seeding 2 KiB from MEM_ZEROS, then doubling it with local copies (log2 transfers).
// The ones column is written once per ring slot and persists across segments (the multicast never moves it), so the
// [V | 1] producer also tracks how many keys of each slot's ones column are still 1.0 and rewrites it whole (and then
// forgets what it knew zero) when a later pair of that slot has more valid keys.
//
// A tile is 32 x 32 bf16 in four 16 x 16 faces (f0 rows 0-15 cols 0-15, f1 rows 0-15 cols 16-31, f2 rows 16-31 cols
// 0-15, f3 rows 16-31 cols 16-31), 32 bytes a face row.

#pragma once

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

namespace nomic_sdpa {

constexpr uint32_t kTileRows = 32;
constexpr uint32_t kFaceRows = 16;
constexpr uint32_t kFaceBytes = 512;    // 16 x 16 bf16
constexpr uint32_t kFaceRowBytes = 32;  // 16 bf16

// Zero [addr, addr + bytes) of this core's L1 (bytes a multiple of 32): NoC reads from MEM_ZEROS. Not barriered.
inline __attribute__((always_inline)) void zero_l1_async(uint32_t addr, uint32_t bytes) {
    const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
    while (bytes > 0) {
        const uint32_t chunk = bytes < MEM_ZEROS_SIZE ? bytes : MEM_ZEROS_SIZE;
        noc_async_read(zeros, addr, chunk);
        addr += chunk;
        bytes -= chunk;
    }
}

// Zero rows [first_row, 32) of one bf16 32 x 32 tile at `tile`. Not barriered.
inline __attribute__((always_inline)) void zero_tile_rows_async(uint32_t tile, uint32_t first_row) {
    if (first_row < kFaceRows) {
        const uint32_t skip = first_row * kFaceRowBytes;
        zero_l1_async(tile + skip, kFaceBytes - skip);               // f0
        zero_l1_async(tile + kFaceBytes + skip, kFaceBytes - skip);  // f1
        zero_l1_async(tile + 2 * kFaceBytes, 2 * kFaceBytes);        // f2, f3
    } else {
        const uint32_t skip = (first_row - kFaceRows) * kFaceRowBytes;
        zero_l1_async(tile + 2 * kFaceBytes + skip, kFaceBytes - skip);  // f2
        zero_l1_async(tile + 3 * kFaceBytes + skip, kFaceBytes - skip);  // f3
    }
}

// The keys a segment reads: the first page of its key tile-rows in the K / V tensors and its real keys (the rows
// past them are zeroed). A (b, h) pair's whole key range, or in segmented mode one text of a packed row.
struct Unit {
    uint32_t first_page;
    uint32_t valid;
};

constexpr uint32_t kZeroSeedBytes = 2048;

// Zero [addr, addr + bytes) of this core's L1 (bytes a multiple of 32): 2 KiB from MEM_ZEROS, then doubling local
// copies of what is already zero. Barriered (each doubling reads what the step before wrote).
inline __attribute__((always_inline)) void zero_l1(uint32_t addr, uint32_t bytes) {
    if (bytes == 0) {
        return;
    }
    const uint32_t seed = bytes < kZeroSeedBytes ? bytes : kZeroSeedBytes;
    zero_l1_async(addr, seed);
    noc_async_read_barrier();
    const uint64_t zeros = get_noc_addr(addr);
    for (uint32_t done = seed; done < bytes;) {
        const uint32_t step = done < bytes - done ? done : bytes - done;
        noc_async_read(zeros, addr + done, step);
        noc_async_read_barrier();
        done += step;
    }
}

// Key tile-rows of a sequence with `valid_keys` real keys: ceil(L_b / 32).
inline __attribute__((always_inline)) uint32_t valid_tile_rows(uint32_t valid_keys) {
    return (valid_keys + kTileRows - 1) / kTileRows;
}

// [V | 1] slot (key tile-row st at st * PV tiles: HD value tiles, then the ones tile): zero the rows of keys >= L_b:
// the boundary tile-row's tail rows, and the whole tile-rows [ceil(L_b / 32), known). known <- ceil(L_b / 32).
template <uint32_t N, uint32_t PV, uint32_t tile_bytes>
inline __attribute__((always_inline)) void mask_value_slot(uint32_t v_base, uint32_t valid_keys, uint32_t& known) {
    static_assert(tile_bytes == 4 * kFaceBytes, "key masking assumes 32 x 32 bf16 tiles");
    const uint32_t rows = valid_tile_rows(valid_keys);
    const uint32_t partial = valid_keys % kTileRows;
    if (partial != 0) {
        for (uint32_t t = 0; t < PV; ++t) {
            zero_tile_rows_async(v_base + ((rows - 1) * PV + t) * tile_bytes, partial);
        }
    }
    if (rows < known) {
        zero_l1(v_base + rows * PV * tile_bytes, (known - rows) * PV * tile_bytes);
    }
    noc_async_read_barrier();
    known = rows;
}

// cb_key_t slot (K^T tile order: tile dt * N + st holds keys st * 32 .. st * 32 + 31 as its rows): the same for K.
template <uint32_t N, uint32_t HD, uint32_t tile_bytes>
inline __attribute__((always_inline)) void mask_key_slot(uint32_t k_base, uint32_t valid_keys, uint32_t& known) {
    static_assert(tile_bytes == 4 * kFaceBytes, "key masking assumes 32 x 32 bf16 tiles");
    const uint32_t rows = valid_tile_rows(valid_keys);
    const uint32_t partial = valid_keys % kTileRows;
    for (uint32_t dt = 0; dt < HD; ++dt) {
        const uint32_t base = k_base + dt * N * tile_bytes;
        if (partial != 0) {
            zero_tile_rows_async(base + (rows - 1) * tile_bytes, partial);
        }
        if (rows < known) {
            zero_l1(base + rows * tile_bytes, (known - rows) * tile_bytes);
        }
    }
    noc_async_read_barrier();
    known = rows;
}

}  // namespace nomic_sdpa
