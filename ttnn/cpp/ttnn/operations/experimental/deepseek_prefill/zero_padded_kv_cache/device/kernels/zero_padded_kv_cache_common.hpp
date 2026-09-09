// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>

// Shared per-chip pad-window computation for the zero_padded_kv_cache kernels. Each kernel (reader,
// compute, writer) reads the same common runtime args and derives this chip's share of the global pad
// window on-device, so a window that straddles a chip boundary is handled by each chip doing its slice.
//
// Common runtime arg layout (set in create_descriptor; indices 3 and 9 patched per call):
//   0 my_sp_coord  1 sp_factor  2 chunk_local(tokens)  3 valid_global  4 pad_align
//   5 layer_idx    6 num_layers 7 Wt                   8 cache_CH_pages 9 slot_idx
struct ZeroPadTokenRange {
    uint32_t count;
    uint32_t base_local_token;
};

// Validation guarantees that [valid_global, ceil_pad(valid_global)) remains in one block-cyclic slab.
// Intersect that window with this chip's contiguous slice of the slab, then map the intersection to the
// chip-local cache rows. Both TILE and ROW_MAJOR paths derive their native page ranges from this result.
inline ZeroPadTokenRange zero_pad_compute_token_range(uint32_t valid_global) {
    const uint32_t my = get_common_arg_val<uint32_t>(0);
    const uint32_t sp = get_common_arg_val<uint32_t>(1);
    const uint32_t chunk_local = get_common_arg_val<uint32_t>(2);
    const uint32_t pad_align = get_common_arg_val<uint32_t>(4);

    const uint32_t pad_end = ((valid_global + pad_align - 1) / pad_align) * pad_align;
    if (pad_end == valid_global) {
        return {0, 0};
    }

    const uint32_t chunk_global = sp * chunk_local;
    const uint32_t slab = valid_global / chunk_global;
    const uint32_t chip_begin = slab * chunk_global + my * chunk_local;
    const uint32_t chip_end = chip_begin + chunk_local;
    const uint32_t begin = valid_global > chip_begin ? valid_global : chip_begin;
    const uint32_t end = pad_end < chip_end ? pad_end : chip_end;
    if (begin >= end) {
        return {0, 0};
    }

    return {end - begin, slab * chunk_local + begin - chip_begin};
}

struct ZeroPadChipWork {
    uint32_t count;            // number of seq-tiles this chip zeroes (0 => nothing to do)
    uint32_t base_local_tile;  // first owned local seq-tile index
    uint32_t first_partial;    // 1 if the first owned tile is the boundary (partial) tile
    uint32_t row_start;        // real rows [0,row_start) kept in the partial tile
    uint32_t Wt;               // width tiles per seq-tile
    uint32_t batch_page_base;  // page offset of this (slot,layer) batch slot in the cache
};

// Core derivation, taking the two per-call values (slot_idx, valid_global=v) as parameters. The
// metadata path reads them on-device from the metadata tensor and calls this; the scalar path reads
// common args 9/3 via the no-arg overload below. All other inputs (incl valid_global for the window
// math, via zero_pad_compute_token_range) are structural common args.
inline ZeroPadChipWork zero_pad_compute_chip_work(uint32_t slot, uint32_t v) {
    const uint32_t layer = get_common_arg_val<uint32_t>(5);
    const uint32_t num_layers = get_common_arg_val<uint32_t>(6);
    const uint32_t Wt = get_common_arg_val<uint32_t>(7);
    const uint32_t cache_CHtWt = get_common_arg_val<uint32_t>(8);

    ZeroPadChipWork w{0, 0, 0, 0, Wt, (slot * num_layers + layer) * cache_CHtWt};
    const ZeroPadTokenRange range = zero_pad_compute_token_range(v);
    if (range.count == 0) {
        return w;
    }

    constexpr uint32_t tile_height = 32;
    w.base_local_tile = range.base_local_token / tile_height;
    w.count = (range.base_local_token + range.count) / tile_height - range.base_local_token / tile_height;
    w.row_start = range.base_local_token % tile_height;
    w.first_partial = w.row_start != 0 ? 1u : 0u;
    return w;
}

// Scalar path (TILE): read the two per-call values from common args (slot_idx = 9, valid_global = 3).
// The metadata (per-element-tensor) path is TILE-only; ROW_MAJOR uses the scalar signature.
inline ZeroPadChipWork zero_pad_compute_chip_work() {
    return zero_pad_compute_chip_work(get_common_arg_val<uint32_t>(9), get_common_arg_val<uint32_t>(3));
}

// ROW_MAJOR counterpart: one page is one token row, so the exact window [valid_global, pad_end) can
// be zeroed without reading or masking a boundary tile. Validation keeps the window within one
// block-cyclic slab; therefore each chip's owned rows form one contiguous local page range even when
// the global window crosses a chip boundary. ROW_MAJOR is scalar-only: valid_global is common arg 3.
struct ZeroPadRowMajorChipWork {
    uint32_t count;
    uint32_t base_local_row;
    uint32_t batch_page_base;
};

inline ZeroPadRowMajorChipWork zero_pad_compute_row_major_chip_work() {
    const uint32_t layer = get_common_arg_val<uint32_t>(5);
    const uint32_t num_layers = get_common_arg_val<uint32_t>(6);
    const uint32_t cache_CH_pages = get_common_arg_val<uint32_t>(8);
    const uint32_t slot = get_common_arg_val<uint32_t>(9);

    const ZeroPadTokenRange range = zero_pad_compute_token_range(get_common_arg_val<uint32_t>(3));
    return {range.count, range.base_local_token, (slot * num_layers + layer) * cache_CH_pages};
}
