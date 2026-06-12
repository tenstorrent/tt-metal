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
//   5 layer_idx    6 num_layers 7 Wt                   8 cache_CHtWt   9 slot_idx
struct ZeroPadChipWork {
    uint32_t count;            // number of seq-tiles this chip zeroes (0 => nothing to do)
    uint32_t base_local_tile;  // first owned local seq-tile index
    uint32_t first_partial;    // 1 if the first owned tile is the boundary (partial) tile
    uint32_t row_start;        // real rows [0,row_start) kept in the partial tile
    uint32_t Wt;               // width tiles per seq-tile
    uint32_t batch_page_base;  // page offset of this (slot,layer) batch slot in the cache
};

inline ZeroPadChipWork zero_pad_compute_chip_work() {
    const uint32_t my = get_common_arg_val<uint32_t>(0);
    const uint32_t sp = get_common_arg_val<uint32_t>(1);
    const uint32_t chunk_local = get_common_arg_val<uint32_t>(2);
    const uint32_t v = get_common_arg_val<uint32_t>(3);
    const uint32_t pad = get_common_arg_val<uint32_t>(4);
    const uint32_t layer = get_common_arg_val<uint32_t>(5);
    const uint32_t num_layers = get_common_arg_val<uint32_t>(6);
    const uint32_t Wt = get_common_arg_val<uint32_t>(7);
    const uint32_t cache_CHtWt = get_common_arg_val<uint32_t>(8);
    const uint32_t slot = get_common_arg_val<uint32_t>(9);

    ZeroPadChipWork w{0, 0, 0, 0, Wt, (slot * num_layers + layer) * cache_CHtWt};

    const uint32_t pad_end = ((v + pad - 1) / pad) * pad;  // ceil_pad(v)
    if (pad_end == v) {
        return w;  // valid_global pad-aligned: no pad tail
    }
    const uint32_t csg = sp * chunk_local;
    const uint32_t first_gt = v / 32;       // first window seq-tile (global)
    const uint32_t last_gt = pad_end / 32;  // exclusive
    const uint32_t row_start = v % 32;
    bool found = false;
    for (uint32_t gt = first_gt; gt < last_gt; ++gt) {
        const uint32_t token = gt * 32;
        const uint32_t chip = (token % csg) / chunk_local;
        if (chip != my) {
            continue;
        }
        const uint32_t local_token = (token / csg) * chunk_local + (token % csg) % chunk_local;
        const uint32_t local_tile = local_token / 32;
        if (!found) {
            w.base_local_tile = local_tile;
            w.first_partial = (gt == first_gt && row_start != 0) ? 1u : 0u;
            w.row_start = row_start;
            found = true;
        }
        ++w.count;
    }
    return w;
}
