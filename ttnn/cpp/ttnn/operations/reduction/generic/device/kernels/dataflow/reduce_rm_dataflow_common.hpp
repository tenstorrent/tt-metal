// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <api/dataflow/dataflow_api.h>
#include <cstdint>
#include <tt-metalium/constants.hpp>
#include "ttnn/cpp/ttnn/operations/pool/device/kernels/experimental_device_api.hpp"

#define RM_DF_ALWI inline __attribute__((always_inline))

// Map a logical (h_idx, w_idx) within a tile to its flat datum offset in tile-layout (4-face) storage.
// Used by the writers that scatter sparse outputs of `reduce()` — column 0 of every row for W-reduce,
// row 0 of every column for H-reduce — into row-major output pages.
RM_DF_ALWI uint32_t get_tilized_idx(uint32_t h_idx, uint32_t w_idx) {
    constexpr uint32_t tile_height = tt::constants::TILE_HEIGHT;
    constexpr uint32_t tile_width = tt::constants::TILE_WIDTH;
    constexpr uint32_t half_tile_height = tile_height / 2;
    constexpr uint32_t half_tile_width = tile_width / 2;

    if (h_idx < half_tile_height && w_idx < half_tile_width) {
        return h_idx * half_tile_width + w_idx;
    }
    if (h_idx < half_tile_height && w_idx >= half_tile_width) {
        return h_idx * half_tile_width + (w_idx % half_tile_width) + half_tile_height * half_tile_width;
    }
    if (h_idx >= half_tile_height && w_idx < half_tile_width) {
        return (h_idx % half_tile_height) * half_tile_width + w_idx + half_tile_height * tile_width;
    }
    return (h_idx % half_tile_height) * half_tile_width + (w_idx % half_tile_width) +
           half_tile_height * (tile_width + half_tile_width);
}

// Same packing convention as pool_kernels_common fill_with_val: repeat uint16 `val` for n uint16 slots.
RM_DF_ALWI void rm_fill_with_val_bf16(uint32_t begin_addr, uint32_t num_u16, uint16_t val) {
    volatile tt_l1_ptr uint32_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(begin_addr);
    uint32_t value = static_cast<uint32_t>(val) | (static_cast<uint32_t>(val) << 16);
    uint32_t num_pairs = num_u16 / 2;
    for (uint32_t i = 0; i < num_pairs; ++i) {
        ptr[i] = value;
    }
    if (num_u16 & 1) {
        volatile tt_l1_ptr uint16_t* ptr16 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(begin_addr);
        ptr16[num_u16 - 1] = val;
    }
}

RM_DF_ALWI void rm_fill_buffer_with_identity_pattern(
    uint32_t begin_addr, uint32_t num_bytes, uint32_t elem_bytes, uint32_t pattern_bits) {
    if (num_bytes == 0) {
        return;
    }
    if (elem_bytes == 2) {
        const uint16_t v = static_cast<uint16_t>(pattern_bits & 0xFFFFu);
        rm_fill_with_val_bf16(begin_addr, num_bytes / 2, v);
    } else if (elem_bytes == 4) {
        volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(begin_addr);
        const uint32_t n = num_bytes / 4;
        for (uint32_t i = 0; i < n; ++i) {
            p[i] = pattern_bits;
        }
    }
}

//
// Reader helpers for dense RM reduce (W and H paths).
//
// CB layout staged into cb_rm: one page per logical RM row (chunk-wide).
//   page_bytes = wt_in_chunk × TILE_WIDTH × elem_bytes  ( == chunk_row_bytes )
// The reader pushes rm_rows_per_tile (= TILE_HEIGHT) pages per h-tile slab — real rows overlaid
// on the identity-pad template, padded rows beyond a partial last h-tile carry pure identity.
// This matches compute_kernel_lib::tilize's asymmetric mode: each tile-block consumes TILE_HEIGHT
// input pages (one per logical row).
//
// Per W chunk (one wt_in_chunk block at column offset wt_base), every logical row contributes
// `valid_bytes` of real data (clipped at W_logical) and the rest is padded identity.
//

// Byte ranges for one W chunk inside a single logical RM row.
struct RmWChunkBytes {
    uint32_t chunk_bytes;        // wt_in_chunk × TILE_WIDTH × elem_bytes
    uint32_t chunk_start_bytes;  // wt_base × TILE_WIDTH × elem_bytes (offset into the source row)
    uint32_t valid_bytes;        // bytes of real data inside this chunk (0 when entirely in pad region)
};

RM_DF_ALWI RmWChunkBytes
rm_compute_w_chunk_bytes(uint32_t wt_base, uint32_t wt_in_chunk, uint32_t valid_row_bytes, uint32_t elem_bytes) {
    const uint32_t chunk_bytes = wt_in_chunk * tt::constants::TILE_WIDTH * elem_bytes;
    const uint32_t chunk_start_bytes = wt_base * tt::constants::TILE_WIDTH * elem_bytes;
    uint32_t valid_bytes = 0;
    if (chunk_start_bytes < valid_row_bytes) {
        const uint32_t remaining = valid_row_bytes - chunk_start_bytes;
        valid_bytes = remaining < chunk_bytes ? remaining : chunk_bytes;
    }
    return {chunk_bytes, chunk_start_bytes, valid_bytes};
}

// Fill a freshly reserved cb_rm region with the padding-identity template (0 / -inf / +inf as
// appropriate). `region_bytes` is the total byte span to fill, which the caller sets to
// rm_rows_per_tile * page_bytes when reserving a full slab worth of row pages at once. The helper
// loops because the clear template is one tile worth and may be smaller than the region.
//
// ClearTemplateSrc is the opaque return type of experimental::local_addr() — templated so callers
// don't need to spell it out.
template <typename ClearTemplateSrc>
RM_DF_ALWI void rm_fill_page_with_clear_template(
    Noc& noc,
    experimental::CB& cb_rm,
    uint32_t region_bytes,
    const ClearTemplateSrc& clear_template_src,
    uint32_t clear_template_bytes) {
    UnicastEndpoint self_ep;
    uint32_t pad_offset = 0;
    while (pad_offset < region_bytes) {
        const uint32_t copy_bytes =
            (region_bytes - pad_offset) < clear_template_bytes ? (region_bytes - pad_offset) : clear_template_bytes;
        noc.async_read(self_ep, cb_rm, copy_bytes, clear_template_src, {.offset_bytes = pad_offset});
        pad_offset += copy_bytes;
    }
}
