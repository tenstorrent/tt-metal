// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <api/dataflow/dataflow_api.h>
#include <cstdint>
#include <tt-metalium/constants.hpp>
#include "experimental/noc.h"
#include "experimental/circular_buffer.h"
#include "experimental/tensor.h"
#include "ttnn/cpp/ttnn/operations/pool/device/kernels/experimental_device_api.hpp"

#define RM_DF_ALWI inline __attribute__((always_inline))

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
// Reader helpers for dense RM W reduce.
//
// Page layout staged into cb_rm (one page per "slab"):
//   bytes = rm_rows_per_tile rows × wt_in_chunk × TILE_WIDTH × elem_bytes
//   logically: rm_rows_per_tile logical RM rows, each holding (wt_in_chunk × TILE_WIDTH) tile-domain columns.
//
// Per W chunk (one wt_in_chunk block at column offset wt_base), every logical row contributes
// `valid_bytes_this_chunk` of real data (clipped at W_logical) and the rest is the padding-identity pattern.
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

// Per-slab placement inside one H chunk (where to read packed rows from / how many rows are real).
struct RmSlabInfo {
    uint32_t first_page;
    uint32_t rows_in_pack;
};

// Cap mirrors host (factory): ht_tiles_per_chunk = std::min<uint32_t>(8, ...).
constexpr uint32_t RM_MAX_HT_TILES_PER_CHUNK = 8;

// Layout slab metadata for one H chunk; advances cursors so the next call resumes correctly.
RM_DF_ALWI uint32_t rm_precompute_slabs_for_h_chunk(
    RmSlabInfo* slabs,
    uint32_t ht_in_chunk,
    uint32_t rm_rows_per_tile,
    uint32_t& packed_row_base,
    uint32_t& rows_remaining) {
    uint32_t total_rows_this_block = 0;
    for (uint32_t hti = 0; hti < ht_in_chunk; ++hti) {
        const uint32_t rows = rows_remaining < rm_rows_per_tile ? rows_remaining : rm_rows_per_tile;
        slabs[hti] = {packed_row_base, rows};
        packed_row_base += rows;
        rows_remaining -= rows;
        total_rows_this_block += rows;
    }
    return total_rows_this_block;
}

// Fill a freshly reserved cb_rm page with the padding-identity template (0 / -inf / +inf as appropriate).
// ClearTemplateSrc is the opaque return type of experimental::local_addr() — templated so callers don't
// need to spell it out.
template <typename ClearTemplateSrc>
RM_DF_ALWI void rm_fill_page_with_clear_template(
    experimental::Noc& noc,
    experimental::CB& cb_rm,
    uint32_t page_bytes,
    const ClearTemplateSrc& clear_template_src,
    uint32_t clear_template_bytes) {
    experimental::UnicastEndpoint self_ep;
    uint32_t pad_offset = 0;
    while (pad_offset < page_bytes) {
        const uint32_t copy_bytes =
            (page_bytes - pad_offset) < clear_template_bytes ? (page_bytes - pad_offset) : clear_template_bytes;
        noc.async_read(self_ep, cb_rm, copy_bytes, clear_template_src, {.offset_bytes = pad_offset});
        pad_offset += copy_bytes;
    }
}

// Issue one async read per logical row in the slab, covering all w_tiles_in_chunk for that row at once
// (the RM source has contiguous bytes across W within a row, so one read is strictly more efficient than
// splitting into per-tile reads — same pattern as pool/reader_pool_2d.cpp's contiguous-read path).
template <typename TensorAccessorT>
RM_DF_ALWI void rm_read_slab_into_page(
    experimental::Noc& noc,
    experimental::CB& cb_rm,
    const TensorAccessorT& tensor_accessor,
    const RmSlabInfo& slab,
    const RmWChunkBytes& w) {
    if (w.valid_bytes == 0) {
        return;  // Entire W chunk falls in padding; identity fill already covers it.
    }
    for (uint32_t r = 0; r < slab.rows_in_pack; ++r) {
        noc.async_read(
            tensor_accessor,
            cb_rm,
            w.valid_bytes,
            {.page_id = slab.first_page + r, .offset_bytes = w.chunk_start_bytes},
            {.offset_bytes = r * w.chunk_bytes});
    }
}
