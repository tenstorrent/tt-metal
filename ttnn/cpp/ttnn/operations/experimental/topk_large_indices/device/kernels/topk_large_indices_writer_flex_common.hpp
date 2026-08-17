// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Output-emission helpers shared by the FLEX output writers (writer_flex.cpp,
// writer_tree_flex.cpp). The flex writers serve the opt-in output formats
// (tile_output=true and/or index_dtype=UINT16); the default ROW_MAJOR/UINT32
// program keeps its original writer sources byte-identical and never includes
// this header.
//
// Source data contract (identical to the default writers): per output row the
// compute kernel pushes one CB page holding the k-element result as 16-element
// slices — already row-major for the 512 LLK window (source_slices_per_row ==
// 32), in pack_untilize face-pair order for the 1024/2048 windows. Indices
// travel as raw 32-bit words (64 B per slice), values as bf16 (32 B per slice).
//
// TILE-layout emission: an output row r of a [rows_2d, k] 2D slice lands in
// tile row (r >> 5) at in-tile row (r & 31). Within a 32x32 tile the four
// 16x16 faces are stored contiguously (f0 rows0-15/cols0-15, f1 rows0-15/
// cols16-31, f2, f3), so each 16-element output slice is one contiguous
// face-row run:
//   dst_slice d -> tile column d>>1, face (in_tile_r<16 ? 0 : 2) + (d&1),
//   byte offset ((face * 256) + (in_tile_r & 15) * 16) * elem_bytes.
// All runs start at 32 B multiples, satisfying Blackhole's 16 B NoC DRAM
// WRITE alignment (writes, unlike reads, have no 64 B requirement).
//
// TILE padding rows are zero-filled from a small zeroed L1 slab so the routed
// composite matches what tilize_with_val_padding produced (pad value 0).

#pragma once

#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

namespace topk_large_indices_writer_flex {

// dst (row-major) slice index -> source-CB slice index. Mirrors the
// pack_untilize face-pair order documented in writer.cpp::copy_row_to_scratch.
template <uint32_t source_slices_per_row>
FORCE_INLINE uint32_t source_slice_of(uint32_t dst_slice) {
    if constexpr (source_slices_per_row == 32) {
        return dst_slice;  // LLK window 512: the CB page is already row-major
    } else {
        static_assert(source_slices_per_row == 64 || source_slices_per_row == 128);
        const uint32_t tile_col = dst_slice >> 2;
        const uint32_t face_col = dst_slice & 0x1;
        const uint32_t face_row_offset = (dst_slice & 0x2) ? source_slices_per_row / 2 : 0;
        return (2 * tile_col) + face_col + face_row_offset;
    }
}

// Narrows the row's uint32 index words into a row-major uint16 row in scratch,
// applying the face-pair reorder in the same pass. The truncation maps real
// winners (< 65536, guaranteed by the op's index_dtype=UINT16 validation)
// losslessly and the 0xFFFFFFFF -inf sentinel to 0xFFFF — the same values a
// UINT32 -> UINT16 typecast op produces.
template <uint32_t source_slices_per_row, uint32_t output_slices_per_row>
FORCE_INLINE void narrow_row_to_u16(uint32_t src_base, uint32_t dst_base) {
    for (uint32_t d = 0; d < output_slices_per_row; ++d) {
        const uint32_t s = source_slice_of<source_slices_per_row>(d);
        volatile tt_l1_ptr uint32_t* src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src_base + s * 64);
        volatile tt_l1_ptr uint32_t* dst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst_base + d * 32);
        // Pack two narrowed uint16 per 32-bit store (little-endian) — halves
        // the RISC store count vs per-element uint16 stores.
        for (uint32_t e = 0; e < 8; ++e) {
            const uint32_t lo = src[2 * e];
            const uint32_t hi = src[2 * e + 1];
            dst[e] = (lo & 0xFFFFu) | (hi << 16);
        }
    }
}

// Scatters one output row's slices into their TILE positions.
// src_slice_bytes: stride of the 16-element slices in L1 (64 for uint32 CB
// data, 32 for bf16 CB data or a narrowed uint16 scratch row).
// src_row_major: the L1 source is already in row-major slice order (a scratch
// row, or the 512-window CB page); otherwise the face-pair mapping applies.
template <
    uint32_t source_slices_per_row,
    uint32_t output_slices_per_row,
    uint32_t src_slice_bytes,
    uint32_t elem_bytes,
    bool src_row_major,
    typename TensorAccessorT>
FORCE_INLINE void issue_tile_row_scatter(
    const Noc& noc,
    const TensorAccessorT& tensor,
    uint32_t src_base,
    uint32_t tile_row,
    uint32_t in_tile_r,
    uint32_t tiles_per_out_row) {
    constexpr uint32_t slice_bytes = 16 * elem_bytes;
    // Byte offset of this row's face-col-0 run within a tile.
    const uint32_t base_off = (((in_tile_r >> 4) & 1) * 512 + (in_tile_r & 15) * 16) * elem_bytes;
    for (uint32_t d = 0; d < output_slices_per_row; ++d) {
        const uint32_t s = src_row_major ? d : source_slice_of<source_slices_per_row>(d);
        const uint32_t page = tile_row * tiles_per_out_row + (d >> 1);
        const uint32_t dst_off = base_off + (d & 1) * 256 * elem_bytes;
        noc.async_write(
            CoreLocalMem<uint32_t>(src_base + s * src_slice_bytes),
            tensor,
            slice_bytes,
            {.offset_bytes = 0},
            {.page_id = page, .offset_bytes = dst_off});
    }
}

// Zero-fills output rows [pad_from, 32) of one output tile row (all its tile
// columns) from the zeroed slab at zero_base. pad_from must be in [1, 31].
// The pad region decomposes into at most 3 contiguous face runs per tile; the
// largest (both bottom faces) is 512 * elem_bytes <= 2048 bytes, which bounds
// the slab size.
template <uint32_t elem_bytes, typename TensorAccessorT>
FORCE_INLINE void issue_tile_row_pad(
    const Noc& noc,
    const TensorAccessorT& tensor,
    uint32_t zero_base,
    uint32_t tile_row,
    uint32_t pad_from,
    uint32_t tiles_per_out_row) {
    const auto write_zeros = [&](uint32_t page, uint32_t dst_off, uint32_t len) {
        noc.async_write(
            CoreLocalMem<uint32_t>(zero_base),
            tensor,
            len,
            {.offset_bytes = 0},
            {.page_id = page, .offset_bytes = dst_off});
    };
    for (uint32_t tile_col = 0; tile_col < tiles_per_out_row; ++tile_col) {
        const uint32_t page = tile_row * tiles_per_out_row + tile_col;
        if (pad_from < 16) {
            const uint32_t top_len = (16 - pad_from) * 16 * elem_bytes;
            write_zeros(page, pad_from * 16 * elem_bytes, top_len);          // f0 rows [pad_from, 16)
            write_zeros(page, (256 + pad_from * 16) * elem_bytes, top_len);  // f1 rows [pad_from, 16)
            write_zeros(page, 512 * elem_bytes, 512 * elem_bytes);           // f2 + f3 entirely
        } else {
            const uint32_t q = pad_from - 16;
            const uint32_t bot_len = (16 - q) * 16 * elem_bytes;
            write_zeros(page, (512 + q * 16) * elem_bytes, bot_len);  // f2 rows [q, 16)
            write_zeros(page, (768 + q * 16) * elem_bytes, bot_len);  // f3 rows [q, 16)
        }
    }
}

// Zeroes the pad slab once at kernel start (512 32-bit words = 2048 bytes).
FORCE_INLINE uint32_t init_pad_slab(uint32_t cb_pad_zero) {
    CircularBuffer pad_cb(cb_pad_zero);
    pad_cb.reserve_back(1);
    const uint32_t base = pad_cb.get_write_ptr();
    volatile tt_l1_ptr uint32_t* slab = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base);
    for (uint32_t i = 0; i < 512; ++i) {
        slab[i] = 0;
    }
    return base;
}

// --- ROW_MAJOR emission helpers (verbatim structure of writer.cpp's) ---

template <uint32_t source_slices_per_row, uint32_t output_slices_per_row, uint32_t slice_bytes>
FORCE_INLINE void copy_row_to_scratch(CircularBuffer& src_cb, CircularBuffer& scratch_cb, const Noc& noc) {
    static_assert(source_slices_per_row == 64 || source_slices_per_row == 128);
    static_assert(output_slices_per_row >= 1 && output_slices_per_row <= source_slices_per_row);
    constexpr uint32_t transfer_bytes = slice_bytes;
    static_assert(transfer_bytes <= NOC_MAX_BURST_SIZE);

    const uint32_t src_base = src_cb.get_read_ptr();
    const uint32_t dst_base = scratch_cb.get_write_ptr();
    const uint32_t noc_id = noc.get_noc_id();
    const auto local_src = [noc_id](uint32_t addr) {
        return noc_traits_t<UnicastEndpoint>::src_args_type{
            .noc_x = static_cast<uint32_t>(my_x[noc_id]), .noc_y = static_cast<uint32_t>(my_y[noc_id]), .addr = addr};
    };

    noc.set_async_read_state<NocOptions::DEFAULT, NOC_MAX_BURST_SIZE>(
        UnicastEndpoint{}, transfer_bytes, local_src(src_base));

    for (uint32_t dst_slice = 0; dst_slice < output_slices_per_row; ++dst_slice) {
        const uint32_t src_slice = source_slice_of<source_slices_per_row>(dst_slice);
        const uint32_t src_addr = src_base + src_slice * slice_bytes;
        const uint32_t dst_addr = dst_base + dst_slice * slice_bytes;
        noc.async_read_with_state<NocOptions::DEFAULT, NOC_MAX_BURST_SIZE>(
            UnicastEndpoint{},
            CoreLocalMem<uint32_t>(dst_addr),
            transfer_bytes,
            local_src(src_addr),
            {.offset_bytes = 0});
    }
    noc.async_read_barrier();
}

template <
    uint32_t source_slices_per_row,
    uint32_t output_slices_per_row,
    uint32_t slice_bytes,
    typename TensorAccessorT>
FORCE_INLINE void issue_reordered_row_write(
    CircularBuffer& src_cb,
    CircularBuffer& scratch_cb,
    const Noc& noc,
    const TensorAccessorT& tensor,
    uint32_t row,
    uint32_t row_bytes) {
    src_cb.wait_front(1);
    scratch_cb.reserve_back(1);
    copy_row_to_scratch<source_slices_per_row, output_slices_per_row, slice_bytes>(src_cb, scratch_cb, noc);
    src_cb.pop_front(1);

    scratch_cb.push_back(1);
    scratch_cb.wait_front(1);
    noc.async_write(scratch_cb, tensor, row_bytes, {.offset_bytes = 0}, {.page_id = row, .offset_bytes = 0});
}

template <typename TensorAccessorT>
FORCE_INLINE void issue_contiguous_row_write(
    CircularBuffer& src_cb, const Noc& noc, const TensorAccessorT& tensor, uint32_t row, uint32_t row_bytes) {
    src_cb.wait_front(1);
    noc.async_write(src_cb, tensor, row_bytes, {.offset_bytes = 0}, {.page_id = row, .offset_bytes = 0});
}

}  // namespace topk_large_indices_writer_flex
