// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Fused 3D-neighborhood gather: densely gather a query chunk's window K/V rows from a ROW_MAJOR
// (B, NH, S, D) K/V table into the SDPA compute's cb_k / cb_v, so the flash compute runs dense over
// only real window tokens (no scattered active-tile streaming, no per-row tile padding).
//
// BUILD-OUT IN PROGRESS.
//
// Packing order (matches na3d _flat_indices / the host twin): the box [t0,t1)x[h_lo,h_hi)x[w_lo,w_hi)
// is enumerated T-outer / H-mid / W-inner, giving each in-window key a dense index j in [0, n_box).
// The reader packs key j into cb_k at seqtile j/32, row j%32; the writer masks by the same j->real map,
// so reader and writer agree without exchanging anything.
//
// cb layout the matmuls expect (reverse-engineered from the streaming reader + compute_common.hpp):
//   * K -> cb_k TRANSPOSED tile-grid: tile (seqtile, dtile) at slot dtile*seqtiles_packed + seqtile;
//     QK matmul_blocks(..., transpose=true) does the face transpose. Natural within-tile faces.
//   * V -> cb_v NATURAL tile-grid: tile (seqtile, dtile) at slot seqtile*D_tiles + dtile.
//   * Float16_b tile = 4 faces of 16x16 (2B); element (r,c) at p[face*256 + (r%16)*16 + (c%16)],
//     face = (r>=16?2:0) + (c>=16?1:0).

#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc.h"
#include "api/core_local_mem.h"
#include <tt-metalium/constants.hpp>
#include "cpp/ttnn/operations/transformer/sdpa/device/kernels/windowed_loop_geometry.hpp"

namespace neighborhood_gather {

constexpr uint32_t FH = tt::constants::FACE_HEIGHT;  // 16
constexpr uint32_t FW = tt::constants::FACE_WIDTH;   // 16
constexpr uint32_t FACE_HW = FH * FW;                // 256
constexpr uint32_t TH = tt::constants::TILE_HEIGHT;  // 32
constexpr uint32_t TW = tt::constants::TILE_WIDTH;   // 32

struct BoxDims {
    uint32_t bt, bh, bw;  // box extents
    uint32_t hw;          // bh*bw
    uint32_t n_box;       // bt*bh*bw
};

inline BoxDims box_dims(const NeighborhoodBox& box) {
    BoxDims d;
    d.bt = box.t1 - box.t0;
    d.bh = box.h_hi - box.h_lo;
    d.bw = box.w_hi - box.w_lo;
    d.hw = d.bh * d.bw;
    d.n_box = d.bt * d.hw;
    return d;
}

// Dense packed index j -> real grid coords (t', h', w') and flat token s = t'*HW + h'*W + w'.
inline uint32_t packed_to_flat(uint32_t j, const NeighborhoodBox& box, const BoxDims& d, uint32_t HW, uint32_t W) {
    const uint32_t jt = j / d.hw;
    const uint32_t rem = j % d.hw;
    const uint32_t jh = rem / d.bw;
    const uint32_t jw = rem % d.bw;
    return (box.t0 + jt) * HW + (box.h_lo + jh) * W + (box.w_lo + jw);
}

// Scatter one token's D-wide Float16_b row (`src`, D contiguous elems in L1) into the reserved packed
// block `cb_base`, at packed index j (seqtile j/32, row j%32). D_tiles col-groups; K uses transposed
// grid, V natural. `seqtiles_packed` only used for the transposed mapping.
template <bool transposed_grid>
inline void scatter_row(
    volatile tt_l1_ptr uint16_t* src,
    uint32_t cb_base,
    uint32_t j,
    uint32_t D_tiles,
    uint32_t seqtiles_packed,
    uint32_t tile_bytes) {
    const uint32_t seqtile = j / TH;
    const uint32_t r = j % TH;
    const uint32_t face_idx = (r >= FH) ? 2u : 0u;
    const uint32_t fr = r & (FH - 1);
    // Within one dtile, the token's 32 D-cols split at the FW boundary into two faces, and each half lands
    // CONTIGUOUSLY in its face row: dest[c] = p[face*256 + fr*16 + (c%16)]. So instead of a per-element
    // face-address compute, copy two contiguous FW-runs (the hot path of the whole gather).
    for (uint32_t dd = 0; dd < D_tiles; ++dd) {
        const uint32_t slot = transposed_grid ? (dd * seqtiles_packed + seqtile) : (seqtile * D_tiles + dd);
        volatile tt_l1_ptr uint16_t* p = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(cb_base + slot * tile_bytes);
        volatile tt_l1_ptr uint16_t* s = src + dd * TW;
        // FW=16 Float16_b elems per face-row = 8 uint32 words; the staging stick and the tile faces are
        // both 4-byte aligned, so copy word-wise (halves the load/store count on the dataflow RISC).
        volatile tt_l1_ptr uint32_t* d_lo =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(p + face_idx * FACE_HW + fr * FW);
        volatile tt_l1_ptr uint32_t* d_hi =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(p + (face_idx + 1) * FACE_HW + fr * FW);
        volatile tt_l1_ptr uint32_t* s_lo = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(s);
        volatile tt_l1_ptr uint32_t* s_hi = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(s + FW);
        for (uint32_t c = 0; c < FW / 2; ++c) {
            d_lo[c] = s_lo[c];
            d_hi[c] = s_hi[c];
        }
    }
}

// Gather packed keys [j0, j1) of the box into the reserved block `cb_base` (D_tiles col-groups per
// seqtile). Each token's D-wide ROW_MAJOR stick (page head_page_base + s) is read into its own slot of
// `staging_l1` (which must hold j1-j0 sticks); reads are issued back-to-back and drained with a single
// barrier per `max_inflight` batch (not one barrier per stick -- that serialization was the wall), then
// all sticks are face-scattered. Caller reserves+pre-zeros cb_base (partial last seqtile) and pushes.
template <bool transposed_grid, typename ReaderType>
inline void gather_range(
    Noc& noc,
    const ReaderType& reader,
    uint32_t cb_base,
    uint32_t staging_l1,
    uint32_t stick_bytes,
    uint32_t j0,
    uint32_t j1,
    const NeighborhoodBox& box,
    const BoxDims& d,
    uint32_t HW,
    uint32_t W,
    uint32_t head_page_base,
    uint32_t D_tiles,
    uint32_t seqtiles_packed,
    uint32_t tile_bytes,
    uint32_t max_inflight) {
    // Phase 1: issue all stick reads into distinct staging slots, barriering only every max_inflight
    // to cap outstanding NOC transactions.
    uint32_t inflight = 0;
    for (uint32_t j = j0; j < j1; ++j) {
        const uint32_t s = packed_to_flat(j, box, d, HW, W);
        const uint32_t dst = staging_l1 + (j - j0) * stick_bytes;
        noc.async_read(reader, CoreLocalMem<uint32_t>(dst), stick_bytes, {.page_id = head_page_base + s}, {});
        if (++inflight == max_inflight) {
            noc.async_read_barrier();
            inflight = 0;
        }
    }
    noc.async_read_barrier();
    // Phase 2: face-scatter every landed stick into cb_base.
    for (uint32_t j = j0; j < j1; ++j) {
        volatile tt_l1_ptr uint16_t* src =
            reinterpret_cast<volatile tt_l1_ptr uint16_t*>(staging_l1 + (j - j0) * stick_bytes);
        scatter_row<transposed_grid>(src, cb_base, j - j0, D_tiles, seqtiles_packed, tile_bytes);
    }
}

// W-run COALESCED gather: K/V are uploaded with page = a full W-row (W*D elems, [B,NH,T*H,W*D] ROW_MAJOR),
// so one page-read fetches an entire (t',h') box-row and the box's contiguous w-run [w_lo,w_hi) is sliced
// out of it -- ~bt*bh reads per chunk instead of n_box (per-token). row_bytes = W*D*2 (bf16); D is the
// head dim in elements; head_row_base = (b*NH+head)*(T*H); real (t,h) row = (t0+jt)*H + (h_lo+jh).
template <bool transposed_grid, typename ReaderType>
inline void gather_range_wrun(
    Noc& noc,
    const ReaderType& reader,
    uint32_t cb_base,
    uint32_t staging_l1,
    uint32_t row_bytes,
    uint32_t j0,
    uint32_t j1,
    const NeighborhoodBox& box,
    const BoxDims& d,
    uint32_t H,
    uint32_t head_row_base,
    uint32_t D,
    uint32_t D_tiles,
    uint32_t seqtiles_packed,
    uint32_t tile_bytes) {
    volatile tt_l1_ptr uint16_t* stg = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(staging_l1);
    const uint32_t bw = d.bw;
    const uint32_t bh = d.bh;
    uint32_t j = j0;
    while (j < j1) {
        const uint32_t br = j / bw;  // box-row index = jt*bh + jh
        const uint32_t jw0 = j - br * bw;
        uint32_t jw1 = bw;
        if (br * bw + jw1 > j1) {
            jw1 = j1 - br * bw;  // clamp the w-run to this packed chunk's end
        }
        const uint32_t jt = br / bh;
        const uint32_t jh = br - jt * bh;
        const uint32_t real_row = (box.t0 + jt) * H + (box.h_lo + jh);
        noc.async_read(
            reader, CoreLocalMem<uint32_t>(staging_l1), row_bytes, {.page_id = head_row_base + real_row}, {});
        noc.async_read_barrier();
        for (uint32_t jw = jw0; jw < jw1; ++jw) {
            const uint32_t local = br * bw + jw - j0;
            scatter_row<transposed_grid>(
                stg + (box.w_lo + jw) * D, cb_base, local, D_tiles, seqtiles_packed, tile_bytes);
        }
        j = br * bw + jw1;
    }
}

}  // namespace neighborhood_gather
