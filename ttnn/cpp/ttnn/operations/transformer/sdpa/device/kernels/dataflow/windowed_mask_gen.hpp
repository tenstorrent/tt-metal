// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Block-diagonal (windowed) attention mask generation helpers.
//
// These build the Float16_b attention mask on-the-fly from cu_window_seqlens. They are invoked by the
// SDPA writer kernel (writer_interleaved) so that the reader is left free to stream Q/K/V only. The mask
// content depends solely on tile indices and the window boundaries -- never on the Q/K/V data -- so it
// can be produced independently of the reader.

#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc.h"
#include <tt-metalium/constants.hpp>
#include "dataflow_common.hpp"

// Zero a [row,col) sub-rectangle of a Float16_b tile that is otherwise -inf (the partial-window boundary
// tile). A Float16_b tile is 4 row-major 16x16 faces of 2-byte elements, so this is a direct per-element
// write of 0x0000. The windowed mask is always Float16_b (the streaming path does not decode block-float
// masks, and the standard path decodes Float16_b fine).
template <uint32_t tile_bytes>
inline void fill_diag_subtile_zeros(
    uint32_t cb_id,
    uint32_t tile_id,
    uint32_t row_start_idx,
    uint32_t row_end_idx,
    uint32_t col_start_idx,
    uint32_t col_end_idx) {
    constexpr uint32_t FH = tt::constants::FACE_HEIGHT;
    constexpr uint32_t FW = tt::constants::FACE_WIDTH;
    CircularBuffer cb(cb_id);
    uint32_t write_addr = cb.get_write_ptr() + tile_id * tile_bytes;
    volatile tt_l1_ptr uint16_t* p = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(write_addr);
    for (uint32_t r = row_start_idx; r < row_end_idx; ++r) {
        const uint32_t face_row = (r >= FH) ? 2u : 0u;
        const uint32_t fr = r & (FH - 1);
        for (uint32_t c = col_start_idx; c < col_end_idx; ++c) {
            const uint32_t face = face_row + ((c >= FW) ? 1u : 0u);
            const uint32_t fc = c & (FW - 1);
            p[face * (FH * FW) + fr * FW + fc] = 0x0000;
        }
    }
}

// Generate and push the full block-diagonal mask (Sq_chunk_t x Sk_chunk_t tiles per K chunk, for all K
// chunks) for a single Q chunk. Self-contained: the start window is searched from cu_window_seqlens for
// this Q chunk's row range, so the result is independent of the order in which Q chunks are scheduled
// (safe under the regular SDPA factory's global-Q scheduling, incl. zigzag). cb_cu_window_in must already
// hold the cu_window_seqlens tensor (loaded + pushed once by the caller).
template <uint32_t mask_tile_bytes, uint32_t cb_mask_in, uint32_t cb_cu_window_in>
inline void generate_windowed_mask_for_q_chunk(
    Noc& noc,
    uint32_t q_chunk,
    uint32_t Sq_chunk_t,
    uint32_t Sk_chunk_t,
    uint32_t valid_Sqt,
    uint32_t valid_Skt,
    uint32_t k_num_chunks,
    uint32_t cu_window_seqlens_eles) {
    // cu_window_seqlens is INT32/UINT32 (validated host-side); both store non-negative cumulative
    // lengths in 32-bit words, so a plain uint32 read is correct for either.
    CircularBuffer cb_cu(cb_cu_window_in);
    volatile tt_l1_ptr uint32_t* cu_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb_cu.get_read_ptr());
    auto get_cu = [&](uint32_t idx) -> uint32_t { return cu_ptr[idx]; };
    auto get_window_indices = [&](uint32_t i) {
        if (i < cu_window_seqlens_eles) {
            auto low = get_cu(i);
            auto high = (i == cu_window_seqlens_eles - 1) ? low : get_cu(i + 1);
            return std::make_pair(low, high);
        }
        auto low = get_cu(cu_window_seqlens_eles - 1);
        return std::make_pair(low, low);
    };

    const uint32_t q_row_start_tile = std::min(q_chunk * Sq_chunk_t, valid_Sqt);
    const uint32_t q_row_end_tile = std::min(q_row_start_tile + Sq_chunk_t, valid_Sqt);
    const uint32_t q_low_tok = q_row_start_tile * tt::constants::TILE_HEIGHT;
    const uint32_t q_high_tok = q_row_end_tile * tt::constants::TILE_HEIGHT;

    uint32_t start_window_idx = 0;
    bool found_mask_windows = false;
    for (uint32_t w = 0; w + 1 < cu_window_seqlens_eles; ++w) {
        auto ws = get_cu(w);
        auto we = get_cu(w + 1);
        if ((q_low_tok >= ws && q_low_tok < we) || (q_high_tok > ws && q_high_tok <= we)) {
            start_window_idx = w;
            found_mask_windows = true;
            break;
        }
    }

    const uint32_t mask_chunk_tiles = Sq_chunk_t * Sk_chunk_t;
    CircularBuffer cb_mask(cb_mask_in);
    uint32_t local_window_idx = start_window_idx;
    for (uint32_t k_chunk = 0; k_chunk < k_num_chunks; ++k_chunk) {
        const uint32_t k_row_start_tile = std::min(k_chunk * Sk_chunk_t, valid_Skt);

        cb_mask.reserve_back(mask_chunk_tiles);
        uint32_t mask_write_ptr_base = cb_mask.get_write_ptr();

        int zero_tile_idx = -1;
        int inf_tile_idx = -1;
        for (uint32_t row = 0; row < Sq_chunk_t; ++row) {
            uint32_t q_start_idx = (q_row_start_tile + row) * tt::constants::TILE_HEIGHT;
            uint32_t q_end_idx = q_start_idx + tt::constants::TILE_HEIGHT;

            auto result = get_window_indices(local_window_idx);
            uint32_t window_low_idx = result.first;
            uint32_t window_high_idx = result.second;

            for (uint32_t col = 0; col < Sk_chunk_t; ++col) {
                uint32_t k_start_idx = (k_row_start_tile + col) * tt::constants::TILE_HEIGHT;
                uint32_t k_end_idx = k_start_idx + tt::constants::TILE_HEIGHT;
                uint32_t in_mask_tile_id = row * Sk_chunk_t + col;

                if (q_start_idx >= window_low_idx && q_end_idx <= window_high_idx && k_start_idx >= window_low_idx &&
                    k_end_idx <= window_high_idx) {
                    if (zero_tile_idx == -1) {
                        fill_tile_zeros<mask_tile_bytes, false>(noc, cb_mask_in, in_mask_tile_id);
                    } else {
                        copy_tile<mask_tile_bytes>(
                            noc, mask_write_ptr_base, mask_write_ptr_base, zero_tile_idx, in_mask_tile_id);
                    }
                    zero_tile_idx = in_mask_tile_id;
                    continue;
                }

                if (inf_tile_idx == -1) {
                    fill_neginf_tile<mask_tile_bytes>(cb_mask_in, in_mask_tile_id);
                } else {
                    copy_tile<mask_tile_bytes>(
                        noc, mask_write_ptr_base, mask_write_ptr_base, inf_tile_idx, in_mask_tile_id);
                }
                if (!found_mask_windows || k_end_idx <= window_low_idx || k_start_idx >= window_high_idx ||
                    window_low_idx >= window_high_idx) {
                    inf_tile_idx = in_mask_tile_id;
                    continue;
                }

                uint32_t cqs, cks, cqe, cke;
                do {
                    cqs = std::max(q_start_idx, window_low_idx);
                    cks = std::max(k_start_idx, window_low_idx);
                    cqe = std::min(q_end_idx, window_high_idx);
                    cke = std::min(k_end_idx, window_high_idx);

                    if (cqs < cqe && cks < cke) {
                        fill_diag_subtile_zeros<mask_tile_bytes>(
                            cb_mask_in,
                            in_mask_tile_id,
                            cqs - q_start_idx,
                            cqe - q_start_idx,
                            cks - k_start_idx,
                            cke - k_start_idx);
                    }

                    if (cqe >= window_high_idx && cke >= window_high_idx) {
                        local_window_idx += 1;
                        auto nxt = get_window_indices(local_window_idx);
                        window_low_idx = nxt.first;
                        window_high_idx = nxt.second;
                    }
                } while (window_low_idx < window_high_idx && cqe < q_end_idx && cke < k_end_idx);
            }
        }
        noc.async_read_barrier();
        cb_mask.push_back(mask_chunk_tiles);
    }
}

// Template wrapper so the windowed generator is instantiated ONLY when use_windowed_mask is true.
// kernel_main is not a template, so an `if constexpr` there does NOT discard its body — it would still
// compile, constexpr-evaluating get_tile_size on a possibly-inactive CB id. Inside this template,
// `if constexpr (W)` discards properly, so non-windowed writer builds never touch the generator.
template <bool W, uint32_t cb_mask_in, uint32_t cb_cu_window_in>
inline void windowed_generate_if_enabled(
    Noc& noc,
    uint32_t q_chunk,
    uint32_t Sq_chunk_t,
    uint32_t Sk_chunk_t,
    uint32_t valid_Sqt,
    uint32_t valid_Skt,
    uint32_t k_num_chunks,
    uint32_t cu_window_seqlens_eles) {
    if constexpr (W) {
        constexpr uint32_t mask_tile_bytes = get_tile_size(cb_mask_in);
        generate_windowed_mask_for_q_chunk<mask_tile_bytes, cb_mask_in, cb_cu_window_in>(
            noc, q_chunk, Sq_chunk_t, Sk_chunk_t, valid_Sqt, valid_Skt, k_num_chunks, cu_window_seqlens_eles);
    }
}
