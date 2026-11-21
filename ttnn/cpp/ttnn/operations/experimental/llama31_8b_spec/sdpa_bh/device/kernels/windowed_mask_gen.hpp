// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <type_traits>

#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tile_constants.h"
#include "dataflow_common.hpp"
#include "cpp/ttnn/operations/transformer/sdpa/device/kernels/windowed_loop_geometry.hpp"

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

template <uint32_t tile_bytes>
inline void fill_upper_triangle_neginf(uint32_t cb_id, uint32_t tile_id) {
    constexpr uint32_t FH = tt::constants::FACE_HEIGHT;
    constexpr uint32_t FW = tt::constants::FACE_WIDTH;
    constexpr uint16_t NEGINF = 0xFF80;
    CircularBuffer cb(cb_id);
    uint32_t write_addr = cb.get_write_ptr() + tile_id * tile_bytes;
    volatile tt_l1_ptr uint16_t* p = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(write_addr);
    for (uint32_t r = 0; r < tt::constants::TILE_HEIGHT; ++r) {
        const uint32_t face_row = (r >= FH) ? 2u : 0u;
        const uint32_t fr = r & (FH - 1);
        for (uint32_t c = r + 1; c < tt::constants::TILE_WIDTH; ++c) {
            const uint32_t face = face_row + ((c >= FW) ? 1u : 0u);
            const uint32_t fc = c & (FW - 1);
            p[face * (FH * FW) + fr * FW + fc] = NEGINF;
        }
    }
}

template <WindowedMode mode, uint32_t mask_tile_bytes, uint32_t cb_mask_in, uint32_t cb_cu_window_in>
inline void generate_windowed_mask_for_q_chunk(
    Noc& noc,
    uint32_t q_chunk,
    uint32_t Sq_chunk_t,
    uint32_t Sk_chunk_t,
    uint32_t valid_Sqt,
    uint32_t valid_Skt,
    uint32_t k_num_chunks,
    uint32_t cu_window_seqlens_eles,
    uint32_t q_tok_offset) {
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
    const uint32_t q_low_tok = q_tok_offset + q_row_start_tile * tt::constants::TILE_HEIGHT;
    const uint32_t q_high_tok = q_tok_offset + q_row_end_tile * tt::constants::TILE_HEIGHT;

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

    const auto k_range = windowed_k_chunk_range<mode>(
        q_chunk,
        Sq_chunk_t,
        valid_Sqt,
        q_tok_offset,
        cu_ptr,
        cu_window_seqlens_eles,
        Sk_chunk_t,
        k_num_chunks,
        tt::constants::TILE_HEIGHT);
    const uint32_t mask_chunk_tiles = Sq_chunk_t * Sk_chunk_t;
    CircularBuffer cb_mask(cb_mask_in);
    uint32_t local_window_idx = start_window_idx;
    for (uint32_t k_chunk = k_range.k_lo; k_chunk < k_range.k_hi; ++k_chunk) {
        const uint32_t k_row_start_tile = std::min(k_chunk * Sk_chunk_t, valid_Skt);

        cb_mask.reserve_back(mask_chunk_tiles);
        uint32_t mask_write_ptr_base = cb_mask.get_write_ptr();

        int zero_tile_idx = -1;
        int inf_tile_idx = -1;
        for (uint32_t row = 0; row < Sq_chunk_t; ++row) {
            uint32_t q_start_idx = q_tok_offset + (q_row_start_tile + row) * tt::constants::TILE_HEIGHT;
            uint32_t q_end_idx = q_start_idx + tt::constants::TILE_HEIGHT;

            auto result = get_window_indices(local_window_idx);
            uint32_t window_low_idx = result.first;
            uint32_t window_high_idx = result.second;

            for (uint32_t col = 0; col < Sk_chunk_t; ++col) {
                uint32_t k_start_idx = (k_row_start_tile + col) * tt::constants::TILE_HEIGHT;
                uint32_t k_end_idx = k_start_idx + tt::constants::TILE_HEIGHT;
                uint32_t in_mask_tile_id = row * Sk_chunk_t + col;

                const bool on_diagonal = mode == WindowedMode::Causal && k_start_idx == q_start_idx;
                if (mode == WindowedMode::Causal && k_start_idx > q_start_idx) {
                    if (inf_tile_idx == -1) {
                        fill_neginf_tile<mask_tile_bytes>(cb_mask_in, in_mask_tile_id);
                        inf_tile_idx = in_mask_tile_id;
                    } else {
                        copy_tile<mask_tile_bytes>(
                            noc, mask_write_ptr_base, mask_write_ptr_base, inf_tile_idx, in_mask_tile_id);
                    }
                    continue;
                }

                if (q_start_idx >= window_low_idx && q_end_idx <= window_high_idx && k_start_idx >= window_low_idx &&
                    k_end_idx <= window_high_idx) {
                    if (zero_tile_idx == -1) {
                        fill_tile_zeros<mask_tile_bytes, false>(noc, cb_mask_in, in_mask_tile_id);
                    } else {
                        copy_tile<mask_tile_bytes>(
                            noc, mask_write_ptr_base, mask_write_ptr_base, zero_tile_idx, in_mask_tile_id);
                    }
                    if (!on_diagonal) {
                        zero_tile_idx = in_mask_tile_id;
                        continue;
                    }
                    noc.async_read_barrier();
                    fill_upper_triangle_neginf<mask_tile_bytes>(cb_mask_in, in_mask_tile_id);
                    continue;
                }

                const bool inf_copied = inf_tile_idx != -1;
                if (inf_copied) {
                    copy_tile<mask_tile_bytes>(
                        noc, mask_write_ptr_base, mask_write_ptr_base, inf_tile_idx, in_mask_tile_id);
                } else {
                    fill_neginf_tile<mask_tile_bytes>(cb_mask_in, in_mask_tile_id);
                }
                if (!found_mask_windows || k_end_idx <= window_low_idx || k_start_idx >= window_high_idx ||
                    window_low_idx >= window_high_idx) {
                    inf_tile_idx = in_mask_tile_id;
                    continue;
                }
                if (inf_copied) {
                    noc.async_read_barrier();
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
                        std::pair<uint32_t, uint32_t> nxt;
                        do {
                            local_window_idx += 1;
                            nxt = get_window_indices(local_window_idx);
                        } while (nxt.first == nxt.second && local_window_idx + 1 < cu_window_seqlens_eles);
                        window_low_idx = nxt.first;
                        window_high_idx = nxt.second;
                    }
                } while (window_low_idx < window_high_idx && cqe < q_end_idx && cke < k_end_idx);

                if (on_diagonal) {
                    fill_upper_triangle_neginf<mask_tile_bytes>(cb_mask_in, in_mask_tile_id);
                }
            }
        }
        noc.async_read_barrier();
        cb_mask.push_back(mask_chunk_tiles);
    }
}

template <
    WindowedMode mode,
    uint32_t cb_mask_in,
    uint32_t cb_cu_window_in,
    std::enable_if_t<!is_windowed_mode(mode), int> = 0,
    typename... Args>
inline void windowed_generate_if_enabled(Args&&...) {}

template <
    WindowedMode mode,
    uint32_t cb_mask_in,
    uint32_t cb_cu_window_in,
    std::enable_if_t<is_windowed_mode(mode), int> = 0>
inline void windowed_generate_if_enabled(
    Noc& noc,
    uint32_t q_chunk,
    uint32_t Sq_chunk_t,
    uint32_t Sk_chunk_t,
    uint32_t valid_Sqt,
    uint32_t valid_Skt,
    uint32_t k_num_chunks,
    uint32_t cu_window_seqlens_eles,
    uint32_t q_tok_offset) {
    constexpr uint32_t mask_tile_bytes = get_tile_size(cb_mask_in);
    generate_windowed_mask_for_q_chunk<mode, mask_tile_bytes, cb_mask_in, cb_cu_window_in>(
        noc, q_chunk, Sq_chunk_t, Sk_chunk_t, valid_Sqt, valid_Skt, k_num_chunks, cu_window_seqlens_eles, q_tok_offset);
}
