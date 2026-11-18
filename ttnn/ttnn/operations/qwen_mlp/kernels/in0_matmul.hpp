// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
#include "matmul_chain.hpp"

template <typename Accessor>
void read_in0_block(
    const Noc& noc,
    const MatmulShape& s,
    const Accessor& in0,
    uint32_t tile_bytes,
    uint32_t dst,
    uint32_t m_tile,
    uint32_t k_tile) {
    for (uint32_t i = 0; i < s.M_block && m_tile + i < s.M_logical; i++) {
        for (uint32_t j = 0; j < s.K_block; j++) {
            noc.async_read(
                in0,
                CoreLocalMem<uint32_t>(dst + (i * s.K_block + j) * tile_bytes),
                tile_bytes,
                {.page_id = (m_tile + i) * s.in0_row_tiles + k_tile + j},
                {});
        }
    }
    noc.async_read_barrier();
}

template <typename Accessor>
void write_out_block(
    const Noc& noc,
    const MatmulShape& s,
    const Accessor& out,
    uint32_t tile_bytes,
    CircularBuffer& cb_out,
    uint32_t m_tile,
    uint32_t n_tile) {
    const uint32_t out_cols = s.swiglu ? s.N_block / 2 : s.N_block;
    const uint32_t out_col = s.swiglu ? n_tile / 2 : n_tile;
    for (uint32_t i = 0; i < s.M_block; i++) {
        cb_out.wait_front(out_cols);
        if (m_tile + i < s.M_logical) {
            uint32_t src = cb_out.get_read_ptr();
            for (uint32_t j = 0; j < out_cols && out_col + j < s.out_N_logical; j++) {
                noc.async_write(
                    CoreLocalMem<uint32_t>(src + j * tile_bytes),
                    out,
                    tile_bytes,
                    {},
                    {.page_id = (m_tile + i) * s.out_row_tiles + out_col + j});
            }
        }
        noc.async_writes_flushed();
        cb_out.pop_front(out_cols);
    }
}

template <typename InAccessor, typename OutAccessor>
void in0_matmul_phase(
    const Noc& noc,
    const MatmulShape& s,
    const MatmulRange& range,
    Chain& chain,
    const InAccessor& in0,
    const OutAccessor& out,
    uint32_t tile_bytes,
    uint32_t defer_k_block) {
    CircularBuffer cb_in0(s.cb_in0);
    CircularBuffer cb_out(s.cb_out);
    const uint32_t block_tiles = s.M_block * s.K_block;

    bool write_pending = false;
    uint32_t pending_m = 0;
    uint32_t pending_n = 0;
    for (uint32_t m_block = 0; m_block < s.M_blocks; m_block++) {
        const uint32_t m_tile = range.M_start + m_block * s.M_block;
        bool k_forward = true;
        bool reuse = false;
        for (uint32_t n_block = 0; n_block < s.N_blocks; n_block++) {
            const uint32_t n_tile = range.N_start + n_block * s.N_block;
            const bool last = m_block == s.M_blocks - 1 && n_block == s.N_blocks - 1;
            for (uint32_t k_iter = 0; k_iter < s.K_blocks; k_iter++) {
                if (write_pending && k_iter == defer_k_block) {
                    write_out_block(noc, s, out, tile_bytes, cb_out, pending_m, pending_n);
                    write_pending = false;
                }
                if (reuse && k_iter == 0) {
                    reuse = false;
                    continue;
                }
                const uint32_t k_block = k_forward ? k_iter : s.K_blocks - 1 - k_iter;
                cb_in0.reserve_back(block_tiles);
                const uint32_t addr = cb_in0.get_write_ptr();
                if (chain.is_injector) {
                    read_in0_block(noc, s, in0, tile_bytes, addr, m_tile, k_block * s.K_block);
                } else {
                    chain.receive(noc);
                }
                cb_in0.push_back(block_tiles);
                if (!chain.is_sink) {
                    chain.forward(noc, addr, block_tiles * tile_bytes);
                }
            }
            k_forward = !k_forward;
            reuse = true;
            if (last || chain.is_injector) {
                write_out_block(noc, s, out, tile_bytes, cb_out, m_tile, n_tile);
            } else {
                write_pending = true;
                pending_m = m_tile;
                pending_n = n_tile;
            }
        }
    }
    noc.async_write_barrier();
}
