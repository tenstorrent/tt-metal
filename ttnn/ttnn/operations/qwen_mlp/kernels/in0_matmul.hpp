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
void in0_matmul_phase(
    const Noc& noc,
    const MatmulShape& s,
    const MatmulRange& range,
    Chain& chain,
    const Accessor& in0,
    uint32_t tile_bytes) {
    CircularBuffer cb_in0(s.cb_in0);
    const uint32_t block_tiles = s.M_block * s.K_block;

    for (uint32_t m_block = 0; m_block < s.M_blocks; m_block++) {
        const uint32_t m_tile = range.M_start + m_block * s.M_block;
        bool k_forward = true;
        bool reuse = false;
        for (uint32_t n_block = 0; n_block < s.N_blocks; n_block++) {
            for (uint32_t k_iter = 0; k_iter < s.K_blocks; k_iter++) {
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
        }
    }
}
