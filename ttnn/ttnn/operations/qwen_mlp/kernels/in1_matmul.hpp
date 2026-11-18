// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
#include "matmul_chain.hpp"

template <typename Accessor>
void read_in1_block(
    const Noc& noc,
    const MatmulShape& s,
    const Accessor& in1,
    uint32_t tile_bytes,
    uint32_t dst,
    uint32_t k_tile,
    uint32_t n_tile) {
    for (uint32_t i = 0; i < s.K_block; i++) {
        for (uint32_t j = 0; j < s.N_block && n_tile + j < s.N_logical; j++) {
            noc.async_read(
                in1,
                CoreLocalMem<uint32_t>(dst + (i * s.N_block + j) * tile_bytes),
                tile_bytes,
                {.page_id = (k_tile + i) * s.in1_row_tiles + n_tile + j},
                {});
        }
    }
    noc.async_read_barrier();
}

template <typename Accessor>
void in1_matmul_phase(
    const Noc& noc,
    const MatmulShape& s,
    const MatmulRange& range,
    Chain& chain,
    const Accessor& in1,
    uint32_t tile_bytes) {
    CircularBuffer cb_in1(s.cb_in1);
    const uint32_t block_tiles = s.K_block * s.N_block;

    for (uint32_t m_block = 0; m_block < s.M_blocks; m_block++) {
        bool k_forward = true;
        for (uint32_t n_block = 0; n_block < s.N_blocks; n_block++) {
            const uint32_t n_tile = range.N_start + n_block * s.N_block;
            for (uint32_t k_iter = 0; k_iter < s.K_blocks; k_iter++) {
                const uint32_t k_block = k_forward ? k_iter : s.K_blocks - 1 - k_iter;
                cb_in1.reserve_back(block_tiles);
                const uint32_t addr = cb_in1.get_write_ptr();
                if (chain.is_injector) {
                    read_in1_block(noc, s, in1, tile_bytes, addr, k_block * s.K_block, n_tile);
                } else {
                    chain.receive(noc);
                }
                cb_in1.push_back(block_tiles);
                if (!chain.is_sink) {
                    chain.forward(noc, addr, block_tiles * tile_bytes);
                }
            }
            k_forward = !k_forward;
        }
    }
}
