// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
#include "matmul_chain.hpp"

// The weight tiles of a bank are issued back to back from one programmed read state. Tile pages are interleaved over
// the DRAM banks, so a page's bank is its id modulo the bank count and the row stride must be a multiple of it.
inline void read_in1_block(
    const MatmulShape& s, uint32_t base_address, uint32_t tile_bytes, uint32_t dst, uint32_t k_tile, uint32_t n_tile) {
    constexpr uint32_t banks = NUM_DRAM_BANKS;
    const uint32_t first_page = n_tile;
    for (uint32_t bank = 0; bank < banks; bank++) {
        const uint32_t first_j = (bank + banks - first_page % banks) % banks;
        if (first_j >= s.N_block) {
            continue;
        }
        noc_async_read_one_packet_set_state(get_noc_addr_from_bank_id<true>(bank, base_address), tile_bytes);
        for (uint32_t i = 0; i < s.K_block; i++) {
            const uint32_t row_page = (k_tile + i) * s.in1_row_tiles;
            for (uint32_t j = first_j; j < s.N_block && n_tile + j < s.N_logical; j += banks) {
                noc_async_read_one_packet_with_state(
                    base_address + ((row_page + n_tile + j) / banks) * tile_bytes,
                    dst + (i * s.N_block + j) * tile_bytes);
            }
        }
    }
    noc_async_read_barrier();
}

template <typename Accessor>
void issue_out_block_writes(
    const Noc& noc,
    const MatmulShape& s,
    const Accessor& out,
    uint32_t tile_bytes,
    CircularBuffer& cb_out,
    uint32_t m_tile,
    uint32_t n_tile) {
    const uint32_t out_cols = s.swiglu ? s.N_block / 2 : s.N_block;
    const uint32_t out_col = s.swiglu ? n_tile / 2 : n_tile;
    cb_out.wait_front(s.M_block * out_cols);
    const uint32_t src = cb_out.get_read_ptr();
    for (uint32_t i = 0; i < s.M_block && m_tile + i < s.M_logical; i++) {
        for (uint32_t j = 0; j < out_cols && out_col + j < s.out_N_logical; j++) {
            noc.async_write(
                CoreLocalMem<uint32_t>(src + (i * out_cols + j) * tile_bytes),
                out,
                tile_bytes,
                {},
                {.page_id = (m_tile + i) * s.out_row_tiles + out_col + j});
        }
    }
}

inline void retire_out_block(const Noc& noc, const MatmulShape& s, CircularBuffer& cb_out) {
    noc.async_writes_flushed();
    cb_out.pop_front(s.M_block * (s.swiglu ? s.N_block / 2 : s.N_block));
}

inline void stream_in1_block(
    const Noc& noc,
    const MatmulShape& s,
    Chain& chain,
    uint32_t in1_address,
    uint32_t tile_bytes,
    uint32_t k_block,
    uint32_t n_tile) {
    CircularBuffer cb_in1(s.cb_in1);
    cb_in1.reserve_back(s.in1_slot_tiles);
    const uint32_t addr = cb_in1.get_write_ptr();
    if (chain.is_injector) {
        read_in1_block(s, in1_address, tile_bytes, addr, k_block * s.K_block, n_tile);
    } else {
        chain.receive(noc);
    }
    cb_in1.push_back(s.in1_slot_tiles);
    if (!chain.is_sink) {
        chain.forward(noc, addr, s.K_block * s.N_block * tile_bytes);
    }
}
