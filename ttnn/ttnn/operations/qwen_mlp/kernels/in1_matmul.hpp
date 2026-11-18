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

template <typename OutAccessor>
void in1_matmul_phase(
    const Noc& noc,
    const MatmulShape& s,
    const MatmulRange& range,
    Chain& chain,
    uint32_t in1_address,
    const OutAccessor& out,
    uint32_t tile_bytes,
    uint32_t out_tile_bytes,
    uint32_t defer_k_block) {
    CircularBuffer cb_in1(s.cb_in1);
    CircularBuffer cb_out(s.cb_out);
    const uint32_t block_tiles = s.K_block * s.N_block;

    bool write_pending = false;
    bool write_in_flight = false;
    uint32_t pending_m = 0;
    uint32_t pending_n = 0;
    for (uint32_t m_block = 0; m_block < s.M_blocks; m_block++) {
        const uint32_t m_tile = range.M_start + m_block * s.M_block;
        bool k_forward = true;
        for (uint32_t n_block = 0; n_block < s.N_blocks; n_block++) {
            const uint32_t n_tile = range.N_start + n_block * s.N_block;
            for (uint32_t k_iter = 0; k_iter < s.K_blocks; k_iter++) {
                if (write_in_flight) {
                    retire_out_block(noc, s, cb_out);
                    write_in_flight = false;
                }
                if (write_pending && k_iter == defer_k_block) {
                    issue_out_block_writes(noc, s, out, out_tile_bytes, cb_out, pending_m, pending_n);
                    write_pending = false;
                    write_in_flight = true;
                }
                const uint32_t k_block = k_forward ? k_iter : s.K_blocks - 1 - k_iter;
                cb_in1.reserve_back(block_tiles);
                const uint32_t addr = cb_in1.get_write_ptr();
                if (chain.is_injector) {
                    read_in1_block(s, in1_address, tile_bytes, addr, k_block * s.K_block, n_tile);
                } else {
                    chain.receive(noc);
                }
                cb_in1.push_back(block_tiles);
                if (!chain.is_sink) {
                    chain.forward(noc, addr, block_tiles * tile_bytes);
                }
            }
            k_forward = !k_forward;
            write_pending = true;
            pending_m = m_tile;
            pending_n = n_tile;
        }
    }
    if (write_in_flight) {
        retire_out_block(noc, s, cb_out);
    }
    issue_out_block_writes(noc, s, out, out_tile_bytes, cb_out, pending_m, pending_n);
    retire_out_block(noc, s, cb_out);
    noc.async_write_barrier();
}
