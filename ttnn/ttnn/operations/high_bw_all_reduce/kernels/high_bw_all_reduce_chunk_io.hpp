// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// high_bw_all_reduce — chunk <-> DRAM page I/O (single source for the reader, the reducer writer's
// tail write and port_bwd's relay write).
//
// Bank-run chunk layout (Refinement 4). A chunk holds tiles [first_page, first_page + valid) of an
// interleaved DRAM tensor. Interleaved page p lives in bank p mod NB at in-bank row p / NB, so
// pages p, p+NB, p+2NB, ... of one chunk are CONTIGUOUS in one bank. With the chunk stored
// bank-major in L1 (tile t at slot (t mod NB) * run_tiles + t / NB), each bank's pages are also
// contiguous in L1, and a chunk moves in NB transfers of run_tiles pages (8 KiB for bf16 and fp32
// at 64 KiB chunks) instead of chunk_tiles page transfers. That cuts the per-chunk NoC command
// count run_tiles-fold on the RISC that issues it — which is what bound the relay port on the
// snake / ring middles (port_bwd forwards the final over Fabric AND writes it to DRAM).
//
// The layout is invisible to everything else: the reduction is elementwise over the chunk (both
// operands use the same layout, so the sum does too), and partials / finals move between devices
// as raw whole-chunk bytes. Padding slots of a ragged chunk are never read from or written to DRAM.
//
// kEnable = false (or chunk_tiles not a multiple of NB) gives num_runs = chunk_tiles and
// run_tiles = 1: the plain page-by-page identity layout (the pre-Refinement-4 transfers).

#pragma once

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

template <uint32_t chunk_tiles, uint32_t tile_bytes, bool kEnable>
struct ChunkIo {
    static constexpr bool bank_runs = kEnable && (chunk_tiles % NUM_DRAM_BANKS == 0);
    static constexpr uint32_t num_runs = bank_runs ? NUM_DRAM_BANKS : chunk_tiles;
    static constexpr uint32_t run_tiles = chunk_tiles / num_runs;
    static constexpr uint32_t run_bytes = run_tiles * tile_bytes;

    // Pages of run j in a chunk of `valid` valid tiles (runs are filled in order, so the first
    // empty run ends the chunk).
    static uint32_t run_len(uint32_t j, uint32_t valid) {
        return valid > j ? (valid - j + num_runs - 1) / num_runs : 0;
    }

    template <typename Accessor>
    static void read(const Accessor& acc, uint32_t first_page, uint32_t valid, uint32_t l1_addr) {
        for (uint32_t j = 0; j < num_runs; ++j) {
            const uint32_t n = run_len(j, valid);
            if (n == 0) {
                break;
            }
            noc_async_read(acc.get_noc_addr(first_page + j), l1_addr + j * run_bytes, n * tile_bytes);
        }
    }

    template <typename Accessor>
    static void write(const Accessor& acc, uint32_t first_page, uint32_t valid, uint32_t l1_addr) {
        for (uint32_t j = 0; j < num_runs; ++j) {
            const uint32_t n = run_len(j, valid);
            if (n == 0) {
                break;
            }
            noc_async_write(l1_addr + j * run_bytes, acc.get_noc_addr(first_page + j), n * tile_bytes);
        }
    }
};
