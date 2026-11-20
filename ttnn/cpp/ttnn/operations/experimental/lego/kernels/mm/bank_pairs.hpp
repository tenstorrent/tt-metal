// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Bank-paired SwiGLU weight reads (in1_bank_pairs = B, the DRAM bank count; 0 = off).
//
// An interleaved DRAM tensor puts page p in bank p % B, and pages p and p + B next to each other in that bank. For a
// weight of N tiles per row with N % (2B) == 0, columns n and n + B of one row are such a pair, and a 2-page read of
// them costs about as much as one 1-page read when the pages are small (bfp4 tiles, 576 B: ~375 GB/s instead of ~275
// GB/s on a p300c). The gate/up interleaved SwiGLU weight (column 2p = gate p, 2p + 1 = up p) is therefore read in
// quads: a core's logical columns [4Q, 4Q + 4) hold, in this order,
//   gate g, gate g + B/2, up g, up g + B/2      with g = B*s + q, s = Q / (B/2), q = Q % (B/2)
// (weight columns 2g, 2g + B, 2g + 1, 2g + 1 + B: two 2-page reads). Compute pairs DST tiles (w, w + 2) and (w + 1,
// w + 3) of each quad and packs the two products in that order, so the output position 2Q + t (t = 0, 1) holds
// output column bp_out_col(2Q + t) = g + t * B/2. The output writers write through that map.

#pragma once

#include <stdint.h>
#include "ttnn/operations/experimental/lego/kernels/lego.hpp"
#if defined(COMPILE_FOR_BRISC) || defined(COMPILE_FOR_NCRISC)
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "ttnn/operations/experimental/minimal_matmul/device/kernels/matmul_dataflow_common.hpp"
#endif

namespace lego::mm {

// output position o of a bank-paired block -> output column
template <uint32_t B>
inline uint32_t bp_out_col(uint32_t o) {
    if constexpr (B == 0) {
        return o;
    } else {
        const uint32_t Q = o >> 1, t = o & 1;
        return B * (Q / (B / 2)) + (Q % (B / 2)) + t * (B / 2);
    }
}

#if defined(COMPILE_FOR_BRISC) || defined(COMPILE_FOR_NCRISC)
// The K rows [d0_start, d0_end) of logical columns [d1_start, d1_end) (multiples of 4), quads as above, into the CB
// block at write_ptr (rows N_block_tiles wide). No barrier.
template <uint32_t B, uint32_t N_block_tiles, uint32_t tile_size, typename Acc>
inline void issue_in1_block_bp(
    Noc& noc,
    const Acc& acc,
    uint32_t N_tiles,
    uint32_t write_ptr,
    uint32_t d0_start,
    uint32_t d0_end,
    uint32_t d1_start,
    uint32_t d1_end) {
    for (uint32_t i = d0_start; i < d0_end; i++) {
        uint32_t wp = write_ptr;
        for (uint32_t l = d1_start; l < d1_end && l < N_tiles; l += 4) {
            const uint32_t Q = l >> 2;
            const uint32_t c = 2 * (B * (Q / (B / 2)) + (Q % (B / 2)));  // the gate column 2g
            const uint32_t row = i * N_tiles;
            noc.async_read(acc, CoreLocalMem<uint32_t>(wp), 2 * tile_size, {.page_id = row + c}, {});
            noc.async_read(
                acc, CoreLocalMem<uint32_t>(wp + 2 * tile_size), 2 * tile_size, {.page_id = row + c + 1}, {});
            wp += 4 * tile_size;
        }
        write_ptr += N_block_tiles * tile_size;
    }
}

// write_block_sync / write_block_sync_granular through bp_out_col
template <uint32_t B, uint32_t N_block_tiles, typename Acc>
inline void write_block_bp(
    const Acc& acc,
    const TensorShape2D& shape,
    uint32_t read_ptr,
    uint32_t tile_size,
    uint32_t d0_start,
    uint32_t d0_end,
    uint32_t d1_start,
    uint32_t d1_end) {
    Noc noc;
    for (uint32_t i = d0_start; i < d0_end && i < shape.logical_d0; i++) {
        for (uint32_t j = d1_start; j < d1_end; j++) {
            const uint32_t col = bp_out_col<B>(j);
            if (col < shape.logical_d1) {
                noc.async_write(
                    CoreLocalMem<uint32_t>(read_ptr), acc, tile_size, {}, {.page_id = i * shape.logical_d1 + col});
            }
            read_ptr += tile_size;
        }
        read_ptr += (N_block_tiles - (d1_end - d1_start)) * tile_size;
    }
    noc.async_writes_flushed();
}

template <uint32_t B, uint32_t M_block_tiles, uint32_t N_block_tiles, typename Acc>
inline void write_block_granular_bp(
    const Acc& acc,
    const TensorShape2D& shape,
    uint32_t cb_id,
    uint32_t tile_size,
    uint32_t d0_start,
    uint32_t d0_end,
    uint32_t d1_start,
    uint32_t d1_end) {
    Noc noc;
    CircularBuffer cb(cb_id);
    for (uint32_t m = 0; m < M_block_tiles; m++) {
        cb.wait_front(N_block_tiles);
        const uint32_t i = d0_start + m;
        if (i < d0_end && i < shape.logical_d0) {
            uint32_t rp = cb.get_read_ptr();
            for (uint32_t j = d1_start; j < d1_end; j++) {
                const uint32_t col = bp_out_col<B>(j);
                if (col < shape.logical_d1) {
                    noc.async_write(
                        CoreLocalMem<uint32_t>(rp), acc, tile_size, {}, {.page_id = i * shape.logical_d1 + col});
                }
                rp += tile_size;
            }
        }
        noc.async_writes_flushed();
        cb.pop_front(N_block_tiles);
    }
}
#endif

}  // namespace lego::mm
