// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/tensor/noc_traits.h"

struct PairPartner {
    uint32_t x;
    uint32_t y;
};

// Loads the tile columns [first_column, first_column + columns) of all k_tiles tile rows, k-major, resident in the CB,
// in chunks of chunk_rows tile rows. The chunks are split between the `groups` cores that hold the same weights: each
// core reads its own chunks (core `group` reads the chunks c with c % groups == group) and writes them to the other
// core's L1 too. Chunks are pushed to the CB in K order as soon as a contiguous prefix has landed.
template <typename Accessor>
struct SharedWeights {
    const Noc& noc;
    const Accessor& weights;
    CircularBuffer cb;
    Semaphore<> partner_landed;
    PairPartner partner;
    uint32_t tile_bytes;
    uint32_t row_tiles;
    uint32_t first_row_tile;
    uint32_t columns;
    uint32_t chunk_rows;
    uint32_t chunks;
    uint32_t group;
    uint32_t groups;
    uint32_t reads_in_flight;
    uint32_t base;
    uint32_t issued = 0;
    uint32_t read = 0;
    uint32_t pushed = 0;

    uint32_t chunk_tiles() const { return chunk_rows * columns; }
    uint32_t own_chunk(uint32_t index) const { return index * groups + group; }
    uint32_t own_chunks() const { return (chunks - group + groups - 1) / groups; }
    bool done() const { return pushed == chunks; }

    void start() {
        cb.reserve_back(chunks * chunk_tiles());
        base = cb.get_write_ptr();
    }

    void issue_chunk() {
        const uint32_t chunk = own_chunk(issued);
        const uint32_t trid = 1 + issued % reads_in_flight;
        for (uint32_t row = 0; row < chunk_rows; row++) {
            for (uint32_t column = 0; column < columns; column++) {
                noc.async_read<NocOptions::TXN_ID>(
                    weights,
                    CoreLocalMem<uint32_t>(base + ((chunk * chunk_rows + row) * columns + column) * tile_bytes),
                    tile_bytes,
                    {.page_id = (chunk * chunk_rows + row) * row_tiles + first_row_tile + column},
                    {},
                    {.trid = trid});
            }
        }
        issued++;
    }

    void share_chunk() {
        if (groups > 1) {
            const uint32_t offset = own_chunk(read) * chunk_tiles() * tile_bytes;
            noc.async_write(
                CoreLocalMem<uint32_t>(base + offset),
                UnicastEndpoint{},
                chunk_tiles() * tile_bytes,
                {},
                {.noc_x = partner.x, .noc_y = partner.y, .addr = base + offset});
            noc.async_writes_flushed();
            partner_landed.up(noc, partner.x, partner.y, 1);
        }
        read++;
    }

    uint32_t landed_prefix() const {
        if (groups == 1) {
            return read;
        }
        const uint32_t partner_chunks = partner_landed.value();
        const uint32_t even = group == 0 ? read : partner_chunks;
        const uint32_t odd = group == 0 ? partner_chunks : read;
        const uint32_t prefix = even * 2 < odd * 2 + 1 ? even * 2 : odd * 2 + 1;
        return prefix < chunks ? prefix : chunks;
    }

    void step() {
        if (issued < own_chunks() && issued - read < reads_in_flight) {
            issue_chunk();
        }
        if (read < issued && noc.is_read_trid_flushed(1 + read % reads_in_flight)) {
            share_chunk();
        }
        const uint32_t prefix = landed_prefix();
        if (prefix > pushed) {
            cb.push_back((prefix - pushed) * chunk_tiles());
            pushed = prefix;
        }
    }

    void finish() {
        while (!done()) {
            step();
        }
        noc.async_atomic_barrier();
    }
};
