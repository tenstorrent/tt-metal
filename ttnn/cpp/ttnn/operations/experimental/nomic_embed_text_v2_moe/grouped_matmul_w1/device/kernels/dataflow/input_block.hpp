// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// One input block's reads, K tiles of one tile row, as the lane's east core and the dedicated senders issue them.

#pragma once

#include <cstdint>

#include "api/core_local_mem.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"

namespace grouped_matmul {

// input_banks > 0: an L1-interleaved input, its bank and address stepped here rather than by the accessor.
// input_run > 0: a sharded input holding input_run tiles of a row contiguously, one read a run. trid tags the reads
// when opts has NocOptions::TXN_ID.
template <
    NocOptions opts,
    uint32_t k_tiles,
    uint32_t tile_bytes,
    uint32_t input_banks,
    uint32_t input_run,
    typename Accessor>
FORCE_INLINE void read_input_block(
    const Noc& noc, const Accessor& input, uint32_t input_address, uint32_t block, uint32_t address, uint32_t trid) {
    uint32_t page = block * k_tiles;
    if constexpr (input_banks > 0) {
        uint32_t bank = page % input_banks;
        uint32_t bank_address = input_address + (page / input_banks) * tile_bytes;
        for (uint32_t tile = 0; tile < k_tiles; ++tile, address += tile_bytes) {
            noc.async_read<opts, tile_bytes>(
                AllocatorBank<AllocatorBankType::L1>{},
                CoreLocalMem<uint32_t>(address),
                tile_bytes,
                {.bank_id = bank, .addr = bank_address},
                {},
                {.trid = trid});
            if (++bank == input_banks) {
                bank = 0;
                bank_address += tile_bytes;
            }
        }
    } else if constexpr (input_run > 0) {
        for (uint32_t tile = 0; tile < k_tiles; tile += input_run, page += input_run) {
            noc.async_read<opts>(
                input, CoreLocalMem<uint32_t>(address), input_run * tile_bytes, {.page_id = page}, {}, {.trid = trid});
            address += input_run * tile_bytes;
        }
    } else {
        for (uint32_t tile = 0; tile < k_tiles; ++tile, ++page, address += tile_bytes) {
            noc.async_read<opts>(
                input, CoreLocalMem<uint32_t>(address), tile_bytes, {.page_id = page}, {}, {.trid = trid});
        }
    }
}

}  // namespace grouped_matmul
