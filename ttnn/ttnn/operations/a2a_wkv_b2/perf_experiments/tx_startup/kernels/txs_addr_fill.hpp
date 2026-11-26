// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// tx_startup experiment (Perf 1): fill one stream head's rows of the fabric_tx destination address table.
//
// Table layout (unchanged from the op, Refinement 3): one entry of tiles_per_packet 8 B NoC addresses per
// packet, packets in (mt block, head, packet) order: entry (mb, s, pp) at
// mb * num_heads * full_block_packets + s * packets_per_block(mb) + pp. Inside one (mb, s) the packets are
// consecutive and so are their tiles, so slot (pp * tpp + c) of the block's first entry is tile
// (pp * tpp + c) of that block: a head fills block_tiles consecutive slots per mt block.
//
// A head's destination tiles are consecutive tile ids (first_tile .. first_tile + Mt * Nt - 1) of the
// L1-interleaved output, so they walk the L1 banks round robin.
//   walk = false: one TensorAccessor get_noc_addr per tile (the op's current generation).
//   walk = true : RAW BANK WALK (bypasses TensorAccessor::get_noc_addr per page): one bank divide per head,
//                 then bank += 1 (wrap: next bank row, + aligned page) per tile, reading the same firmware
//                 bank tables (bank_to_l1_offset, l1_bank_to_noc_xy) the interleaved accessor reads. The
//                 accessor has no incremental page iterator for interleaved tensors (its Pages iterator calls
//                 get_noc_addr per page), which is what this needs.
// NoC addresses are taken on the fabric's local-chip NoC (edm_to_local_chip_noc = 1 in
// tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_utils.hpp), as addrgen_detail::get_noc_address does on
// Blackhole (Wormhole re-encodes to noc0 coordinates; this experiment is Blackhole only).

#pragma once

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

namespace txs {

constexpr uint8_t kFabricLocalNoc = 1;

#if defined(ARCH_WORMHOLE)
#error "tx_startup bank walk: Blackhole only (Wormhole fabric addresses are noc0-re-encoded)"
#endif

template <uint32_t Mt, uint32_t Nt, uint32_t mt_block, uint32_t tpp, bool walk, typename Accessor>
FORCE_INLINE void fill_head(
    tt_l1_ptr uint64_t* table,
    const Accessor& out,
    uint32_t out_addr,
    uint32_t tile_bytes,
    uint32_t s,
    uint32_t num_heads,
    uint32_t first_tile) {
    constexpr uint32_t num_mt_blocks = (Mt + mt_block - 1) / mt_block;
    constexpr uint32_t full_block_packets = (mt_block * Nt + tpp - 1) / tpp;
    const uint32_t aligned_page = align_power_of_2(tile_bytes, interleaved_addr_gen::get_allocator_alignment<false>());
    uint32_t t = first_tile;
    uint32_t q = interleaved_addr_gen::get_bank_offset_index<false>(t);
    uint32_t b = interleaved_addr_gen::get_bank_index<false>(t, q);
    uint32_t qaddr = out_addr + q * aligned_page;
    for (uint32_t mb = 0; mb < num_mt_blocks; ++mb) {
        const uint32_t block_tiles = (mb + 1 == num_mt_blocks ? (Mt - mb * mt_block) : mt_block) * Nt;
        const uint32_t ppb = (block_tiles + tpp - 1) / tpp;
        tt_l1_ptr uint64_t* e = table + (mb * num_heads * full_block_packets + s * ppb) * tpp;
        for (uint32_t i = 0; i < block_tiles; ++i) {
            if constexpr (walk) {
                e[i] = noc_address_backend::bank_address<false>(
                    b, qaddr + interleaved_addr_gen::get_bank_offset<false>(b), kFabricLocalNoc);
                if (++b == NUM_L1_BANKS) {
                    b = 0;
                    qaddr += aligned_page;
                }
            } else {
                e[i] = out.get_noc_addr(t + i, 0, kFabricLocalNoc);
            }
        }
        t += block_tiles;
    }
}

}  // namespace txs
