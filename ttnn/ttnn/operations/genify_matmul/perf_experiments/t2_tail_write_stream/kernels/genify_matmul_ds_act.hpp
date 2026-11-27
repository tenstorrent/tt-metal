// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Regime R4 `dram_sharded` A broadcast — the one implementation of an A round, shared by the reader
// (DS_ACT_ON_WRITER = False: rounds interleaved with the B stream on NCRISC / NoC0) and the writer
// (DS_ACT_ON_WRITER = True, Refinement 6: rounds on BRISC / NoC1, off the B stream's RISC and NoC).
//
// A is L1 width-sharded over K: shard r (one per A core, in K order) holds K tiles [r*kw, (r+1)*kw) of
// every row. K-block (row block mb, K-block kb) of the sender's row-major shard (rows x kw) starts at
// row mb*block_m, column (kb % k_blocks_per_round) * block_k -- one contiguous run, because the host
// splits a shard (k_blocks_per_round > 1) only when block_m == 1. In round kb / k_blocks_per_round the
// shard's owner multicasts the K-block into every worker's reserved cb_act_tiles slot (rotating senders,
// ttnn.Mcast + McastArgs); a worker that owns the shard sends instead of receiving.

#pragma once

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

namespace genify_ds {

template <uint32_t block_m_tiles, uint32_t block_k_tiles, uint32_t k_blocks_per_round, uint32_t a_tile_bytes>
struct ActKBlocks {
    static constexpr uint32_t shard_k_tiles = block_k_tiles * k_blocks_per_round;
    static constexpr uint32_t tiles = block_m_tiles * block_k_tiles;
    static constexpr uint32_t bytes = tiles * a_tile_bytes;
    uint32_t a_addr;  // the A shard's L1 address (the same on every A core)

    FORCE_INLINE uint32_t src(uint32_t mb, uint32_t kb) const {
        return a_addr + (mb * block_m_tiles * shard_k_tiles + (kb % k_blocks_per_round) * block_k_tiles) * a_tile_bytes;
    }
    static FORCE_INLINE uint32_t round(uint32_t kb) { return kb / k_blocks_per_round; }
};

// A worker's round for K-block (mb, kb) into its reserved slot `dst` (the caller reserves / pushes).
template <typename Blocks, typename Mc, typename Sender, typename Receiver>
FORCE_INLINE void worker_round(
    const Blocks& blocks, const Mc& mc, Sender& sender, Receiver& receiver, uint32_t mb, uint32_t kb, uint32_t dst) {
    const uint32_t round = Blocks::round(kb);
#ifdef GM_ABLATE_A_ROUND
    return;  // perf tooling: no round at all (no handshake, no payload)
#endif
    if (mc.should_send(round)) {
#ifdef GM_ABLATE_A_SEND
        sender->send(blocks.src(mb, kb), dst, 32);  // perf tooling: the round's handshake, no payload
#else
        sender->send(blocks.src(mb, kb), dst, Blocks::bytes);
#endif
    } else {
        receiver->receive(round);
    }
}

// An A core that is not a bank worker: send this core's rounds into the workers' slots, no CB traffic.
// The workers' cb_act_tiles write pointer visits slot (K-block % depth) in lockstep from `act_base`.
// Cores of the receiver rectangle that are neither only land the payload (nothing to do here).
template <typename Blocks, typename Mc, typename Sender>
FORCE_INLINE void send_only(
    const Blocks& blocks,
    const Mc& mc,
    Sender& sender,
    uint32_t act_base,
    uint32_t act_cb_blocks,
    uint32_t nb_m,
    uint32_t num_k_blocks) {
#ifdef GM_ABLATE_A_ROUND
    return;
#endif
    if (!sender) {
        return;
    }
    uint32_t slot = 0;
    for (uint32_t mb = 0; mb < nb_m; ++mb) {
        for (uint32_t kb = 0; kb < num_k_blocks; ++kb) {
            if (mc.should_send(Blocks::round(kb))) {
#ifdef GM_ABLATE_A_SEND
                sender->send(blocks.src(mb, kb), act_base + slot * Blocks::bytes, 32);
#else
                sender->send(blocks.src(mb, kb), act_base + slot * Blocks::bytes, Blocks::bytes);
#endif
            }
            slot = (slot + 1 == act_cb_blocks) ? 0 : slot + 1;
        }
    }
}

}  // namespace genify_ds
