// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// OPTION B — PROGRAM A (tilize-only) OUTPUT-DRAIN data-movement kernel.
//
// Program A's compute kernel (conv_tilize_only_metal2.cpp) tilizes num_blocks x in0_block_w tiles STRAIGHT INTO
// dfb::out (borrowed from the op's OUTPUT tensor). DFB_OUT.num_entries is sized to the full [M, full_K] shard =
// num_blocks * in0_block_w tiles, so the packer fills the ring EXACTLY. On Quasar a borrowed-output DFB ring
// holds only num_entries-1 outstanding entries, so the compute's FINAL reserve_back(in0_block_w) stalls at
// exact-fill unless a CONSUMER advances the ring's read/ack credit. This kernel IS that consumer: it pops
// dfb::out in lockstep with the compute's per-block push_back. The tilized data is ALREADY resident in the
// borrowed OUTPUT L1 (the packer wrote it in place), so this is CREDIT-ONLY — no NOC / data movement. It runs
// on the DM processor left free in Program A (the weights writer is skipped) and on a DIFFERENT DM processor
// than the activation reader (per the Quasar per-node SPSC DFB invariant). Matches conv_tilize_only_metal2.cpp's
// num_blocks / block-width expression exactly so pops track pushes 1:1.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t in0_block_w = get_arg(args::in0_block_w);
    constexpr uint32_t in0_num_blocks_h = get_arg(args::in0_num_blocks_h);
    constexpr uint32_t reader_num_h_subblocks = get_arg(args::reader_num_h_subblocks);

    // Identical block count / width to conv_tilize_only_metal2.cpp so the pops track the compute's pushes 1:1.
    const uint32_t num_blocks = in0_num_blocks_h * reader_num_h_subblocks;

    DataflowBuffer out_cb(dfb::out);
    for (uint32_t block = 0; block < num_blocks; ++block) {
        out_cb.wait_front(in0_block_w);  // block until the compute has push_back'd this block
        out_cb.pop_front(in0_block_w);   // ack -> frees a ring slot for the compute's next reserve_back
    }
}
