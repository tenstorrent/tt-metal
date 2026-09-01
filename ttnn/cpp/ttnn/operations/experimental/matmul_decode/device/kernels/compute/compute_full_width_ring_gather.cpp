// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/circular_buffer.h"

using std::uint32_t;

// Compute for the ring-gather in0 path. Instead of waiting for a fully assembled full_in0 CB
// (as compute_full_width_sharded.cpp does), this kernel drains per-shard K-blocks out of the
// ring-gather CBs (cb_in2_cw, cb_in2_ccw, and optionally cb_in0 for source-and-compute cores)
// and accumulates each shard's K-slice into the output CB via pack_reconfig_l1_acc(1). Because
// each shard is exactly one sender's K-slice (inA_K_tiles_per_core K-tiles), the inner K loop
// per shard is a single K-block; the outer loop over shards replaces the old K loop.
//
// The runtime args carry the ordered list of global sender IDs whose shards this core will see
// on each ring; the compute kernel uses the sender ID to index into the resident in1 (weights)
// CB. That decouples this kernel from the ring's routing decisions (host owns those).
//
// Weight residency: in1 is L1-resident (globally allocated, either directly over the weight
// tensor or, in the packed-weight case, over an offset region of a fused L1 tensor). The kernel
// therefore does not wait on cb_in1 per K-block -- it's addressable by tile index throughout.

using namespace ckernel;

// Sentinel signalling "this core does not own a local shard" (i.e. it is compute-only).
static constexpr uint32_t NO_LOCAL_SENDER = 0xFFFFFFFFu;

void kernel_main() {
    constexpr uint32_t out_block_w = 1;

    constexpr uint32_t M_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t K_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t N_tiles_per_core = get_compile_time_arg_val(2);
    constexpr uint32_t inA_K_tiles_per_core = get_compile_time_arg_val(3);
    constexpr uint32_t num_senders = get_compile_time_arg_val(4);  // max num_arriving_cw
    // has_local_shard, local_sender_id, num_arriving_cw, and num_arriving_ccw are runtime args
    // (see below): a single program launch covers overlap and non-overlap cores, and each
    // compute core's arrival count varies depending on whether it's also a source.

    constexpr uint32_t out_block_h = M_tiles;
    constexpr uint32_t in0_block_w = inA_K_tiles_per_core;
    constexpr uint32_t shard_num_tiles = M_tiles * inA_K_tiles_per_core;

    constexpr uint32_t in0_local_cb_id = get_named_compile_time_arg_val("cb_in0");
    constexpr uint32_t in2_cw_cb_id = get_named_compile_time_arg_val("cb_in2_cw");
    constexpr uint32_t in2_ccw_cb_id = get_named_compile_time_arg_val("cb_in2_ccw");
    constexpr uint32_t in1_cb_id = get_named_compile_time_arg_val("cb_in1");
    constexpr uint32_t out_cb_id = get_named_compile_time_arg_val("cb_out");

    CircularBuffer in1_cb(in1_cb_id);
    CircularBuffer out_cb(out_cb_id);
    CircularBuffer in2_cw_cb(in2_cw_cb_id);
    CircularBuffer in2_ccw_cb(in2_ccw_cb_id);

    // cb_in2_cw is always allocated on this core (whole-ring CB) and has the same tile
    // format as cb_in0 / cb_in2_ccw, so it's a safe HW-init CB regardless of runtime counts.
    compute_kernel_hw_startup<SrcOrder::Reverse>(in2_cw_cb_id, in1_cb_id, out_cb_id);
    matmul_block_init(in2_cw_cb_id, in1_cb_id, false, out_block_w, out_block_h, in0_block_w);

    // Runtime args layout:
    //   [0] has_local_shard, [1] local_sender_id, [2] num_arriving_cw, [3] num_arriving_ccw,
    //   [4..4+num_arriving_cw-1] cw_sender_ids, [4+num_arriving_cw..] ccw_sender_ids
    uint32_t rt_idx = 0;
    const uint32_t has_local_shard = get_arg_val<uint32_t>(rt_idx++);
    const uint32_t local_sender_id = get_arg_val<uint32_t>(rt_idx++);
    const uint32_t num_arriving_cw = get_arg_val<uint32_t>(rt_idx++);
    const uint32_t num_arriving_ccw = get_arg_val<uint32_t>(rt_idx++);
    // Only take addresses of slots that actually exist -- the watcher trips
    // "unique runtime arg index out of bounds" if we point past the last written arg even when
    // the resulting pointer is never dereferenced.
    volatile tt_l1_ptr uint32_t* cw_sender_ids = nullptr;
    if (num_arriving_cw > 0) {
        cw_sender_ids = (volatile tt_l1_ptr uint32_t*)get_arg_addr(rt_idx);
        rt_idx += num_arriving_cw;
    }
    volatile tt_l1_ptr uint32_t* ccw_sender_ids = nullptr;
    if (num_arriving_ccw > 0) {
        ccw_sender_ids = (volatile tt_l1_ptr uint32_t*)get_arg_addr(rt_idx);
        rt_idx += num_arriving_ccw;
    }

    // cb_in1 is globally allocated over the L1-resident weight (or a packed region of one), so
    // there is no producer that would ever push it -- indexing into it by tile is enough.
    // Mirrors the non-GCB path of compute_full_width_sharded.cpp, which likewise gates its
    // in1_cb.wait_front behind ENABLE_GLOBAL_CB. Waiting here unconditionally deadlocks.
    if (has_local_shard) {
        CircularBuffer in0_local_cb(in0_local_cb_id);
        in0_local_cb.wait_front(shard_num_tiles);
    }

    out_cb.reserve_back(M_tiles * N_tiles_per_core);

    // Accumulate one K-slice (one shard) into the output CB. All slices except the first
    // enable packer accumulate mode so the packer sums into the resident output tiles rather
    // than overwriting them. With ENABLE_ALL_GATHER the last shard publishes each N-column
    // (column-major dest) as soon as it is K-complete so the writer can overlap fabric send.
    bool packer_in_acc = false;
    auto process_shard = [&](uint32_t in0_cb, uint32_t sender_id, bool publish_columns) {
        const uint32_t in1_k_row_base = sender_id * inA_K_tiles_per_core;  // K-tile index
        for (uint32_t bw = 0; bw < N_tiles_per_core; ++bw) {
            tile_regs_acquire();
            for (uint32_t kc = 0; kc < inA_K_tiles_per_core; ++kc) {
                const uint32_t in0_tile = kc;
                const uint32_t in1_tile = (in1_k_row_base + kc) * N_tiles_per_core + bw;
                matmul_block(in0_cb, in1_cb_id, in0_tile, in1_tile, 0, false, out_block_w, out_block_h, in0_block_w);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t mt = 0; mt < out_block_h; ++mt) {
#ifdef ENABLE_ALL_GATHER
                // Intermediate shards pack at dest bw*M+mt from the original wr_ptr (no
                // push). On the last shard each push_back(M_tiles) advances wr_ptr, so
                // dest is just mt relative to the current column.
                pack_tile<true>(mt, out_cb_id, publish_columns ? mt : bw * M_tiles + mt);
#else
                pack_tile<true>(mt, out_cb_id, mt * N_tiles_per_core + bw);
#endif
            }
            tile_regs_release();
#ifdef ENABLE_ALL_GATHER
            if (publish_columns) {
                out_cb.push_back(M_tiles);
            }
#endif
        }
        if (!packer_in_acc) {
            pack_reconfig_l1_acc(1);
            packer_in_acc = true;
        }
    };

    const uint32_t total_shards = num_arriving_cw + num_arriving_ccw + has_local_shard;
    uint32_t shard_i = 0;
    // Consume CW ring shards in arrival order.
    for (uint32_t k = 0; k < num_arriving_cw; ++k) {
        in2_cw_cb.wait_front(shard_num_tiles);
        process_shard(in2_cw_cb_id, cw_sender_ids[k], ++shard_i == total_shards);
        in2_cw_cb.pop_front(shard_num_tiles);
    }
    // Then CCW ring shards.
    for (uint32_t k = 0; k < num_arriving_ccw; ++k) {
        in2_ccw_cb.wait_front(shard_num_tiles);
        process_shard(in2_ccw_cb_id, ccw_sender_ids[k], ++shard_i == total_shards);
        in2_ccw_cb.pop_front(shard_num_tiles);
    }
    // Finally the local shard on source-and-compute cores.
    if (has_local_shard) {
        process_shard(in0_local_cb_id, local_sender_id, ++shard_i == total_shards);
    }

    // Turn accumulate mode back off in case anything after this expects the default packer state.
    pack_reconfig_l1_acc(0);

#ifndef ENABLE_ALL_GATHER
    out_cb.push_back(M_tiles * N_tiles_per_core);
#endif
}
