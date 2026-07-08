// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/circular_buffer.h"

using std::uint32_t;

// Batched-width-sharded matmul compute.
//
// A rank-4 activation A ([d0, d1, M, K], batch = d0*d1) is width(K)-sharded; the reader gathers
// only THIS core's batch block (the Bc batches [b_idx*Bc, (b_idx+1)*Bc)) into full_in0_cb. The
// weights are folded along BOTH batch and N, so this core holds a resident [Bc*K, Nc] weight block
// (in1_cb) covering the same Bc batches and one N-slice. The batched matmul is block-diagonal, so
// there is NO cross-core reduction: this core computes its own [Bc, M, Nc] output block directly.
//
//   full_in0_cb (c_3): gathered batch-block A  [Bc*M_tiles x K_tiles]  (published by reader)
//   in1_cb (c_1):      this core's weight block [Bc*K_tiles x Nc_tiles] (resident in L1)
//   -> out_cb (c_2):   this core's output block [Bc*M_tiles x Nc_tiles] (consumed by writer)
//
// full A (full_in0) layout (built by reader): A is width(K)-sharded across
// `num_senders = K_tiles / inA_K_tiles_per_core` sender cores; the reader copies each sender's
// batch-block slice ([Bc*M_tiles*32, inA_K_tiles_per_core*32], TILE row-major) into full_in0 at
// offset sender*sender_slice_tiles. So full_in0 is SENDER-MAJOR, indexed by the LOCAL batch offset
// bc_i in [0, Bc): tile (row = bc_i*M_tiles + mt, k_global) lives at
//   sender * sender_slice_tiles + (bc_i*M_tiles + mt) * inA_K_tiles_per_core + kc_local
// with sender = k_global / inA_K_tiles_per_core, kc_local = k_global % inA_K_tiles_per_core,
// and sender_slice_tiles = Bc * M_tiles * inA_K_tiles_per_core.
//
// Weight block (in1) layout: [Bc*K, Nc] tile row-major, so weight tile for (batch offset bc_i,
// K-tile kt, N-tile nc) is at (bc_i*K_tiles + kt) * Nc_tiles + nc.
//
// The whole M dimension of each batch is computed in one DST block (out_block_h = M_tiles); the
// factory asserts M_tiles <= 8 so it fits in DST. matmul_block does NOT internally reduce over
// kt_dim (in0_block_w is only the in0 row stride), so the reduction over K is the explicit kt loop
// accumulating into the same DST.
using namespace ckernel;
void kernel_main() {
    constexpr uint32_t M_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t K_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t Nc_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t Bc = get_compile_time_arg_val(3);
    constexpr uint32_t inA_K_tiles_per_core = get_compile_time_arg_val(4);

    constexpr uint32_t full_in0_cb_id = tt::CBIndex::c_3;
    constexpr uint32_t in1_cb_id = tt::CBIndex::c_1;
    constexpr uint32_t out_cb_id = tt::CBIndex::c_2;

    constexpr uint32_t full_in0_num_tiles = Bc * M_tiles * K_tiles;
    constexpr uint32_t in1_num_tiles = Bc * K_tiles * Nc_tiles;
    constexpr uint32_t out_num_tiles = Bc * M_tiles * Nc_tiles;
    constexpr uint32_t sender_slice_tiles = Bc * M_tiles * inA_K_tiles_per_core;

    // The whole M dimension of a batch is computed in one DST block.
    constexpr uint32_t out_block_h = M_tiles;
    constexpr uint32_t out_block_w = 1;
    // in0 row stride within a sender slice (K-tiles per M-row) == matmul kt_dim.
    constexpr uint32_t in0_block_w = inA_K_tiles_per_core;

    CircularBuffer full_in0_cb(full_in0_cb_id);
    CircularBuffer in1_cb(in1_cb_id);
    CircularBuffer out_cb(out_cb_id);

    compute_kernel_hw_startup<SrcOrder::Reverse>(full_in0_cb_id, in1_cb_id, out_cb_id);

    full_in0_cb.wait_front(full_in0_num_tiles);
    in1_cb.wait_front(in1_num_tiles);

    matmul_block_init(full_in0_cb_id, in1_cb_id, false, out_block_w, out_block_h, in0_block_w);

    out_cb.reserve_back(out_num_tiles);
    for (uint32_t bc_i = 0; bc_i < Bc; ++bc_i) {
        const uint32_t in0_batch_base = (bc_i * M_tiles) * inA_K_tiles_per_core;  // row offset in a sender slice
        const uint32_t in1_batch_base = bc_i * K_tiles * Nc_tiles;                // this batch's weight sub-block
        for (uint32_t nc = 0; nc < Nc_tiles; ++nc) {
            tile_regs_acquire();
            for (uint32_t kt = 0; kt < K_tiles; ++kt) {
                const uint32_t sender = kt / inA_K_tiles_per_core;
                const uint32_t kc_local = kt - sender * inA_K_tiles_per_core;
                // in0: K-column kc_local of batch b in this sender's slice; matmul reads
                // + r*in0_block_w for r in [0, M_tiles), i.e. every M-row of that K-column.
                const uint32_t in0_tile = sender * sender_slice_tiles + in0_batch_base + kc_local;
                const uint32_t in1_tile = in1_batch_base + kt * Nc_tiles + nc;
                matmul_block(
                    full_in0_cb_id, in1_cb_id, in0_tile, in1_tile, 0, false, out_block_w, out_block_h, in0_block_w);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t mt = 0; mt < out_block_h; ++mt) {
                pack_tile<true>(mt, out_cb_id, (bc_i * M_tiles + mt) * Nc_tiles + nc);
            }
            tile_regs_release();
        }
    }
    out_cb.push_back(out_num_tiles);
    full_in0_cb.pop_front(full_in0_num_tiles);
}
