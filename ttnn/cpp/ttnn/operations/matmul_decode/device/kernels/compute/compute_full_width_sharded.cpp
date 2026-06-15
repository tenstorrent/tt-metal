// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/matmul.h"

using std::uint32_t;

// Full-width-sharded matmul compute: C = A @ B per core.
//
// Inputs (published by reader_full_width_sharded):
//   full_in0_cb (c_3): full gathered A in sender-major K layout
//   in1_cb (c_1):      this core's B shard (buffer-backed, already in L1)
// Output:
//   out_cb (c_2):      this core's output shard (buffer-backed)
//
// Blocking: in0_block_w (K) = 1, out_block_h (M) = 1, out_block_w (N) = 1.
//
// M_tiles>1 fix: the original kernel hardcoded a single M-tile (out_block_h=1,
// in0 index without an M offset, pack always to out slot 0), so it only ever
// computed M-tile 0 and left M-rows 1..M_tiles-1 unwritten/garbage. We now loop
// over every M-tile (bh) x N-tile (bw), accumulating over K, and pack each
// (bh,bw) tile to its sequential output slot.
//
// full_in0 layout (built by reader_full_width_sharded): A is width(K)-sharded
// across `num_senders` sender cores, each holding a contiguous [M_tiles*32,
// inA_K_tiles_per_core*32] slice in TILE row-major order, and the reader copies
// each sender's whole slice into full_in0 at offset sender*shard_num_tiles. So
// full_in0 is SENDER-MAJOR: the tile for global (m, k_global) lives at
//   sender * (M_tiles * inA_K_tiles_per_core)  // sender base
//   + m * inA_K_tiles_per_core                 // M-row within the sender slice
//   + kc_local                                 // K-col within the sender slice
// where sender = k_global / inA_K_tiles_per_core, kc_local = k_global % ...
// (For M_tiles==1 this collapses to k_global, matching the old behaviour.)

// deep-plan_13 Phase 1 -- REAL fat systolic fill.
//
// The original kernel hardcoded out_block_h=out_block_w=1, packing ONE DST tile per
// tile_regs_acquire -> 1x1 systolic underfill (the established ~2-4x loss). We now take
// (out_subblock_h, out_subblock_w) as NEW compile-time args (args 4,5 -- the previously
// DEAD last_out_block_h/num_blocks_h slots are repurposed) and fill DST with
// out_h*out_w tiles per acquire via a single matmul_block(ct_dim=out_w, rt_dim=out_h)
// per inner-K step, then pack the whole out_h x out_w rectangle.
//
// DST-slot contract (from api/compute/matmul.h matmul_block):
//   matmul_block(in0,in1, in0_tile, in1_tile, idst, transpose, ct_dim, rt_dim, kt_dim)
//   computes a rt_dim(rows) x ct_dim(cols) output rectangle into DST slots
//   idst .. idst+rt_dim*ct_dim-1, reading rt_dim CONSECUTIVE in0 tiles and ct_dim
//   CONSECUTIVE in1 tiles, and ACCUMULATES across inner-K calls into the SAME slots.
//
// out_w (N) fat-fill is SAFE WITHOUT a probe: in1 (B) is [K_tiles x N_tiles_per_core]
// row-major, so the out_w in1 tiles for a fixed K-row are contiguous (in1_tile =
// k*N_tiles_per_core + bw0, +1 per col). out_h (M) fat-fill is GATED by P0-A: full_in0
// is SENDER-MAJOR (A tiles for a fixed K-col across M-rows are at stride
// inA_K_tiles_per_core, NOT contiguous) so rt_dim>1 cannot read a contiguous out_h x kt
// A rectangle. v1 ships out_w-only (factory clamps out_subblock_h=1); out_h>1 is enabled
// only if P0-A proved M-fill (factory then derives out_h from mmd_get_subblock_sizes).

using namespace ckernel;
void kernel_main() {
    constexpr uint32_t M_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t K_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t N_tiles_per_core = get_compile_time_arg_val(2);
    constexpr uint32_t inA_K_tiles_per_core = get_compile_time_arg_val(3);
    constexpr uint32_t out_subblock_h = get_compile_time_arg_val(4);  // M-fill (P0-A gated; default 1)
    constexpr uint32_t out_subblock_w = get_compile_time_arg_val(5);  // N-fill (safe)
    // deep-plan_14 Lever 0: in0_block_w now a compile-time arg (was hardcoded constexpr 1).
    // Default 1 (factory passes 1 unless overridden) keeps the K loop byte-identical.
    constexpr uint32_t in0_block_w = get_compile_time_arg_val(6);

    constexpr uint32_t in0_cb_id = tt::CBIndex::c_3;
    constexpr uint32_t in1_cb_id = tt::CBIndex::c_1;
    constexpr uint32_t out_cb_id = tt::CBIndex::c_2;

    constexpr uint32_t in0_num_tiles = M_tiles * K_tiles;

    constexpr uint32_t num_K_blocks = K_tiles / in0_block_w;
    // tiles per sender slice in full_in0 (== reader's shard_num_tiles)
    constexpr uint32_t sender_slice_tiles = M_tiles * inA_K_tiles_per_core;

    // Gathered A is a regular CB (reader publishes via push_back).
    cb_wait_front(in0_cb_id, in0_num_tiles);
    // in1/out are buffer-backed: data is already in L1 (see vecadd_sharding).

    // deep-plan_14 Lever 1 (Route B): the fat-fill issues out_subblock_h SEPARATE
    // matmul_block(rt_dim=1) calls (sender-major A is not M-contiguous), so the block init
    // is configured for rt_dim=1 (ct=out_subblock_w, kt=in0_block_w) to match the per-row
    // call shape. At out_subblock_h==1 this is identical to the original init.
    mm_block_init(in0_cb_id, in1_cb_id, out_cb_id, false, out_subblock_w, 1, in0_block_w);

    // Reserve the whole output shard ([M_tiles x N_tiles_per_core] tiles, row-major).
    cb_reserve_back(out_cb_id, M_tiles * N_tiles_per_core);
    // Blocked fat-fill: walk M in out_subblock_h-row blocks, N in out_subblock_w-col blocks.
    //
    // deep-plan_14 Lever 1 (Route B): full_in0 is SENDER-MAJOR -- for a fixed K-col the M-rows
    // are at stride inA_K_tiles_per_core (NOT contiguous), so a single matmul_block(rt_dim>1)
    // would read the WRONG in0 tiles (P0-A falsified this: out_h=2 -> PCC 0.50). Route B keeps
    // A sender-major and issues out_subblock_h SEPARATE matmul_block(rt_dim=1) calls, one per
    // M-row, each with the correct M-strided in0_tile, packing into the adjacent DST slot
    // i*out_subblock_w + j. This fills the out_h x out_w DST rectangle per acquire (fewer
    // acquire/pack cycles than 1x1) WITHOUT requiring A-contiguity. At out_subblock_h==1 this
    // collapses to the original single-call path (BYTE-IDENTICAL).
    for (uint32_t mt0 = 0; mt0 < M_tiles; mt0 += out_subblock_h) {
        for (uint32_t bw0 = 0; bw0 < N_tiles_per_core; bw0 += out_subblock_w) {
            tile_regs_acquire();
            for (uint32_t kb = 0; kb < num_K_blocks; ++kb) {
                // Global K-tile index at the start of this K-block (in0_block_w-wide).
                const uint32_t k = kb * in0_block_w;
                // Translate global K-tile index k into its sender-major slot in full_in0.
                const uint32_t sender = k / inA_K_tiles_per_core;
                const uint32_t kc_local = k - sender * inA_K_tiles_per_core;
                // in1 base tile for (k, bw0); ct_dim>1 walks +1 per col (B row-major, contiguous).
                const uint32_t in1_tile = k * N_tiles_per_core + bw0;
                // Issue one rt_dim=1 matmul_block per M-row of the subblock, each reading the
                // correct sender-major in0 tile for (mt0+i, k) and accumulating into DST rows
                // [i*out_subblock_w, i*out_subblock_w+out_subblock_w).
                for (uint32_t i = 0; i < out_subblock_h; ++i) {
                    const uint32_t in0_tile =
                        sender * sender_slice_tiles + (mt0 + i) * inA_K_tiles_per_core + kc_local;
                    matmul_block(
                        in0_cb_id,
                        in1_cb_id,
                        in0_tile,
                        in1_tile,
                        i * out_subblock_w,  // idst: this M-row's slot range
                        false,
                        out_subblock_w,  // ct_dim (N cols)
                        1,               // rt_dim = 1 (one M-row per call -- Route B)
                        in0_block_w);    // kt_dim
                }
            }
            tile_regs_commit();
            tile_regs_wait();
            // Pack the out_h x out_w rectangle from DST to its row-major output slots.
            for (uint32_t i = 0; i < out_subblock_h; ++i) {
                for (uint32_t j = 0; j < out_subblock_w; ++j) {
                    const uint32_t dst = i * out_subblock_w + j;
                    const uint32_t out_slot = (mt0 + i) * N_tiles_per_core + (bw0 + j);
                    pack_tile<true>(dst, out_cb_id, out_slot);
                }
            }
            tile_regs_release();
        }
    }
    cb_push_back(out_cb_id, M_tiles * N_tiles_per_core);

    cb_pop_front(in0_cb_id, in0_num_tiles);
}
