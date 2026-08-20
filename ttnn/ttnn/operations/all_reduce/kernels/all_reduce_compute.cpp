// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// all_reduce — compute (TRISC), Phase B only (the reduce MeshProgramDescriptor
// wires this kernel; Phase A has no compute stage).
//
// Local element-wise N-way tile sum. For each owned output-tile position i, the
// reader pushes the N gather blocks' tile i into cb_gathered_shards (block order
// c=0..N-1). This kernel sums them into ONE DST register and packs the result:
//     output_tile[i] = Σ_{c=0..N-1} gather_buffer_tile[c*P + i]
//
// Seed-then-accumulate into DST[0] (the proven reduce_to_one idiom): copy_tile
// seeds DST[0] with block 0, then binary_dest_reuse_tiles<ELWADD, DEST_TO_SRCA>
// pulls DST[0] back into SrcA and adds block c for c=1..N-1. Using only DST[0] is
// DST-capacity-safe for any N and for float32 (DST holds 4 tiles); fp32_dest_acc
// (set on the compute config) keeps the sum in fp32 before the pack.
//
// This is the RAW compute API by design: reduce_helpers' reduce() COLLAPSES the
// within-tile 32x32 dims, and there is no element-wise-add chain helper in this
// tree — see op_design.md "Helpers considered and rejected (Phase B)".

#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"

void kernel_main() {
    constexpr uint32_t cb_gathered_shards = get_compile_time_arg_val(0);
    constexpr uint32_t cb_reduced = get_compile_time_arg_val(1);
    constexpr uint32_t num_devices = get_compile_time_arg_val(2);  // N blocks to sum

    const uint32_t num_tiles = get_arg_val<uint32_t>(0);  // owned output-tile positions

    // Boot init (full hw_configure for unpack/math/pack). Both binary operands come
    // from cb_gathered_shards; the pack target is cb_reduced.
    binary_op_init_common(cb_gathered_shards, cb_gathered_shards, cb_reduced);

    for (uint32_t t = 0; t < num_tiles; ++t) {
        // Reader pushed N tiles for this position (block order c=0..N-1).
        copy_tile_to_dst_init_short(cb_gathered_shards);
        cb_wait_front(cb_gathered_shards, num_devices);

        tile_regs_acquire();
        // Seed DST[0] = block 0's tile.
        copy_tile(cb_gathered_shards, 0, 0);
        // Accumulate blocks 1..N-1 into DST[0].
        binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(
            cb_gathered_shards);
        for (uint32_t c = 1; c < num_devices; ++c) {
            binary_dest_reuse_tiles<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(
                cb_gathered_shards, c, 0);
        }
        tile_regs_commit();
        cb_pop_front(cb_gathered_shards, num_devices);

        // Pack the single summed tile to the output CB.
        cb_reserve_back(cb_reduced, 1);
        tile_regs_wait();
        pack_tile(0, cb_reduced);
        tile_regs_release();
        cb_push_back(cb_reduced, 1);
    }
}
