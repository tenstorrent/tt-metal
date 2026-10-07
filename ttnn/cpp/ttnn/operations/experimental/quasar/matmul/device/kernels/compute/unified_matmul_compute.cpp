// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Unified matmul compute kernel: per K chunk, multiplies the reader's A slice by its B slice one
// subblock (what DST holds) at a time; running sums spill to C_partials between K chunks (or
// accumulate there via packer_l1_acc) and the last K chunk packs into C_slice for the writer.
// Loop order matches the reader and the writer: batch, MN chunk, K chunk, subblocks, k_tile.
// Compute threads (Quasar NEOs; one thread elsewhere) take the subblocks round-robin in walk order
// and all see the whole A and B slices; a thread's subblocks sit back to back in its share of
// C_slice / C_partials (C_entries_per_thread entries), reserved and pushed once per K chunk.
// Compile-time args are the template parameters, runtime args the function parameters.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/kernel_thread_globals.h"
#include "experimental/kernel_args.h"

template <
    uint32_t batch_size,
    uint32_t K_chunk_tiles,
    uint32_t num_K_chunks,
    uint32_t C_slice_M_padded_tiles,  // C slice dims rounded up to subblock multiples; overshoot is clipped by the
                                      // writer
    uint32_t C_slice_N_padded_tiles,
    uint32_t subblock_M_tiles,
    uint32_t subblock_N_tiles,
    uint32_t num_compute_threads,
    uint32_t C_entries_per_thread,  // C_slice / C_partials entries in one thread's share of the C slice
    uint32_t packer_l1_acc,
    uint32_t partials_format_differs>            // C_partials and C_slice hold different formats
TT_KERNEL void compute(uint32_t num_C_slices) {  // num_C_slices: this core's C slices, per batch
    compute_kernel_hw_startup<SrcOrder::Reverse>(dfb::A_slice, dfb::B_slice, dfb::C_partials);
    constexpr uint32_t A_slice_tiles = C_slice_M_padded_tiles * K_chunk_tiles;
    constexpr uint32_t B_slice_tiles = K_chunk_tiles * C_slice_N_padded_tiles;
    constexpr uint32_t subblock_tiles = subblock_M_tiles * subblock_N_tiles;  // what DST holds
    constexpr uint32_t num_subblocks =
        (C_slice_M_padded_tiles / subblock_M_tiles) * (C_slice_N_padded_tiles / subblock_N_tiles);
    const uint32_t thread = get_my_thread_id();
    // A thread past the last subblock still moves every credit; dummy_unpack / dummy_pack order its pops
    // and pushes after the waits on Quasar (no-ops elsewhere).
    const bool has_subblocks = thread < num_subblocks;

    DataflowBuffer A_slice(dfb::A_slice);
    DataflowBuffer B_slice(dfb::B_slice);
    DataflowBuffer C_slice(dfb::C_slice);
    DataflowBuffer C_partials(dfb::C_partials);

    matmul_block_init(dfb::A_slice, dfb::B_slice, /*transpose=*/0, subblock_N_tiles, subblock_M_tiles, K_chunk_tiles);

    for (uint32_t batch = 0; batch < batch_size; ++batch) {
        for (uint32_t MN_chunk = 0; MN_chunk < num_C_slices; ++MN_chunk) {
            for (uint32_t K_chunk = 0; K_chunk < num_K_chunks; ++K_chunk) {
                const bool last_K_chunk = K_chunk == num_K_chunks - 1;
                // Without packer L1 accumulation every later K chunk reloads the partials; with it only
                // the last one does.
                const bool reload_partials = K_chunk > 0 && (last_K_chunk || !packer_l1_acc);
                // Pack target: C_slice on the last K chunk, C_partials before that.
                auto& pack_target = last_K_chunk ? C_slice : C_partials;
                const uint32_t pack_target_id = last_K_chunk ? uint32_t(dfb::C_slice) : uint32_t(dfb::C_partials);
                A_slice.wait_front(A_slice_tiles);
                B_slice.wait_front(B_slice_tiles);

                // With packer L1 accumulation, K chunk 0 overwrites the partials, later K chunks add DST
                // onto them, and the finished sum is packed without accumulation.
                if constexpr (partials_format_differs) {
                    pack_reconfig_data_format(pack_target_id);
                }
                if constexpr (packer_l1_acc) {
                    pack_reconfig_l1_acc((!last_K_chunk && K_chunk > 0) ? 1 : 0);
                }
#ifdef ARCH_QUASAR
                // Quasar: the pack destination alternates between C_partials and C_slice each K chunk. The pack
                // BFD is baked at pack_init (pack_reconfig_data_format is gasket-only, and only runs when the
                // formats differ), so re-init the packer for the current target every K chunk — otherwise the
                // last chunk's result is packed to the C_partials BFD left by compute_kernel_hw_startup.
                pack_init(pack_target_id);
#endif

                // Release the previous K chunk's partials credits so the reserve below gets this thread's share
                // back. The reloads below still read them: a thread's part of C_partials is exactly one share, so
                // the cursor returns to it, and this NEO is its only writer (each subblock is reloaded before it
                // is overwritten). dummy_unpack orders the pop after the wait on Quasar (a no-op elsewhere).
                if (K_chunk > 0) {
                    C_partials.wait_front(C_entries_per_thread);
                    dummy_unpack(dfb::C_partials);
                    C_partials.pop_front(C_entries_per_thread);
                }
                pack_target.reserve_back(C_entries_per_thread);

                // (m_tile, n_tile) is the subblock's first tile within the C slice; entry_tile is where this
                // thread's next subblock sits in its share.
                uint32_t subblock = 0;
                uint32_t entry_tile = 0;
                for (uint32_t m_tile = 0; m_tile < C_slice_M_padded_tiles; m_tile += subblock_M_tiles) {
                    const uint32_t A_subblock_first_tile = m_tile * K_chunk_tiles;  // A slice tile (m_tile, 0)
                    for (uint32_t n_tile = 0; n_tile < C_slice_N_padded_tiles; n_tile += subblock_N_tiles) {
                        if (subblock++ % num_compute_threads != thread) {
                            continue;
                        }
                        const uint32_t B_subblock_first_tile = n_tile;  // B slice tile (0, n_tile)
                        tile_regs_acquire();
                        if (reload_partials) {
                            // Reload this subblock's partials into DST; the matmul MOP must be re-initialised
                            // after any copy_init.
                            reconfig_data_format_srca(dfb::B_slice, dfb::C_partials);
                            copy_init(dfb::C_partials);
                            copy_block(
                                dfb::C_partials,
                                /*start_in_tile_index=*/entry_tile,
                                /*start_dst_tile_index=*/0,
                                subblock_tiles);
                            reconfig_data_format_srca(dfb::C_partials, dfb::B_slice);
                            matmul_block_init(
                                dfb::A_slice,
                                dfb::B_slice,
                                /*transpose=*/0,
                                subblock_N_tiles,
                                subblock_M_tiles,
                                K_chunk_tiles);
                        }

                        // One matmul_block call per K tile (the LLK has no multi-K-tile call; kt_dim is
                        // only the A-slice row stride).
                        uint32_t A_slice_tile = A_subblock_first_tile;
                        uint32_t B_slice_tile = B_subblock_first_tile;
                        for (uint32_t k_tile = 0; k_tile < K_chunk_tiles; ++k_tile) {
                            matmul_block(
                                dfb::A_slice,
                                dfb::B_slice,
                                A_slice_tile,
                                B_slice_tile,
                                /*idst=*/0,
                                /*transpose=*/0,
                                subblock_N_tiles,
                                subblock_M_tiles,
                                K_chunk_tiles);
                            A_slice_tile += 1;                       // next K tile along the A slice row
                            B_slice_tile += C_slice_N_padded_tiles;  // next K row of the B slice
                        }
                        tile_regs_commit();

                        // Packs continue where the previous subblock's pack left off in the share.
                        tile_regs_wait();
                        pack_block(/*ifrom_dst=*/0, pack_target_id, subblock_tiles);
                        tile_regs_release();
                        entry_tile += subblock_tiles;
                    }
                }

                if (!has_subblocks) {
                    dummy_pack(pack_target_id);
                    dummy_unpack(dfb::A_slice);
                    dummy_unpack(dfb::B_slice);
                }
                pack_target.push_back(C_entries_per_thread);
                A_slice.pop_front(A_slice_tiles);
                B_slice.pop_front(B_slice_tiles);
            }
        }
    }
}
