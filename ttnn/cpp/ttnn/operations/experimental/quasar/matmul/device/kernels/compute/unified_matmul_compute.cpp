// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Unified matmul compute kernel: per K chunk, multiplies the reader's A slice by its B slice one
// subblock (what DST holds) at a time; running sums spill to C_partials between K chunks (or
// accumulate there via packer_l1_acc) and the last K chunk packs into C_slice for the writer.
// Loop order matches the reader and the writer: batch, MN chunk, K chunk, subblock rounds, k_tile.
//
// Compute threads (Quasar NEOs; one thread elsewhere): the C slice's subblocks, numbered across N
// then down M, are dealt round-robin, thread t taking subblocks t, t + T, ... Every thread sees the
// whole A and B slices (one resident copy). C_slice and C_partials are striped by thread, and the
// pack / unpack tile indices are dense, so a lane is addressed one entry per credit. Every thread
// runs the same number of rounds; a round past the last subblock only moves credits, so the lanes
// carry equal traffic (the writer's round-robin over lanes and the packer's L1 accumulation both
// need that).
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
    uint32_t packer_l1_acc,
    uint32_t partials_format_differs>            // C_partials and C_slice hold different formats
TT_KERNEL void compute(uint32_t num_C_slices) {  // num_C_slices: this core's C slices, per batch
    compute_kernel_hw_startup<SrcOrder::Reverse>(dfb::A_slice, dfb::B_slice, dfb::C_partials);
    constexpr uint32_t A_slice_tiles = C_slice_M_padded_tiles * K_chunk_tiles;
    constexpr uint32_t B_slice_tiles = K_chunk_tiles * C_slice_N_padded_tiles;
    constexpr uint32_t subblock_tiles = subblock_M_tiles * subblock_N_tiles;  // what DST holds
    constexpr uint32_t subblocks_across_N = C_slice_N_padded_tiles / subblock_N_tiles;
    constexpr uint32_t num_subblocks = (C_slice_M_padded_tiles / subblock_M_tiles) * subblocks_across_N;
    constexpr uint32_t subblock_rounds = (num_subblocks + num_compute_threads - 1) / num_compute_threads;
    const uint32_t thread = get_my_thread_id();

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
                DataflowBuffer& pack_target = last_K_chunk ? C_slice : C_partials;
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

                for (uint32_t round = 0; round < subblock_rounds; ++round) {
                    const uint32_t subblock = round * num_compute_threads + thread;
                    if (subblock >= num_subblocks) {
                        // Credit-only round: move this lane's partials and output credits like a real
                        // subblock would (dummy_unpack / dummy_pack order the pop / push after the wait on
                        // Quasar; no-ops elsewhere).
                        if (reload_partials) {
                            C_partials.wait_front(subblock_tiles);
                            dummy_unpack(dfb::C_partials);
                            C_partials.pop_front(subblock_tiles);
                        }
                        pack_target.reserve_back(subblock_tiles);
                        dummy_pack(pack_target_id);
                        pack_target.push_back(subblock_tiles);
                        continue;
                    }
                    // (m_tile, n_tile) is the subblock's first tile within the C slice.
                    const uint32_t m_tile = (subblock / subblocks_across_N) * subblock_M_tiles;
                    const uint32_t n_tile = (subblock % subblocks_across_N) * subblock_N_tiles;
                    const uint32_t A_subblock_first_tile = m_tile * K_chunk_tiles;  // A slice tile (m_tile, 0)
                    const uint32_t B_subblock_first_tile = n_tile;                  // B slice tile (0, n_tile)
                    tile_regs_acquire();
                    if (reload_partials) {
                        // Reload this subblock's partials into DST, one lane entry per pop; the matmul MOP
                        // must be re-initialised after any copy_init.
                        reconfig_data_format_srca(dfb::B_slice, dfb::C_partials);
                        copy_init(dfb::C_partials);
                        C_partials.wait_front(subblock_tiles);
                        for (uint32_t tile = 0; tile < subblock_tiles; ++tile) {
                            copy_tile(dfb::C_partials, /*in_tile_index=*/0, /*dst_tile_index=*/tile);
                            C_partials.pop_front(1);
                        }
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

                    // Pack the subblock one lane entry per push.
                    pack_target.reserve_back(subblock_tiles);
                    tile_regs_wait();
                    for (uint32_t tile = 0; tile < subblock_tiles; ++tile) {
                        pack_tile(tile, pack_target_id);
                        pack_target.push_back(1);
                    }
                    tile_regs_release();
                }

                if constexpr (packer_l1_acc) {
                    // The entries pushed this K chunk are only credits (the sums live in L1): pop them so the next
                    // K chunk lands on the same addresses. Two exceptions: the last K chunk pushed nothing here, and
                    // the second-to-last K chunk's entries are what the last K chunk reloads.
                    const bool second_to_last_K_chunk = K_chunk + 2 == num_K_chunks;
                    if (!last_K_chunk && !second_to_last_K_chunk) {
                        // Pop without reading: dummy_unpack orders the pop after the wait on Quasar (a no-op
                        // elsewhere).
                        for (uint32_t round = 0; round < subblock_rounds; ++round) {
                            C_partials.wait_front(subblock_tiles);
                            dummy_unpack(dfb::C_partials);
                            C_partials.pop_front(subblock_tiles);
                        }
                    }
                }

                if (thread >= num_subblocks) {
                    // A thread with no subblock at all issued no unpack this K chunk: order the pops after the
                    // waits (Quasar TEN-4746; no-ops elsewhere).
                    dummy_unpack(dfb::A_slice);
                    dummy_unpack(dfb::B_slice);
                }
                A_slice.pop_front(A_slice_tiles);
                B_slice.pop_front(B_slice_tiles);
            }
        }
    }
}
