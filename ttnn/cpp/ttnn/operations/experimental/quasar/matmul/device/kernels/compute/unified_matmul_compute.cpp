// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Unified matmul compute kernel: per K chunk, multiplies the reader's A slice by its B slice one
// subblock (what DST holds) at a time; running sums spill to C_partials between K chunks (or
// accumulate there via packer_l1_acc, or straight in C_slice with accumulate_in_C_slice) and the last K chunk
// packs into C_slice, handing each finished subblock to the writer.
// Loop order matches the reader and the writer: C slice (over all the core's batches), K chunk, subblocks, k_tile.
// Compute threads (Quasar NEOs; one thread elsewhere) take the subblocks round-robin in walk order
// and all see the whole A and B slices; a thread's subblocks sit back to back in its C_entries_per_thread
// entries of C_slice / C_partials, reserved once per K chunk (once per C slice when accumulating in C_slice).
// FUSE_BIAS: K chunk 0 starts DST at the bias row instead of zero, so the partials carry it and the sum gets it
// once. SFPU_OP_*_ACTIVATION: the activation runs on the finished sum in DST before the last K chunk's pack.
// Compile-time args are the template parameters, runtime args the function parameters.

#include <cstdint>

#include "api/compute/bcast.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/kernel_thread_globals.h"
#include "experimental/kernel_args.h"

template <
    uint32_t K_chunk_tiles,
    uint32_t K_chunks_per_C_slice,
    uint32_t K_chunks_per_C_slice_padded,
    uint32_t C_slice_M_padded_tiles,  // C slice dims rounded up to subblock multiples; overshoot is clipped by the
                                      // writer
    uint32_t C_slice_N_padded_tiles,
    uint32_t subblock_M_tiles,
    uint32_t subblock_N_tiles,
    uint32_t num_compute_threads,
    uint32_t C_entries_per_thread,  // C_slice / C_partials entries one thread's subblocks of the C slice take
    uint32_t packer_l1_acc,
    uint32_t accumulate_in_C_slice,              // packer L1 accumulation straight onto the C slice, no C_partials
    uint32_t partials_format_differs>            // C_partials and C_slice hold different formats
TT_KERNEL void compute(uint32_t num_C_slices) {  // num_C_slices: this core's C slices over all its batches
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
#ifdef FUSE_BIAS
    DataflowBuffer bias_row(dfb::bias);
#endif

#ifdef ARCH_QUASAR
    if constexpr (accumulate_in_C_slice) {
        pack_init(dfb::C_slice);
    }
#endif

    matmul_block_init(dfb::A_slice, dfb::B_slice, /*transpose=*/0, subblock_N_tiles, subblock_M_tiles, K_chunk_tiles);

    for (uint32_t MN_chunk = 0; MN_chunk < num_C_slices; ++MN_chunk) {
        for (uint32_t K_chunk = 0; K_chunk < K_chunks_per_C_slice_padded; ++K_chunk) {
            if (K_chunk >= K_chunks_per_C_slice) {
                // Padding K chunk (rounded up to the reader threads): credits only.
                A_slice.wait_front(A_slice_tiles);
                B_slice.wait_front(B_slice_tiles);
                dummy_unpack(dfb::A_slice);
                dummy_unpack(dfb::B_slice);
                A_slice.pop_front(A_slice_tiles);
                B_slice.pop_front(B_slice_tiles);
                continue;
            }
            const bool last_K_chunk = K_chunk == K_chunks_per_C_slice - 1;
            // Without packer L1 accumulation every later K chunk reloads the partials; with it in C_partials only
            // the last one does, and accumulating in C_slice none.
            const bool reload_partials = !accumulate_in_C_slice && K_chunk > 0 && (last_K_chunk || !packer_l1_acc);
            // Pack target: C_slice on the last K chunk (or always when accumulating in it), C_partials before that.
            const bool pack_to_C_slice = accumulate_in_C_slice || last_K_chunk;
            auto& pack_target = pack_to_C_slice ? C_slice : C_partials;
            const uint32_t pack_target_id = pack_to_C_slice ? uint32_t(dfb::C_slice) : uint32_t(dfb::C_partials);
            A_slice.wait_front(A_slice_tiles);
            B_slice.wait_front(B_slice_tiles);
#ifdef FUSE_BIAS
            if (K_chunk == 0) {
                bias_row.wait_front(C_slice_N_padded_tiles);
            }
#endif

            // With packer L1 accumulation K chunk 0 overwrites the running sum and later K chunks add DST onto it;
            // a sum kept in C_partials is packed to C_slice without accumulation on the last K chunk.
            if constexpr (partials_format_differs) {
                pack_reconfig_data_format(pack_target_id);
            }
            if constexpr (packer_l1_acc) {
                pack_reconfig_l1_acc((K_chunk > 0 && (accumulate_in_C_slice || !last_K_chunk)) ? 1 : 0);
            }
#ifdef ARCH_QUASAR
            // Quasar bakes the pack target into pack_init (pack_reconfig_data_format only reprograms the gasket), so
            // re-init it every K chunk as the target alternates; accumulating in C_slice packs there only (init above).
            if constexpr (!accumulate_in_C_slice) {
                pack_init(pack_target_id);
            }
#endif

            // Release the previous K chunk's partials credits so the reserve below gets this thread's entries
            // back. The reloads below still read them: a thread's part of C_partials is exactly those
            // C_entries_per_thread entries, so the cursor returns to them, and this NEO is its only writer (each
            // subblock is reloaded before it is overwritten). dummy_unpack orders the pop after the wait on Quasar
            // (a no-op elsewhere).
            if (!accumulate_in_C_slice && K_chunk > 0) {
                C_partials.wait_front(C_entries_per_thread);
                dummy_unpack(dfb::C_partials);
                C_partials.pop_front(C_entries_per_thread);
            }
            if (!accumulate_in_C_slice || K_chunk == 0) {
                pack_target.reserve_back(C_entries_per_thread);
            }

            // (m_tile, n_tile) is the subblock's first tile within the C slice; entry_tile is where this
            // thread's next subblock sits in its entries.
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
#ifdef FUSE_BIAS
                    if (K_chunk == 0) {
                        // The bias tile over each subblock column, broadcast down every row of the DST tiles
                        // in that column; it is unpacked through SrcB, where the A slice's tiles go.
                        reconfig_data_format_srcb(dfb::A_slice, dfb::bias);
                        unary_bcast_init<BroadcastType::ROW>(dfb::bias);
                        for (uint32_t m = 0; m < subblock_M_tiles; ++m) {
                            for (uint32_t n = 0; n < subblock_N_tiles; ++n) {
                                const uint32_t bias_tile = n_tile + n;  // C slice column of DST tile (m, n)
                                const uint32_t dst_tile = m * subblock_N_tiles + n;
                                unary_bcast<BroadcastType::ROW>(dfb::bias, bias_tile, dst_tile);
                            }
                        }
                        reconfig_data_format_srcb(dfb::bias, dfb::A_slice);
                        matmul_block_init(
                            dfb::A_slice,
                            dfb::B_slice,
                            /*transpose=*/0,
                            subblock_N_tiles,
                            subblock_M_tiles,
                            K_chunk_tiles);
                    }
#endif

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
#ifdef SFPU_OP_INIT_ACTIVATION
                    if (last_K_chunk) {
                        SFPU_OP_INIT_ACTIVATION
                        for (uint32_t i = 0; i < subblock_tiles; ++i) {
                            SFPU_OP_FUNC_ACTIVATION
                        }
                        matmul_block_init(
                            dfb::A_slice,
                            dfb::B_slice,
                            /*transpose=*/0,
                            subblock_N_tiles,
                            subblock_M_tiles,
                            K_chunk_tiles);
                    }
#endif
                    tile_regs_commit();

                    // pack_block continues where the previous subblock's pack left off in the thread's entries.
                    // Accumulating in C_slice, the K chunks before the last add onto the same entries every time.
                    tile_regs_wait();
                    if (accumulate_in_C_slice && !last_K_chunk) {
                        for (uint32_t i = 0; i < subblock_tiles; ++i) {
                            pack_tile<true>(i, dfb::C_slice, entry_tile + i);
                        }
                    } else {
                        pack_block(/*ifrom_dst=*/0, pack_target_id, subblock_tiles);
                    }
                    tile_regs_release();
                    if (last_K_chunk) {
                        C_slice.push_back(subblock_tiles);  // the writer takes each finished subblock
                    }
                    entry_tile += subblock_tiles;
                }
            }

            if (!has_subblocks) {
                dummy_unpack(dfb::A_slice);
                dummy_unpack(dfb::B_slice);
#ifdef FUSE_BIAS
                if (K_chunk == 0) {
                    dummy_unpack(dfb::bias);
                }
#endif
            }
            // The rest of this thread's entries: on the last K chunk the padding past its subblocks (each subblock
            // went when it was packed); before that all of them, unless the sum stays in C_slice until the last.
            const uint32_t entries_left = last_K_chunk            ? C_entries_per_thread - entry_tile
                                          : accumulate_in_C_slice ? 0
                                                                  : C_entries_per_thread;
            if (entries_left > 0) {
                if (!has_subblocks) {
                    dummy_pack(pack_target_id);
                }
                pack_target.push_back(entries_left);
            }
            A_slice.pop_front(A_slice_tiles);
            B_slice.pop_front(B_slice_tiles);
#ifdef FUSE_BIAS
            if (K_chunk == 0) {
                bias_row.pop_front(C_slice_N_padded_tiles);
            }
#endif
        }
    }
    if constexpr (accumulate_in_C_slice) {
        pack_reconfig_l1_acc(0);  // else L1 accumulation stays on into the next program (seen on Quasar)
    }
}
