// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Unified matmul compute kernel: C = A x B for one cluster, as a classic blocked GEMM.
//
// GEMM view, all sizes in 32x32 tiles: C[M x N] = A[M x K] x B[K x N], batch_size times. An MN chunk is
// MN_chunk_M_tiles x MN_chunk_N_tiles tiles of C, the L1-fittable piece of the output region this cluster
// owns; normally the region is one chunk. This kernel produces num_MN_chunks chunks for every batch and does
// not care where in C they sit. For every chunk it accumulates over K, K_iteration_tiles per iteration: the
// reader delivers one A slice ([MN_chunk_M_tiles][K_iteration_tiles] tiles) and one B slice
// ([K_iteration_tiles][MN_chunk_N_tiles] tiles) per iteration, and the MATH engine multiplies them one
// subblock (subblock_M_tiles x subblock_N_tiles C tiles, the amount DST holds) at a time.
//
// Between K iterations the running sums have to leave DST. Default: they are packed into the C_partials
// ring and copied back into DST at the start of the next iteration (spill / reload). With PACKER_L1_ACC the
// packer adds DST onto the partials already in L1 instead, so only the last iteration reloads. The last K
// iteration packs the finished subblocks into the MN_chunk ring for the writer.
//
// Loop order matches the reader and the writer: batch, MN chunk, K iteration, subblocks (m_tile, n_tile)
// row-major over the chunk, k_tile within the iteration. Runtime args: num_MN_chunks. Compile-time args:
// batch_size, K_iteration_tiles, num_K_iterations, MN_chunk_M_tiles, MN_chunk_N_tiles, subblock_M_tiles,
// subblock_N_tiles. Defines: FP32_DEST_ACC_EN, PACKER_L1_ACC.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

// Copies one subblock of partial sums from the C_partials ring back into DST so the next K iteration keeps
// accumulating onto it. The unpacker is switched to the partials format for the copy and back to B's
// format afterwards; the matmul MOP must be re-initialised after any copy_init.
FORCE_INLINE void reload_partials_into_dst(
    uint32_t subblock_tiles, uint32_t subblock_M_tiles, uint32_t subblock_N_tiles, uint32_t K_iteration_tiles) {
    DataflowBuffer C_partials(dfb::C_partials);
    reconfig_data_format_srca(dfb::B_slice, dfb::C_partials);
    copy_init(dfb::C_partials);
    C_partials.wait_front(subblock_tiles);
    copy_block(dfb::C_partials, /*start_in_tile_index=*/0, /*start_dst_tile_index=*/0, subblock_tiles);
    C_partials.pop_front(subblock_tiles);
    reconfig_data_format_srca(dfb::C_partials, dfb::B_slice);
    // Metalium API: matmul_block_init(A, B, transpose, ct_dim = N tiles, rt_dim = M tiles, kt_dim = K tiles).
    matmul_block_init(
        dfb::A_slice, dfb::B_slice, /*transpose=*/0, subblock_N_tiles, subblock_M_tiles, K_iteration_tiles);
}

void kernel_main() {
    const uint32_t num_MN_chunks = get_arg(args::num_MN_chunks);  // this core's chunks, per batch

    constexpr uint32_t batch_size = get_arg(args::batch_size);
    constexpr uint32_t K_iteration_tiles = get_arg(args::K_iteration_tiles);
    constexpr uint32_t num_K_iterations = get_arg(args::num_K_iterations);
    constexpr uint32_t MN_chunk_M_tiles = get_arg(args::MN_chunk_M_tiles);
    constexpr uint32_t MN_chunk_N_tiles = get_arg(args::MN_chunk_N_tiles);
    constexpr uint32_t subblock_M_tiles = get_arg(args::subblock_M_tiles);
    constexpr uint32_t subblock_N_tiles = get_arg(args::subblock_N_tiles);

    constexpr uint32_t A_slice_tiles = MN_chunk_M_tiles * K_iteration_tiles;
    constexpr uint32_t B_slice_tiles = K_iteration_tiles * MN_chunk_N_tiles;
    constexpr uint32_t MN_chunk_tiles = MN_chunk_M_tiles * MN_chunk_N_tiles;
    constexpr uint32_t subblock_tiles = subblock_M_tiles * subblock_N_tiles;  // what DST holds
    // Partial sums exist only when K is split into more than one iteration.
    constexpr bool accumulate_across_K_iterations = num_K_iterations > 1;

    DataflowBuffer A_slice(dfb::A_slice);
    DataflowBuffer B_slice(dfb::B_slice);
    DataflowBuffer MN_chunk(dfb::MN_chunk);
    DataflowBuffer C_partials(dfb::C_partials);

    compute_kernel_hw_startup<SrcOrder::Reverse>(dfb::A_slice, dfb::B_slice, dfb::C_partials);
    matmul_block_init(
        dfb::A_slice, dfb::B_slice, /*transpose=*/0, subblock_N_tiles, subblock_M_tiles, K_iteration_tiles);

    for (uint32_t batch = 0; batch < batch_size; ++batch) {
        for (uint32_t MN_chunk_index = 0; MN_chunk_index < num_MN_chunks; ++MN_chunk_index) {
            if (batch > 0 || MN_chunk_index > 0) {
                // The previous chunk's last K iteration left the packer on MN_chunk's format. (The unpacker
                // needs no fix-up: reload_partials_into_dst already restores SrcA to B's format.)
                pack_reconfig_data_format(dfb::C_partials);
            }
            bool reload_partials = false;

            for (uint32_t K_iteration = 0; K_iteration < num_K_iterations; ++K_iteration) {
                const bool last_K_iteration = K_iteration == num_K_iterations - 1;
                A_slice.wait_front(A_slice_tiles);
                B_slice.wait_front(B_slice_tiles);

                // Slice layouts: A is [MN_chunk_M_tiles][K_iteration_tiles] row-major, B is
                // [K_iteration_tiles][MN_chunk_N_tiles]. Walk the chunk in subblocks: (m_tile, n_tile) is the
                // subblock's first tile within the chunk.
                for (uint32_t m_tile = 0; m_tile < MN_chunk_M_tiles; m_tile += subblock_M_tiles) {
                    const uint32_t A_subblock_first_tile = m_tile * K_iteration_tiles;  // A slice tile (m_tile, 0)
                    for (uint32_t n_tile = 0; n_tile < MN_chunk_N_tiles; n_tile += subblock_N_tiles) {
                        const uint32_t B_subblock_first_tile = n_tile;  // B slice tile (0, n_tile)
                        tile_regs_acquire();
                        if (reload_partials) {
                            reload_partials_into_dst(
                                subblock_tiles, subblock_M_tiles, subblock_N_tiles, K_iteration_tiles);
                        }

                        // Accumulate this subblock over the K iteration, one K tile per matmul_block call: each
                        // call multiplies A's subblock_M_tiles-tall column of tiles at k_tile by B's
                        // subblock_N_tiles-wide row of tiles at k_tile and adds the subblock_M_tiles x
                        // subblock_N_tiles products onto DST tiles 0..subblock_tiles-1. The LLK has no
                        // multi-K-tile call: kt_dim is only the row stride of the A slice
                        // ([MN_chunk_M_tiles][K_iteration_tiles] tiles), so the k loop lives here.
                        uint32_t A_tile = A_subblock_first_tile;
                        uint32_t B_tile = B_subblock_first_tile;
                        for (uint32_t k_tile = 0; k_tile < K_iteration_tiles; ++k_tile) {
                            matmul_block(
                                dfb::A_slice,
                                dfb::B_slice,
                                A_tile,
                                B_tile,
                                /*idst=*/0,
                                /*transpose=*/0,
                                subblock_N_tiles,
                                subblock_M_tiles,
                                K_iteration_tiles);
                            A_tile += 1;                 // next K tile along the A slice row
                            B_tile += MN_chunk_N_tiles;  // next K row of the B slice
                        }
                        tile_regs_commit();

                        if (last_K_iteration) {
                            MN_chunk.reserve_back(subblock_tiles);
                            tile_regs_wait();
#if defined FP32_DEST_ACC_EN or defined PACKER_L1_ACC
                            pack_reconfig_data_format(dfb::MN_chunk);
#endif
#ifdef PACKER_L1_ACC
                            pack_reconfig_l1_acc(0);
#endif
                            pack_block(/*ifrom_dst=*/0, dfb::MN_chunk, subblock_tiles);
                            tile_regs_release();
                            MN_chunk.push_back(subblock_tiles);
                        } else {
                            C_partials.reserve_back(subblock_tiles);
                            tile_regs_wait();
#ifdef PACKER_L1_ACC
                            // Iteration 0 overwrites the partials; from iteration 1 on the packer adds DST onto L1.
                            if (K_iteration == 0) {
                                pack_reconfig_l1_acc(0);
                            } else if (K_iteration == 1) {
                                pack_reconfig_l1_acc(1);
                            }
#endif
                            pack_block(/*ifrom_dst=*/0, dfb::C_partials, subblock_tiles);
                            tile_regs_release();
                            C_partials.push_back(subblock_tiles);
                        }
                    }
                }

#ifdef PACKER_L1_ACC
                // The packer accumulated in place, so the entries pushed this iteration carry nothing new:
                // pop them without reading (the ring holds exactly one chunk, so the next iteration lands on
                // the same L1 addresses). The second-to-last iteration's entries stay: the last one reloads
                // them. dummy_unpack orders the pop after the wait on Quasar; it is a no-op elsewhere.
                if (K_iteration + 2 < num_K_iterations) {
                    for (uint32_t popped = 0; popped < MN_chunk_tiles; popped += subblock_tiles) {
                        C_partials.wait_front(subblock_tiles);
                        dummy_unpack(dfb::C_partials);
                        C_partials.pop_front(subblock_tiles);
                    }
                }
                if (K_iteration + 2 == num_K_iterations) {
                    reload_partials = true;
                }
#else
                if constexpr (accumulate_across_K_iterations) {
                    reload_partials = true;
                }
#endif

                A_slice.pop_front(A_slice_tiles);
                B_slice.pop_front(B_slice_tiles);
            }
        }
    }
}
