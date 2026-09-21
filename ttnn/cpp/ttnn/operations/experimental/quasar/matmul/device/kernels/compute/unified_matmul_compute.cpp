// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Unified matmul compute kernel: C = A x B for one cluster, as a classic blocked GEMM.
//
// GEMM view, all sizes in 32x32 tiles: C[M x N] = A[M x K] x B[K x N], batch_size times. An MN chunk is
// MN_chunk_M_tiles x MN_chunk_N_tiles tiles of C, the L1-fittable piece of the output region this cluster
// owns; normally the region is one chunk. This kernel produces num_MN_chunks chunks for every batch and does
// not care where in C they sit. For every chunk it accumulates over K, K_chunk_tiles per K chunk: the reader
// delivers one A slice ([MN_chunk_M_tiles][K_chunk_tiles] tiles) and one B slice
// ([K_chunk_tiles][MN_chunk_N_tiles] tiles) per K chunk, and the MATH engine multiplies them one subblock
// (subblock_M_tiles x subblock_N_tiles C tiles, the amount DST holds) at a time.
//
// Between K chunks the running sums have to leave DST. Default: they are packed into the C_partials ring
// and copied back into DST at the start of the next K chunk (spill / reload). With packer_l1_acc the packer
// adds DST onto the partials already in L1 instead, so only the last K chunk reloads. The last K chunk packs
// the finished subblocks into the MN_chunk ring for the writer.
//
// Loop order matches the reader and the writer: batch, MN chunk, K chunk, subblocks (m_tile, n_tile)
// row-major over the chunk, k_tile within the K chunk. Runtime args: num_MN_chunks. Compile-time args:
// batch_size, K_chunk_tiles, num_K_chunks, MN_chunk_M_tiles, MN_chunk_N_tiles, subblock_M_tiles,
// subblock_N_tiles, packer_l1_acc, partials_format_differs (C_partials and MN_chunk hold different formats,
// so the packer must be reconfigured when switching between them).

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t num_MN_chunks = get_arg(args::num_MN_chunks);  // this core's chunks, per batch

    constexpr uint32_t batch_size = get_arg(args::batch_size);
    constexpr uint32_t K_chunk_tiles = get_arg(args::K_chunk_tiles);
    constexpr uint32_t num_K_chunks = get_arg(args::num_K_chunks);
    constexpr uint32_t MN_chunk_M_tiles = get_arg(args::MN_chunk_M_tiles);
    constexpr uint32_t MN_chunk_N_tiles = get_arg(args::MN_chunk_N_tiles);
    constexpr uint32_t subblock_M_tiles = get_arg(args::subblock_M_tiles);
    constexpr uint32_t subblock_N_tiles = get_arg(args::subblock_N_tiles);
    constexpr bool packer_l1_acc = get_arg(args::packer_l1_acc) != 0;
    constexpr bool partials_format_differs = get_arg(args::partials_format_differs) != 0;

    constexpr uint32_t A_slice_tiles = MN_chunk_M_tiles * K_chunk_tiles;
    constexpr uint32_t B_slice_tiles = K_chunk_tiles * MN_chunk_N_tiles;
    constexpr uint32_t MN_chunk_tiles = MN_chunk_M_tiles * MN_chunk_N_tiles;
    constexpr uint32_t subblock_tiles = subblock_M_tiles * subblock_N_tiles;  // what DST holds

    DataflowBuffer A_slice(dfb::A_slice);
    DataflowBuffer B_slice(dfb::B_slice);
    DataflowBuffer MN_chunk(dfb::MN_chunk);
    DataflowBuffer C_partials(dfb::C_partials);

    compute_kernel_hw_startup<SrcOrder::Reverse>(dfb::A_slice, dfb::B_slice, dfb::C_partials);
    matmul_block_init(dfb::A_slice, dfb::B_slice, /*transpose=*/0, subblock_N_tiles, subblock_M_tiles, K_chunk_tiles);

    for (uint32_t batch = 0; batch < batch_size; ++batch) {
        for (uint32_t MN_chunk_index = 0; MN_chunk_index < num_MN_chunks; ++MN_chunk_index) {
            if constexpr (partials_format_differs && num_K_chunks > 1) {
                if (batch > 0 || MN_chunk_index > 0) {
                    // The previous chunk's last K chunk left the packer on MN_chunk's format. (The unpacker needs
                    // no fix-up: the partials reload already restores SrcA to B's format.)
                    pack_reconfig_data_format(dfb::C_partials);
                }
            }

            for (uint32_t K_chunk = 0; K_chunk < num_K_chunks; ++K_chunk) {
                const bool last_K_chunk = K_chunk == num_K_chunks - 1;
                // Partials exist once a previous K chunk has packed them. Without packer L1 accumulation every
                // later K chunk reloads them; with it the packer has been accumulating in L1 and only the last
                // K chunk reloads the sum.
                const bool reload_partials = K_chunk > 0 && (last_K_chunk || !packer_l1_acc);
                A_slice.wait_front(A_slice_tiles);
                B_slice.wait_front(B_slice_tiles);

                // Slice layouts: A is [MN_chunk_M_tiles][K_chunk_tiles] row-major, B is
                // [K_chunk_tiles][MN_chunk_N_tiles]. Walk the chunk in subblocks: (m_tile, n_tile) is the
                // subblock's first tile within the chunk.
                for (uint32_t m_tile = 0; m_tile < MN_chunk_M_tiles; m_tile += subblock_M_tiles) {
                    const uint32_t A_subblock_first_tile = m_tile * K_chunk_tiles;  // A slice tile (m_tile, 0)
                    for (uint32_t n_tile = 0; n_tile < MN_chunk_N_tiles; n_tile += subblock_N_tiles) {
                        const uint32_t B_subblock_first_tile = n_tile;  // B slice tile (0, n_tile)
                        tile_regs_acquire();
                        if (reload_partials) {
                            // Copy this subblock's partial sums from the C_partials ring back into DST and keep
                            // accumulating onto them. The unpacker is switched to the partials format for the
                            // copy and back to B's format afterwards; the matmul MOP must be re-initialised
                            // after any copy_init.
                            reconfig_data_format_srca(dfb::B_slice, dfb::C_partials);
                            copy_init(dfb::C_partials);
                            C_partials.wait_front(subblock_tiles);
                            copy_block(
                                dfb::C_partials, /*start_in_tile_index=*/0, /*start_dst_tile_index=*/0, subblock_tiles);
                            C_partials.pop_front(subblock_tiles);
                            reconfig_data_format_srca(dfb::C_partials, dfb::B_slice);
                            matmul_block_init(
                                dfb::A_slice,
                                dfb::B_slice,
                                /*transpose=*/0,
                                subblock_N_tiles,
                                subblock_M_tiles,
                                K_chunk_tiles);
                        }

                        // Accumulate this subblock over the K chunk, one K tile per matmul_block call: each call
                        // multiplies A's subblock_M_tiles-tall column of tiles at k_tile by B's
                        // subblock_N_tiles-wide row of tiles at k_tile and adds the subblock_M_tiles x
                        // subblock_N_tiles products onto DST tiles 0..subblock_tiles-1. The LLK has no
                        // multi-K-tile call: kt_dim is only the row stride of the A slice
                        // ([MN_chunk_M_tiles][K_chunk_tiles] tiles), so the k loop lives here.
                        uint32_t A_tile = A_subblock_first_tile;
                        uint32_t B_tile = B_subblock_first_tile;
                        for (uint32_t k_tile = 0; k_tile < K_chunk_tiles; ++k_tile) {
                            matmul_block(
                                dfb::A_slice,
                                dfb::B_slice,
                                A_tile,
                                B_tile,
                                /*idst=*/0,
                                /*transpose=*/0,
                                subblock_N_tiles,
                                subblock_M_tiles,
                                K_chunk_tiles);
                            A_tile += 1;                 // next K tile along the A slice row
                            B_tile += MN_chunk_N_tiles;  // next K row of the B slice
                        }
                        tile_regs_commit();

                        if (last_K_chunk) {
                            MN_chunk.reserve_back(subblock_tiles);
                            tile_regs_wait();
                            if constexpr (partials_format_differs) {
                                pack_reconfig_data_format(dfb::MN_chunk);
                            }
                            if constexpr (packer_l1_acc) {
                                pack_reconfig_l1_acc(0);
                            }
                            pack_block(/*ifrom_dst=*/0, dfb::MN_chunk, subblock_tiles);
                            tile_regs_release();
                            MN_chunk.push_back(subblock_tiles);
                        } else {
                            C_partials.reserve_back(subblock_tiles);
                            tile_regs_wait();
                            if constexpr (packer_l1_acc) {
                                // K chunk 0 overwrites the partials; from K chunk 1 on the packer adds DST onto L1.
                                pack_reconfig_l1_acc(K_chunk > 0 ? 1 : 0);
                            }
                            pack_block(/*ifrom_dst=*/0, dfb::C_partials, subblock_tiles);
                            tile_regs_release();
                            C_partials.push_back(subblock_tiles);
                        }
                    }
                }

                if constexpr (packer_l1_acc) {
                    // The entries pushed this K chunk are only credits (the sums live in L1): pop them so the next
                    // K chunk lands on the same addresses. Two exceptions: the last K chunk pushed nothing here, and
                    // the second-to-last K chunk's entries are what the last K chunk reloads.
                    const bool second_to_last_K_chunk = K_chunk + 2 == num_K_chunks;
                    if (!last_K_chunk && !second_to_last_K_chunk) {
                        // Pop without reading: dummy_unpack orders the pop after the wait on Quasar (a no-op
                        // elsewhere).
                        for (uint32_t popped = 0; popped < MN_chunk_tiles; popped += subblock_tiles) {
                            C_partials.wait_front(subblock_tiles);
                            dummy_unpack(dfb::C_partials);
                            C_partials.pop_front(subblock_tiles);
                        }
                    }
                }

                A_slice.pop_front(A_slice_tiles);
                B_slice.pop_front(B_slice_tiles);
            }
        }
    }
}
