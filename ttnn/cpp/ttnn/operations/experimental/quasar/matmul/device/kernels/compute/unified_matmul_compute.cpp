// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Unified matmul compute kernel: produces num_C_slices C slices per batch, position-agnostic. Per K
// chunk the reader delivers one A slice ([C_slice_M_tiles][K_chunk_tiles] tiles) and one B slice
// ([K_chunk_tiles][C_slice_N_tiles] tiles), multiplied one subblock (what DST holds) at a time.
//
// Between K chunks the running sums leave DST: packed into C_partials and reloaded at the start of the
// next K chunk (spill / reload), or with packer_l1_acc the packer adds DST onto the partials in L1 and
// only the last K chunk reloads. The last K chunk packs the finished subblocks into C_slice for the
// writer. Loop order matches the reader and the writer: batch, MN chunk, K chunk, subblocks row-major
// over the C slice, k_tile within the K chunk.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t num_C_slices = get_arg(args::num_C_slices);  // this core's C slices, per batch

    constexpr uint32_t batch_size = get_arg(args::batch_size);
    constexpr uint32_t K_chunk_tiles = get_arg(args::K_chunk_tiles);
    constexpr uint32_t num_K_chunks = get_arg(args::num_K_chunks);
    constexpr uint32_t C_slice_M_tiles = get_arg(args::C_slice_M_tiles);
    constexpr uint32_t C_slice_N_tiles = get_arg(args::C_slice_N_tiles);
    constexpr uint32_t subblock_M_tiles = get_arg(args::subblock_M_tiles);
    constexpr uint32_t subblock_N_tiles = get_arg(args::subblock_N_tiles);
    constexpr bool packer_l1_acc = get_arg(args::packer_l1_acc) != 0;
    constexpr bool partials_format_differs = get_arg(args::partials_format_differs) != 0;

    constexpr uint32_t A_slice_tiles = C_slice_M_tiles * K_chunk_tiles;
    constexpr uint32_t B_slice_tiles = K_chunk_tiles * C_slice_N_tiles;
    constexpr uint32_t C_slice_tiles = C_slice_M_tiles * C_slice_N_tiles;
    constexpr uint32_t subblock_tiles = subblock_M_tiles * subblock_N_tiles;  // what DST holds

    DataflowBuffer A_slice(dfb::A_slice);
    DataflowBuffer B_slice(dfb::B_slice);
    DataflowBuffer C_slice(dfb::C_slice);
    DataflowBuffer C_partials(dfb::C_partials);

    compute_kernel_hw_startup<SrcOrder::Reverse>(dfb::A_slice, dfb::B_slice, dfb::C_partials);
    matmul_block_init(dfb::A_slice, dfb::B_slice, /*transpose=*/0, subblock_N_tiles, subblock_M_tiles, K_chunk_tiles);

    for (uint32_t batch = 0; batch < batch_size; ++batch) {
        for (uint32_t MN_chunk = 0; MN_chunk < num_C_slices; ++MN_chunk) {
            for (uint32_t K_chunk = 0; K_chunk < num_K_chunks; ++K_chunk) {
                const bool last_K_chunk = K_chunk == num_K_chunks - 1;
                // Partials exist once a previous K chunk has packed them. Without packer L1 accumulation every
                // later K chunk reloads them; with it the packer has been accumulating in L1 and only the last
                // K chunk reloads the sum.
                const bool reload_partials = K_chunk > 0 && (last_K_chunk || !packer_l1_acc);
                // Every subblock of this K chunk is packed to the same DFB: the finished sum to C_slice on the
                // last K chunk, the running sum to C_partials before that.
                DataflowBuffer& pack_target = last_K_chunk ? C_slice : C_partials;
                const uint32_t pack_target_id = last_K_chunk ? uint32_t(dfb::C_slice) : uint32_t(dfb::C_partials);
                A_slice.wait_front(A_slice_tiles);
                B_slice.wait_front(B_slice_tiles);

                // Packer setup for this K chunk. The two DFBs may hold different formats; with packer L1
                // accumulation K chunk 0 overwrites the partials, later K chunks add DST onto them, and the
                // finished sum is packed without accumulation. (The reload below touches only the unpacker.)
                if constexpr (partials_format_differs) {
                    pack_reconfig_data_format(pack_target_id);
                }
                if constexpr (packer_l1_acc) {
                    pack_reconfig_l1_acc((!last_K_chunk && K_chunk > 0) ? 1 : 0);
                }

                // Slice layouts: A is [C_slice_M_tiles][K_chunk_tiles] row-major, B is
                // [K_chunk_tiles][C_slice_N_tiles]. Walk the C slice in subblocks: (m_tile, n_tile) is the
                // subblock's first tile within the C slice.
                for (uint32_t m_tile = 0; m_tile < C_slice_M_tiles; m_tile += subblock_M_tiles) {
                    const uint32_t A_subblock_first_tile = m_tile * K_chunk_tiles;  // A slice tile (m_tile, 0)
                    for (uint32_t n_tile = 0; n_tile < C_slice_N_tiles; n_tile += subblock_N_tiles) {
                        const uint32_t B_subblock_first_tile = n_tile;  // B slice tile (0, n_tile)
                        tile_regs_acquire();
                        if (reload_partials) {
                            // Copy this subblock's partial sums from the C_partials back into DST and keep
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
                        // ([C_slice_M_tiles][K_chunk_tiles] tiles), so the k loop lives here.
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
                            A_slice_tile += 1;                // next K tile along the A slice row
                            B_slice_tile += C_slice_N_tiles;  // next K row of the B slice
                        }
                        tile_regs_commit();

                        pack_target.reserve_back(subblock_tiles);
                        tile_regs_wait();
                        pack_block(/*ifrom_dst=*/0, pack_target_id, subblock_tiles);
                        tile_regs_release();
                        pack_target.push_back(subblock_tiles);
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
                        for (uint32_t popped = 0; popped < C_slice_tiles; popped += subblock_tiles) {
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
