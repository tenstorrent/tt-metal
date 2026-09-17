// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Unified matmul compute kernel: C = A x B for one cluster, as a classic blocked GEMM.
//
// GEMM view, all sizes in 32x32 tiles: C[M x N] = A[M x K] x B[K x N]. The unit of work is one C block
// (per_core_M x per_core_N tiles) of one batch; this cluster owns a contiguous run of them and this kernel
// does not care which. For every work item it accumulates over K in steps of K_step_tiles: the reader
// delivers one A panel ([per_core_M][K_step_tiles] tiles) and one B panel ([K_step_tiles][per_core_N]
// tiles) per step, and the MATH engine multiplies them one DST-sized subblock (subblock_M_tiles x
// subblock_N_tiles tiles) at a time.
//
// Between K steps the running sums have to leave DST. Default: they are packed into the C_partials ring
// and copied back into DST at the start of the next step (spill / reload). With PACKER_L1_ACC the packer
// adds DST onto the partials already in L1 instead, so only the last step reloads. The last K step packs
// the finished subblocks into the C_block ring for the writer.
//
// Loop order matches the reader and the writer: work item, K step, subblock row, subblock column.
// Runtime args: num_work_items. Compile-time args: K_step_tiles, num_K_steps, per_core_M, per_core_N,
// subblock_M_tiles, subblock_N_tiles. Defines: FP32_DEST_ACC_EN, PACKER_L1_ACC.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

// Copies one subblock of partial sums from the C_partials ring back into DST so the next K step keeps
// accumulating onto it. The unpacker is switched to the partials format for the copy and back to B's
// format afterwards; the matmul MOP must be re-initialised after any copy_init.
FORCE_INLINE void reload_partials_into_dst(
    uint32_t subblock_tiles, uint32_t subblock_M_tiles, uint32_t subblock_N_tiles, uint32_t K_step_tiles) {
    DataflowBuffer C_partials(dfb::C_partials);
    reconfig_data_format_srca(dfb::B_panel, dfb::C_partials);
    copy_init(dfb::C_partials);
    C_partials.wait_front(subblock_tiles);
    copy_block(dfb::C_partials, /*start_in_tile_index=*/0, /*start_dst_tile_index=*/0, subblock_tiles);
    C_partials.pop_front(subblock_tiles);
    reconfig_data_format_srca(dfb::C_partials, dfb::B_panel);
    // matmul_block_init(A, B, transpose, ct_dim = N tiles, rt_dim = M tiles, kt_dim = K tiles)
    matmul_block_init(dfb::A_panel, dfb::B_panel, /*transpose=*/0, subblock_N_tiles, subblock_M_tiles, K_step_tiles);
}

void kernel_main() {
    const uint32_t num_work_items = get_arg(args::num_work_items);

    constexpr uint32_t K_step_tiles = get_arg(args::K_step_tiles);
    constexpr uint32_t num_K_steps = get_arg(args::num_K_steps);
    constexpr uint32_t per_core_M = get_arg(args::per_core_M);
    constexpr uint32_t per_core_N = get_arg(args::per_core_N);
    constexpr uint32_t subblock_M_tiles = get_arg(args::subblock_M_tiles);
    constexpr uint32_t subblock_N_tiles = get_arg(args::subblock_N_tiles);

    constexpr uint32_t A_panel_tiles = per_core_M * K_step_tiles;
    constexpr uint32_t B_panel_tiles = K_step_tiles * per_core_N;
    constexpr uint32_t C_block_tiles = per_core_M * per_core_N;
    constexpr uint32_t subblock_tiles = subblock_M_tiles * subblock_N_tiles;
    constexpr uint32_t num_subblock_rows = per_core_M / subblock_M_tiles;
    constexpr uint32_t num_subblock_columns = per_core_N / subblock_N_tiles;
    // Partial sums exist only when K is split into more than one step.
    constexpr bool accumulate_across_K_steps = num_K_steps > 1;

    DataflowBuffer A_panel(dfb::A_panel);
    DataflowBuffer B_panel(dfb::B_panel);
    DataflowBuffer C_block(dfb::C_block);
    DataflowBuffer C_partials(dfb::C_partials);

    compute_kernel_hw_startup<SrcOrder::Reverse>(dfb::A_panel, dfb::B_panel, dfb::C_partials);
    matmul_block_init(dfb::A_panel, dfb::B_panel, /*transpose=*/0, subblock_N_tiles, subblock_M_tiles, K_step_tiles);

    for (uint32_t work_item = 0; work_item < num_work_items; ++work_item) {
        {
            if (work_item > 0) {
                // The previous item's last K step left the packer on C_block's format. (The unpacker needs
                // no fix-up: reload_partials_into_dst already restores SrcA to B's format.)
                pack_reconfig_data_format(dfb::C_partials);
            }
            bool reload_partials = false;

            for (uint32_t K_step = 0; K_step < num_K_steps; ++K_step) {
                const bool last_K_step = K_step == num_K_steps - 1;
                A_panel.wait_front(A_panel_tiles);
                B_panel.wait_front(B_panel_tiles);

                // Panel layouts: A is [per_core_M][K_step_tiles] row-major, B is [K_step_tiles][per_core_N].
                uint32_t A_subblock_first_tile = 0;  // A panel tile (subblock_row * subblock_M_tiles, 0)
                for (uint32_t subblock_row = 0; subblock_row < num_subblock_rows; ++subblock_row) {
                    uint32_t B_subblock_first_tile = 0;  // B panel tile (0, subblock_column * subblock_N_tiles)
                    for (uint32_t subblock_column = 0; subblock_column < num_subblock_columns; ++subblock_column) {
                        tile_regs_acquire();
                        if (reload_partials) {
                            reload_partials_into_dst(subblock_tiles, subblock_M_tiles, subblock_N_tiles, K_step_tiles);
                        }

                        // Accumulate this subblock over the K step, one K tile per matmul_block call. The call
                        // multiplies subblock_M_tiles rows of the A panel (row stride K_step_tiles) by one row
                        // of the B panel (subblock_N_tiles wide) into DST tiles 0..subblock_tiles-1.
                        uint32_t A_tile = A_subblock_first_tile;
                        uint32_t B_tile = B_subblock_first_tile;
                        for (uint32_t k = 0; k < K_step_tiles; ++k) {
                            matmul_block(
                                dfb::A_panel,
                                dfb::B_panel,
                                A_tile,
                                B_tile,
                                /*idst=*/0,
                                /*transpose=*/0,
                                subblock_N_tiles,
                                subblock_M_tiles,
                                K_step_tiles);
                            A_tile += 1;           // next K tile along the A panel row
                            B_tile += per_core_N;  // next K row of the B panel
                        }
                        tile_regs_commit();

                        if (last_K_step) {
                            C_block.reserve_back(subblock_tiles);
                            tile_regs_wait();
#if defined FP32_DEST_ACC_EN or defined PACKER_L1_ACC
                            pack_reconfig_data_format(dfb::C_block);
#endif
#ifdef PACKER_L1_ACC
                            pack_reconfig_l1_acc(0);
#endif
                            pack_block(/*ifrom_dst=*/0, dfb::C_block, subblock_tiles);
                            tile_regs_release();
                            C_block.push_back(subblock_tiles);
                        } else {
                            C_partials.reserve_back(subblock_tiles);
                            tile_regs_wait();
#ifdef PACKER_L1_ACC
                            // K step 0 overwrites the partials; from step 1 on the packer adds DST onto L1.
                            if (K_step == 0) {
                                pack_reconfig_l1_acc(0);
                            } else if (K_step == 1) {
                                pack_reconfig_l1_acc(1);
                            }
#endif
                            pack_block(/*ifrom_dst=*/0, dfb::C_partials, subblock_tiles);
                            tile_regs_release();
                            C_partials.push_back(subblock_tiles);
                        }

                        B_subblock_first_tile += subblock_N_tiles;
                    }
                    A_subblock_first_tile += subblock_M_tiles * K_step_tiles;
                }

#ifdef PACKER_L1_ACC
                // The packer accumulated in place, so the entries pushed this step carry nothing new: pop
                // them without reading (the ring holds exactly one C block, so the next step lands on the
                // same L1 addresses). The second-to-last step's entries stay: the last step reloads them.
                // dummy_unpack orders the pop after the wait on Quasar; it is a no-op elsewhere.
                if (K_step + 2 < num_K_steps) {
                    for (uint32_t popped = 0; popped < C_block_tiles; popped += subblock_tiles) {
                        C_partials.wait_front(subblock_tiles);
                        dummy_unpack(dfb::C_partials);
                        C_partials.pop_front(subblock_tiles);
                    }
                }
                if (K_step + 2 == num_K_steps) {
                    reload_partials = true;
                }
#else
                if constexpr (accumulate_across_K_steps) {
                    reload_partials = true;
                }
#endif

                A_panel.pop_front(A_panel_tiles);
                B_panel.pop_front(B_panel_tiles);
            }
        }
    }
}
