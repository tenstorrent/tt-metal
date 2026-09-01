// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "api/dataflow/dataflow_buffer.h"
#include "api/compute/topk.h"

// stable_sort selects the LLK's stable bitonic network: equal values keep the lowest index, so ties
// are broken deterministically rather than by array position. It defaults to false so every existing
// caller (ttnn.topk, the deepseek grouped-gate/MoE kernels) keeps its current behaviour; only callers
// that explicitly instantiate with true opt in.
//
// fused selects the fused-key stable mode: right after the transposes, the u16 index tiles (DEST
// 2,3) are packed into the value words (DEST 0,1) as [bf16 value | u16 index'] sort keys with a
// sign-and-direction-conditional index complement (fused_largest = the op's GLOBAL order), and the
// plain UNSTABLE network then sorts the packed words. Indices stop travelling separately: the
// index-transposed CB is never touched (and need not exist), all index copy/pack traffic below
// disappears, and the packed intermediate CBs must be UInt32 (raw-bit transport — a float pack
// would denormal-flush 0x0000xxxx keys). Requires 32-bit DEST. Mutually exclusive with stable_sort
// (fused provides the same torch-stable tie order through the key itself).
template <bool stable_sort = false, bool fused = false, bool fused_largest = false>
void process_and_sort_tiles(
    std::uint32_t input_dfb_index,
    std::uint32_t index_dfb_index,
    std::uint32_t input_transposed_dfb_index,
    std::uint32_t index_transposed_dfb_index,
    std::uint32_t Wt,
    bool switch_dir,
    bool& ascending,
    int end_phase) {
    static_assert(!(fused && stable_sort), "fused and comparator-stable modes are mutually exclusive");
    DataflowBuffer input_dfb(input_dfb_index);
    DataflowBuffer index_dfb(index_dfb_index);
    DataflowBuffer input_transposed_dfb(input_transposed_dfb_index);
    DataflowBuffer index_transposed_dfb(index_transposed_dfb_index);

    input_transposed_dfb.reserve_back(Wt);
    if constexpr (!fused) {
        index_transposed_dfb.reserve_back(Wt);
    }

    // streaming in input and index tiles to transpose and bitonic local sort them, two tiles at a time
    for (std::uint32_t wt = 0; wt < Wt; wt += 2) {
        // local sort into k groups
        // for the last iteration, we only need to wait for 1 tile if Wt is odd, otherwise we wait for 2 tiles
        std::uint32_t tiles_to_wait = ((Wt % 2 != 0) && (wt + 2 > Wt)) ? 1 : 2;
        input_dfb.wait_front(tiles_to_wait);
        index_dfb.wait_front(tiles_to_wait);

        tile_regs_acquire();
        reconfig_data_format_srca(input_dfb_index);
        transpose_init(input_dfb_index);
        transpose_tile(input_dfb_index, 0, 0);
        if (tiles_to_wait == 2) {
            transpose_tile(input_dfb_index, 1, 1);
        }
        reconfig_data_format_srca(index_dfb_index);
        transpose_init(index_dfb_index);
        transpose_tile(index_dfb_index, 0, 2);
        if (tiles_to_wait == 2) {
            transpose_tile(index_dfb_index, 1, 3);
        }
        if constexpr (fused) {
            // Pack the slab into [bf16|u16] keys once, with the GLOBAL polarity; DEST 2,3 are dead
            // afterwards and the network below runs unstable on the packed words.
            ckernel::topk_fuse_tile<fused_largest>(0);
        }
        // llk_topk_sort -> inplace
        // (stable) tie-break polarity is set once per kernel from the global `largest`;
        // it must not follow `ascending`, which alternates for bitonic sequence building.
        ckernel::topk_local_sort<stable_sort, DST_ACCUM_MODE, fused>(0, (int)ascending, end_phase);
        tile_regs_commit();

        input_dfb.pop_front(tiles_to_wait);
        index_dfb.pop_front(tiles_to_wait);

        tile_regs_wait();
        // pack value (or packed-key) tiles into cb_intermed0
        pack_reconfig_data_format(input_transposed_dfb_index);
        pack_tile(0, input_transposed_dfb_index);
        if (tiles_to_wait == 2) {
            pack_tile(1, input_transposed_dfb_index);
        }
        if constexpr (!fused) {
            // pack index tiles into cb_intermed1
            pack_reconfig_data_format(index_transposed_dfb_index);
            pack_tile(2, index_transposed_dfb_index);
            if (tiles_to_wait == 2) {
                pack_tile(3, index_transposed_dfb_index);
            }
        }
        tile_regs_release();
        ascending = switch_dir ? !ascending : ascending;
    }

    input_transposed_dfb.push_back(Wt);
    if constexpr (!fused) {
        index_transposed_dfb.push_back(Wt);
    }
}

template <bool stable_sort = false, bool fused = false>
void process_tile_pair(
    std::uint32_t left_ind,
    std::uint32_t right_ind,
    std::uint32_t input_transposed_dfb_index,
    std::uint32_t index_transposed_dfb_index,
    std::uint32_t input_dest_start,
    std::uint32_t input_dest_end,
    std::uint32_t index_dest_start,
    std::uint32_t index_dest_end,
    bool ascending,
    std::uint32_t m_iter,
    std::uint32_t K,
    std::uint32_t logk,
    bool target_tiles_is_one) {
    tile_regs_acquire();

    if constexpr (fused) {
        // With fused keys there are no index tiles.
        reconfig_data_format_srca(input_transposed_dfb_index);
    } else {
        // Without fused keys, the previous iteration left srca configured for index unpacking.
        reconfig_data_format_srca(index_transposed_dfb_index, input_transposed_dfb_index);
    }
    copy_init(input_transposed_dfb_index);
    copy_tile(input_transposed_dfb_index, left_ind, input_dest_start);
    if (!target_tiles_is_one) {
        copy_tile(input_transposed_dfb_index, right_ind, input_dest_end);
    }

    if constexpr (!fused) {
        // unpack indices into dest
        reconfig_data_format_srca(input_transposed_dfb_index, index_transposed_dfb_index);
        copy_init(index_transposed_dfb_index);
        copy_tile(index_transposed_dfb_index, left_ind, index_dest_start);
        if (!target_tiles_is_one) {
            copy_tile(index_transposed_dfb_index, right_ind, index_dest_end);
        }
    }

    // merge values - move larger 32 values into 0th dest and lower 32 values into 1st dest
    // sort within the larger 32 values
    // (stable) tie-break polarity is set once per kernel from the global `largest`; `ascending`
    // here may be flipped per core (direction_init) to alternate output direction for the final
    // cross-core bitonic merge, and the tie polarity must not flip with it.
    ckernel::topk_rebuild<stable_sort, DST_ACCUM_MODE, fused>(
        0, (std::uint32_t)ascending, m_iter, K, logk, target_tiles_is_one);

    tile_regs_commit();
    tile_regs_wait();
    // pack value tiles in-place in the single-buffered cb_intermed0, we only need the upper 32
    // values for topk, which was in input_dest_start
    pack_reconfig_data_format(input_transposed_dfb_index);
    pack_tile<true>(input_dest_start, input_transposed_dfb_index, left_ind);
    if (!target_tiles_is_one) {
        pack_tile<true>(input_dest_end, input_transposed_dfb_index, right_ind);
    }

    if constexpr (!fused) {
        // pack index tiles in-place in the single-buffered cb_intermed1, we only need the upper 32
        // values for topk, which was in index_dest_start
        pack_reconfig_data_format(index_transposed_dfb_index);
        pack_tile<true>(index_dest_start, index_transposed_dfb_index, left_ind);
        if (!target_tiles_is_one) {
            pack_tile<true>(index_dest_end, index_transposed_dfb_index, right_ind);
        }
    }
    tile_regs_release();
}

template <bool stable_sort = false, bool fused = false>
void process_tiles(
    std::uint32_t m_iter,
    std::uint32_t K,
    std::uint32_t Wt,
    std::uint32_t num_k_sequences,
    std::uint32_t tiles_per_seq,
    std::uint32_t input_transposed_dfb_index,
    std::uint32_t index_transposed_dfb_index,
    std::uint32_t input_dest_start,
    std::uint32_t input_dest_end,
    std::uint32_t index_dest_start,
    std::uint32_t index_dest_end,
    bool largest,
    int seq_per_2tiles) {
    std::uint32_t dist = ((1 << m_iter) * K) >> 5;
    for (std::uint32_t i = 0; i < num_k_sequences; i += seq_per_2tiles) {
        for (std::uint32_t t = 0; t < tiles_per_seq; t++) {
            std::uint32_t left_tile_id = ((i * (1 << m_iter) * K) >> 5) + t;
            std::uint32_t right_tile_id = left_tile_id + dist;
            if (left_tile_id == right_tile_id) {
                right_tile_id = left_tile_id + 1;
            }

            if (left_tile_id >= Wt || right_tile_id >= Wt) {
                break;
            }

            tile_regs_acquire();

            if constexpr (fused) {
                // With fused keys there are no index tiles.
                reconfig_data_format_srca(input_transposed_dfb_index);
            } else {
                reconfig_data_format_srca(index_transposed_dfb_index, input_transposed_dfb_index);
            }
            copy_init(input_transposed_dfb_index);
            copy_tile(input_transposed_dfb_index, left_tile_id, input_dest_start);
            copy_tile(input_transposed_dfb_index, right_tile_id, input_dest_end);

            if constexpr (!fused) {
                // unpack indices into dest
                reconfig_data_format_srca(input_transposed_dfb_index, index_transposed_dfb_index);
                copy_init(index_transposed_dfb_index);
                copy_tile(index_transposed_dfb_index, left_tile_id, index_dest_start);
                copy_tile(index_transposed_dfb_index, right_tile_id, index_dest_end);
            }

            // merge values - move larger 32 values into 0th dest and lower 32 values into 1st dest
            if (largest) {
                ckernel::topk_merge<false, stable_sort, DST_ACCUM_MODE, fused>(0, m_iter, K);
            } else {
                ckernel::topk_merge<true, stable_sort, DST_ACCUM_MODE, fused>(0, m_iter, K);
            }

            tile_regs_commit();
            tile_regs_wait();
            // pack value tiles in-place in the single-buffered cb_intermed0, we only need the upper 32 values
            // for topk, which was in input_dest_start
            pack_reconfig_data_format(input_transposed_dfb_index);
            pack_tile<true>(input_dest_start, input_transposed_dfb_index, left_tile_id);
            pack_tile<true>(input_dest_end, input_transposed_dfb_index, right_tile_id);

            if constexpr (!fused) {
                // pack index tiles in-place in the single-buffered cb_intermed1, we only need the upper 32 values
                // for topk, which was in index_dest_start
                pack_reconfig_data_format(index_transposed_dfb_index);
                pack_tile<true>(index_dest_start, index_transposed_dfb_index, left_tile_id);
                pack_tile<true>(index_dest_end, index_transposed_dfb_index, right_tile_id);
            }
            tile_regs_release();
        }
    }
}

template <bool stable_sort = false, bool fused = false>
void process_iteration(
    std::uint32_t m_iter,
    std::uint32_t K,
    std::uint32_t Wt,
    std::uint32_t& num_k_sequences,
    std::uint32_t tiles_per_seq,
    std::uint32_t input_transposed_dfb_index,
    std::uint32_t index_transposed_dfb_index,
    std::uint32_t input_dest_start,
    std::uint32_t input_dest_end,
    std::uint32_t index_dest_start,
    std::uint32_t index_dest_end,
    bool largest,
    bool switch_dir,
    std::uint32_t logk,
    int& seq_per_2tiles,
    bool largest_param) {
    DataflowBuffer input_transposed_dfb(input_transposed_dfb_index);
    DataflowBuffer index_transposed_dfb(index_transposed_dfb_index);

    input_transposed_dfb.wait_front(Wt);
    if constexpr (!fused) {
        index_transposed_dfb.wait_front(Wt);
    }

    process_tiles<stable_sort, fused>(
        m_iter,
        K,
        Wt,
        num_k_sequences,
        tiles_per_seq,
        input_transposed_dfb_index,
        index_transposed_dfb_index,
        input_dest_start,
        input_dest_end,
        index_dest_start,
        index_dest_end,
        largest_param,
        seq_per_2tiles);

    input_transposed_dfb.reserve_back(Wt);
    input_transposed_dfb.pop_front(Wt);
    input_transposed_dfb.push_back(Wt);
    if constexpr (!fused) {
        index_transposed_dfb.reserve_back(Wt);
        index_transposed_dfb.pop_front(Wt);
        index_transposed_dfb.push_back(Wt);
    }

    // we have decreased our search space by half
    num_k_sequences = num_k_sequences >> 1;
    int target_tiles = (Wt == 1 || ((num_k_sequences == 1) && (tiles_per_seq == 1))) ? 1 : 2;

    int sel_tile_id[2];
    int sel_tile_id_ptr = 0;
    seq_per_2tiles = (seq_per_2tiles == 2) ? 2 : seq_per_2tiles >> 1;
    bool ascending = !largest;

    input_transposed_dfb.wait_front(Wt);
    if constexpr (!fused) {
        index_transposed_dfb.wait_front(Wt);
    }

    for (std::uint32_t idx = 0; idx < num_k_sequences; idx += (seq_per_2tiles >> 1)) {
        for (std::uint32_t t = 0; t < tiles_per_seq; t++) {
            std::uint32_t left_ind = ((idx * (1 << (m_iter + 1)) * K) >> 5) + t;
            if (left_ind >= Wt) {
                break;
            }
            sel_tile_id[sel_tile_id_ptr] = left_ind;
            sel_tile_id_ptr++;
            if (sel_tile_id_ptr == target_tiles) {
                process_tile_pair<stable_sort, fused>(
                    sel_tile_id[0],
                    sel_tile_id[1],
                    input_transposed_dfb_index,
                    index_transposed_dfb_index,
                    input_dest_start,
                    input_dest_end,
                    index_dest_start,
                    index_dest_end,
                    ascending,
                    m_iter,
                    K,
                    logk,
                    target_tiles == 1);
                sel_tile_id_ptr = 0;
                ascending = switch_dir ? !ascending : ascending;
            }
        }
    }

    input_transposed_dfb.reserve_back(Wt);
    input_transposed_dfb.pop_front(Wt);
    input_transposed_dfb.push_back(Wt);
    if constexpr (!fused) {
        index_transposed_dfb.reserve_back(Wt);
        index_transposed_dfb.pop_front(Wt);
        index_transposed_dfb.push_back(Wt);
    }
}

void transpose_and_pack(
    std::uint32_t transposed_dfb_index, std::uint32_t dest_dfb_index, std::uint32_t Kt, std::uint32_t Wt) {
    DataflowBuffer transposed_dfb(transposed_dfb_index);
    DataflowBuffer dest_dfb(dest_dfb_index);

    reconfig_data_format_srca(transposed_dfb_index);
    transpose_init(transposed_dfb_index);
    // Pack using the DESTINATION CB format: transposed_dfb may be bf16 (higher-precision
    // intermediate) while dest_dfb is the original bfp8/bfp4 output format.
    pack_reconfig_data_format(dest_dfb_index);

    transposed_dfb.wait_front(Kt);
    for (std::uint32_t i = 0; i < Kt; ++i) {
        tile_regs_acquire();
        transpose_tile(transposed_dfb_index, i, 0);
        tile_regs_commit();

        dest_dfb.reserve_back(1);

        tile_regs_wait();
        pack_tile(0, dest_dfb_index);
        tile_regs_release();

        dest_dfb.push_back(1);
    }
    transposed_dfb.wait_front(Wt);
    transposed_dfb.pop_front(Wt);
}

// Fused-key final extraction. The packed key tiles must not be transposed as 32-bit words (the
// transpose datapath goes through the 16-bit source registers), so the split happens first:
// defuse each packed tile in place via a plain raw copy into DEST — values [bf16|0x0000] at DEST 0
// (the following Float32->bf16 pack is exact), u16 indices at DEST 2 in the packer-visible high
// half (mode-9 store) — and stage both halves in 16-bit CBs still in column layout. The transpose
// back to row layout then runs on plain 16-bit tiles through the proven path. The transposed u16
// index datums land in the LOW half of the 32-bit DEST words while the packer reads the high half,
// so they are moved up (strip + mode-9) before the final pack. largest must match the fuse's
// polarity.
template <bool largest>
void defuse_and_pack_outputs(
    std::uint32_t packed_dfb_index,
    std::uint32_t staging_values_dfb_index,
    std::uint32_t staging_indices_dfb_index,
    std::uint32_t values_dfb_index,
    std::uint32_t indices_dfb_index,
    std::uint32_t Kt,
    std::uint32_t Wt) {
    DataflowBuffer packed_dfb(packed_dfb_index);
    DataflowBuffer staging_values_dfb(staging_values_dfb_index);
    DataflowBuffer staging_indices_dfb(staging_indices_dfb_index);
    DataflowBuffer indices_dfb(indices_dfb_index);

    // Split the top Kt packed key tiles into staged value/index tiles (column layout).
    reconfig_data_format_srca(packed_dfb_index);
    copy_init(packed_dfb_index);

    packed_dfb.wait_front(Kt);
    staging_values_dfb.reserve_back(Kt);
    staging_indices_dfb.reserve_back(Kt);
    for (std::uint32_t i = 0; i < Kt; ++i) {
        tile_regs_acquire();
        copy_tile(packed_dfb_index, i, 0);
        ckernel::topk_defuse_tile<largest>(0, 1);
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(staging_values_dfb_index);
        pack_tile(0, staging_values_dfb_index);
        pack_reconfig_data_format(staging_indices_dfb_index);
        pack_tile(2, staging_indices_dfb_index);
        tile_regs_release();
    }
    staging_values_dfb.push_back(Kt);
    staging_indices_dfb.push_back(Kt);
    packed_dfb.wait_front(Wt);
    packed_dfb.pop_front(Wt);

    // Values: the standard 16-bit transpose back to row layout.
    transpose_and_pack(staging_values_dfb_index, values_dfb_index, Kt, Kt);

    // Indices: same transpose, plus the low->high move before packing.
    reconfig_data_format_srca(staging_indices_dfb_index);
    transpose_init(staging_indices_dfb_index);
    pack_reconfig_data_format(indices_dfb_index);
    staging_indices_dfb.wait_front(Kt);
    for (std::uint32_t i = 0; i < Kt; ++i) {
        tile_regs_acquire();
        transpose_tile(staging_indices_dfb_index, i, 0);
        ckernel::topk_uint16_move_dest_tile_to_pack_half(0);
        tile_regs_commit();

        indices_dfb.reserve_back(1);

        tile_regs_wait();
        pack_tile(0, indices_dfb_index);
        tile_regs_release();

        indices_dfb.push_back(1);
    }
    staging_indices_dfb.pop_front(Kt);
}
