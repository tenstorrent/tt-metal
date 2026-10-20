// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/operations/experimental/kda/device/kernels/chronology.hpp"
//
// Phase B (scan) compute kernel: the sequential-over-chunk recurrence for one
// head.

#include <cstdint>

#include "api/compute/bcast.h"
#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/matmul.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/operations/experimental/kda/device/kernels/compute/matmul_subblock.hpp"

enum class ElementwiseOperation { ADD, SUBTRACT };
enum class ChunkInputPolicy { RETAIN, CONSUME };

template <uint32_t Mt, uint32_t Kt, uint32_t Nt, typename DFBA, typename DFBB, typename DFBOut>
FORCE_INLINE void matrix_multiply(DFBA& a, DFBB& b, DFBOut& output) {
    constexpr uint32_t subblock_columns = kda::MatmulSubblock<Mt, Nt>::columns;
    constexpr uint32_t subblock_rows = kda::MatmulSubblock<Mt, Nt>::rows;
    const uint32_t a_id = a.get_id();
    const uint32_t b_id = b.get_id();
    const uint32_t output_id = output.get_id();

    output.reserve_back(Mt * Nt);
    reconfig_data_format<SrcOrder::Reverse>(a_id, b_id);
    matmul_block_init(a_id, b_id, false, subblock_columns, subblock_rows, Kt);
    for (uint32_t row_start = 0; row_start < Mt; row_start += subblock_rows) {
        for (uint32_t column_start = 0; column_start < Nt; column_start += subblock_columns) {
            tile_regs_acquire();
            for (uint32_t k = 0; k < Kt; ++k) {
                const uint32_t b_index = k * Nt + column_start;
                matmul_block(a_id, b_id, row_start * Kt + k, b_index, 0, false, subblock_columns, subblock_rows, Kt);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t row = 0; row < subblock_rows; ++row) {
                for (uint32_t column = 0; column < subblock_columns; ++column) {
                    pack_tile(
                        row * subblock_columns + column, output_id, (row_start + row) * Nt + column_start + column);
                }
            }
            tile_regs_release();
        }
    }
    output.push_back(Mt * Nt);
}

// Inputs remain resident; PacketTiles specifies output publication granularity.
template <
    ElementwiseOperation Operation,
    uint32_t Count,
    uint32_t PacketTiles,
    typename DFBA,
    typename DFBB,
    typename DFBOut>
FORCE_INLINE void elementwise(DFBA& a, DFBB& b, DFBOut& output) {
    static_assert(PacketTiles > 0 && Count % PacketTiles == 0);
    constexpr uint32_t dst_tiles =
        ckernel::get_dest_max_tiles<DST_SYNC_MODE, DST_ACCUM_MODE, ckernel::DstTileShape::Tile32x32>();
    const uint32_t a_id = a.get_id();
    const uint32_t b_id = b.get_id();
    const uint32_t output_id = output.get_id();
    reconfig_data_format(a_id, b_id);
    if constexpr (Operation == ElementwiseOperation::ADD) {
        add_init(a_id, b_id);
    } else {
        sub_init(a_id, b_id);
    }
    for (uint32_t packet = 0; packet < Count; packet += PacketTiles) {
        output.reserve_back(PacketTiles);
        for (uint32_t first = 0; first < PacketTiles; first += dst_tiles) {
            const uint32_t count = first + dst_tiles <= PacketTiles ? dst_tiles : PacketTiles - first;
            const uint32_t input_start = packet + first;
            tile_regs_acquire();
            if constexpr (Operation == ElementwiseOperation::ADD) {
                add_block(a_id, b_id, input_start, input_start, 0, count);
            } else {
                sub_block(a_id, b_id, input_start, input_start, 0, count);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t tile = 0; tile < count; ++tile) {
                pack_tile(tile, output_id, first + tile);
            }
            tile_regs_release();
        }
        output.push_back(PacketTiles);
    }
}

template <uint32_t Count, uint32_t PacketTiles, typename DFBIn, typename DFBOut>
FORCE_INLINE void copy(DFBIn& input, DFBOut& output) {
    static_assert(PacketTiles > 0 && Count % PacketTiles == 0);
    constexpr uint32_t dst_tiles =
        ckernel::get_dest_max_tiles<DST_SYNC_MODE, DST_ACCUM_MODE, ckernel::DstTileShape::Tile32x32>();
    const uint32_t input_id = input.get_id();
    const uint32_t output_id = output.get_id();
    reconfig_data_format_srca(input_id);
    copy_init(input_id);
    for (uint32_t packet = 0; packet < Count; packet += PacketTiles) {
        output.reserve_back(PacketTiles);
        for (uint32_t first = 0; first < PacketTiles; first += dst_tiles) {
            const uint32_t count = first + dst_tiles <= PacketTiles ? dst_tiles : PacketTiles - first;
            tile_regs_acquire();
            for (uint32_t tile = 0; tile < count; ++tile) {
                copy_tile(input_id, packet + first + tile, tile);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t tile = 0; tile < count; ++tile) {
                pack_tile(tile, output_id, first + tile);
            }
            tile_regs_release();
        }
        output.push_back(PacketTiles);
    }
}

// difference[path * Vt + v] = values[v] - projection[path * Vt + v] for one tile row of Paths value groups.
template <uint32_t Vt, uint32_t Paths, typename DFBValues, typename DFBProjection, typename DFBDifference>
FORCE_INLINE void subtract_from_values(DFBValues& values, DFBProjection& projection, DFBDifference& difference) {
    constexpr uint32_t count = Vt * Paths;
    constexpr uint32_t dst_tiles =
        ckernel::get_dest_max_tiles<DST_SYNC_MODE, DST_ACCUM_MODE, ckernel::DstTileShape::Tile32x32>();
    const uint32_t values_id = values.get_id();
    const uint32_t projection_id = projection.get_id();
    const uint32_t difference_id = difference.get_id();
    reconfig_data_format(values_id, projection_id);
    sub_init(values_id, projection_id);
    difference.reserve_back(count);
    for (uint32_t first = 0; first < count; first += dst_tiles) {
        const uint32_t block = first + dst_tiles <= count ? dst_tiles : count - first;
        tile_regs_acquire();
        for (uint32_t tile = 0; tile < block; ++tile) {
            sub_tiles(values_id, projection_id, (first + tile) % Vt, first + tile, tile);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t tile = 0; tile < block; ++tile) {
            pack_tile(tile, difference_id, first + tile);
        }
        tile_regs_release();
    }
    difference.push_back(count);
}

// The summary carries the zero-seeded state B and the identity-seeded state A + B side by side: a [Kt, 2 * Vt]
// state whose first Vt columns hold B. Publish the complement form E = (A + B) - B - I, or copy B, one tile row of Vt
// tiles per packet, from the FP32 carry (UnpackToDest) with SFPU subtractions, so E is exact in FP32 before its single
// BF16 rounding. I is subtracted on the global diagonal tiles, where this value block's column v meets key tile row
// value_block * Vt + v (summaries require K == V).
template <
    bool Difference,
    uint32_t Kt,
    uint32_t Vt,
    uint32_t PacketRows,
    typename DFBCarry,
    typename DFBOut,
    typename DFBIdentity = DataflowBuffer>
FORCE_INLINE void extract_paths(
    DFBCarry& carry, DFBOut& output, DFBIdentity* identity = nullptr, uint32_t value_block = 0) {
    static_assert(Kt % PacketRows == 0);
    const uint32_t carry_id = carry.get_id();
    const uint32_t output_id = output.get_id();
    const uint32_t identity_id = identity == nullptr ? 0 : identity->get_id();
    reconfig_data_format_srca(carry_id);
    for (uint32_t first_row = 0; first_row < Kt; first_row += PacketRows) {
        output.reserve_back(PacketRows * Vt);
        for (uint32_t row = 0; row < PacketRows; ++row) {
            for (uint32_t value = 0; value < Vt; ++value) {
                const uint32_t b_tile = (first_row + row) * 2 * Vt + value;
                tile_regs_acquire();
                copy_init(carry_id);
                if constexpr (Difference) {
                    copy_tile(carry_id, b_tile + Vt, 0);
                    copy_tile(carry_id, b_tile, 1);
                    const bool diagonal = first_row + row == value_block * Vt + value;
                    if (diagonal) {
                        reconfig_data_format_srca(carry_id, identity_id);
                        copy_init(identity_id);
                        copy_tile(identity_id, 0, 2);
                        reconfig_data_format_srca(identity_id, carry_id);
                    }
                    sub_binary_tile_init();
                    sub_binary_tile(0, 1, 0);
                    if (diagonal) {
                        sub_binary_tile(0, 2, 0);
                    }
                } else {
                    copy_tile(carry_id, b_tile, 0);
                }
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, output_id, row * Vt + value);
                tile_regs_release();
            }
        }
        output.push_back(PacketRows * Vt);
    }
}

// Copy an UnpackToDest source exactly into one destination, releasing the source.
template <uint32_t Count, typename DFBSeed, typename DFBOut>
FORCE_INLINE void seed_state(DFBSeed& seed, DFBOut& out) {
    const uint32_t seed_id = seed.get_id();
    seed.wait_front(Count);
    out.reserve_back(Count);
    reconfig_data_format_srca(seed_id);
    copy_init(seed_id);
    for (uint32_t tile = 0; tile < Count; ++tile) {
        tile_regs_acquire();
        copy_tile(seed_id, tile, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, out.get_id(), tile);
        tile_regs_release();
    }
    out.push_back(Count);
    seed.pop_front(Count);
}

// Copy an UnpackToDest seed exactly into both the ring (the matmul view of the state) and the FP32 carry.
template <uint32_t Count, typename DFBSeed, typename DFBRing, typename DFBCarry>
FORCE_INLINE void seed_state(DFBSeed& seed, DFBRing& ring, DFBCarry& carry) {
    const uint32_t seed_id = seed.get_id();
    seed.wait_front(Count);
    ring.reserve_back(Count);
    carry.reserve_back(Count);
    pack_reconfig_data_format(ring.get_id());
    reconfig_data_format_srca(seed_id);
    copy_init(seed_id);
    for (uint32_t tile = 0; tile < Count; ++tile) {
        tile_regs_acquire();
        copy_tile(seed_id, tile, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, ring.get_id(), tile);
        pack_tile(0, carry.get_id(), tile);
        tile_regs_release();
    }
    ring.push_back(Count);
    carry.push_back(Count);
    seed.pop_front(Count);
}

template <
    ChunkInputPolicy InputPolicy,
    uint32_t Ct,
    uint32_t Kt,
    uint32_t Vt,
    uint32_t Paths = 1,
    typename DFBState,
    typename DFBKd,
    typename DFBVBeta,
    typename DFBTInv,
    typename DFBProjection,
    typename DFBDifference,
    typename DFBCorrected>
FORCE_INLINE void compute_value_new(
    DFBState& current_state,
    DFBKd& kd,
    DFBVBeta& v_beta,
    DFBTInv& t_inv,
    DFBProjection& state_projection,
    DFBDifference& difference,
    DFBCorrected& corrected_value) {
    constexpr uint32_t chunk_key_tiles = Ct * Kt;
    constexpr uint32_t chunk_chunk_tiles = Ct * Ct;
    constexpr uint32_t chunk_value_tiles = Ct * Vt;
    constexpr uint32_t state_columns = Paths * Vt;
    constexpr uint32_t chunk_state_tiles = Ct * state_columns;
    constexpr uint32_t key_state_tiles = Kt * state_columns;

    if constexpr (InputPolicy == ChunkInputPolicy::CONSUME) {
        kd.wait_front(chunk_key_tiles);
    }
    current_state.wait_front(key_state_tiles);
    matrix_multiply<Ct, Kt, state_columns>(kd, current_state, state_projection);
    state_projection.wait_front(chunk_state_tiles);
    if constexpr (InputPolicy == ChunkInputPolicy::CONSUME) {
        kd.pop_front(chunk_key_tiles);
        v_beta.wait_front(chunk_value_tiles);
    }
    if constexpr (Paths == 1) {
        elementwise<ElementwiseOperation::SUBTRACT, chunk_value_tiles, chunk_value_tiles>(
            v_beta, state_projection, difference);
    } else {
        static_assert(Ct == 1, "paired summaries take one tile row per chunk");
        subtract_from_values<Vt, Paths>(v_beta, state_projection, difference);
    }
    difference.wait_front(chunk_state_tiles);
    if constexpr (InputPolicy == ChunkInputPolicy::CONSUME) {
        v_beta.pop_front(chunk_value_tiles);
    }
    state_projection.pop_front(chunk_state_tiles);
    if constexpr (InputPolicy == ChunkInputPolicy::CONSUME) {
        t_inv.wait_front(chunk_chunk_tiles);
    }
    matrix_multiply<Ct, Ct, state_columns>(t_inv, difference, corrected_value);
    corrected_value.wait_front(chunk_state_tiles);
    if constexpr (InputPolicy == ChunkInputPolicy::CONSUME) {
        t_inv.pop_front(chunk_chunk_tiles);
    }
    difference.pop_front(chunk_state_tiles);
}

template <
    uint32_t Ct,
    uint32_t Kt,
    uint32_t Vt,
    typename DFBState,
    typename DFBCorrected,
    typename DFBQDecay,
    typename DFBIntra,
    typename DFBStateProjection,
    typename DFBValueProjection,
    typename DFBOut>
FORCE_INLINE void compute_chunk_output(
    DFBState& current_state,
    DFBCorrected& corrected_value,
    DFBQDecay& q_decay,
    DFBIntra& intra,
    DFBStateProjection& state_projection,
    DFBValueProjection& value_projection,
    DFBOut& output) {
    constexpr uint32_t chunk_chunk_tiles = Ct * Ct;
    constexpr uint32_t chunk_key_tiles = Ct * Kt;
    constexpr uint32_t chunk_value_tiles = Ct * Vt;

    q_decay.wait_front(chunk_key_tiles);
    matrix_multiply<Ct, Kt, Vt>(q_decay, current_state, state_projection);
    state_projection.wait_front(chunk_value_tiles);
    q_decay.pop_front(chunk_key_tiles);
    intra.wait_front(chunk_chunk_tiles);
    matrix_multiply<Ct, Ct, Vt>(intra, corrected_value, value_projection);
    value_projection.wait_front(chunk_value_tiles);
    intra.pop_front(chunk_chunk_tiles);
    pack_reconfig_data_format(output.get_id());
    elementwise<ElementwiseOperation::ADD, chunk_value_tiles, chunk_value_tiles>(
        state_projection, value_projection, output);
    state_projection.pop_front(chunk_value_tiles);
    value_projection.pop_front(chunk_value_tiles);
}

// S_{n+1} = S_n + (final_decay * S_n + k_dec_t @ U_n) with final_decay = expm1(G_last) (complement form). Only the
// small per-chunk change is formed in DST (the decay term by an eltwise broadcast multiply, the update accumulated onto
// it by the matmul); the packer then adds it onto the FP32 state in L1 (L1 accumulation), in place in both the ring
// (the matmul view of the state) and the carry (its UnpackToDest view). The state itself never passes through a source
// register or DST, so a long-memory channel's change is not rounded away. Both buffers hold exactly one state: popping
// it returns the write pointer to the same tiles, which the reads of this update are done with before each is packed.
template <
    ChunkInputPolicy InputPolicy,
    uint32_t Ct,
    uint32_t Kt,
    uint32_t Columns,
    typename DFBState,
    typename DFBCarry,
    typename DFBCorrected,
    typename DFBKDecayT,
    typename DFBFinalDecay>
FORCE_INLINE void update_state(
    DFBState& state,
    DFBCarry& carry,
    DFBCorrected& corrected_value,
    DFBKDecayT& k_decay_transposed,
    DFBFinalDecay& final_decay) {
    constexpr uint32_t dst_tiles =
        ckernel::get_dest_max_tiles<DST_SYNC_MODE, DST_ACCUM_MODE, ckernel::DstTileShape::Tile32x32>();
    // A DST block is `rows` key rows of `columns` state columns: whole rows when a row fits, else part of one row.
    // Each tile's change reads only that tile of the state, so any block can be accumulated in place.
    constexpr uint32_t columns = Columns < dst_tiles ? Columns : dst_tiles;
    static_assert(Columns % columns == 0, "state columns must split into equal DST blocks");
    constexpr uint32_t rows = columns < Columns ? 1 : (Kt < dst_tiles / Columns ? Kt : dst_tiles / Columns);
    static_assert(Kt % rows == 0, "key rows must split into equal DST blocks");
    constexpr uint32_t chunk_state_tiles = Ct * Columns;
    constexpr uint32_t key_state_tiles = Kt * Columns;
    constexpr uint32_t key_chunk_tiles = Kt * Ct;

    if constexpr (InputPolicy == ChunkInputPolicy::CONSUME) {
        k_decay_transposed.wait_front(key_chunk_tiles);
        final_decay.wait_front(Kt);
    }
    state.wait_front(key_state_tiles);
    carry.wait_front(key_state_tiles);
    // Release and re-reserve the same tiles: the packer accumulates the change onto them in place.
    state.pop_front(key_state_tiles);
    carry.pop_front(key_state_tiles);
    state.reserve_back(key_state_tiles);
    carry.reserve_back(key_state_tiles);

    const uint32_t state_id = state.get_id();
    const uint32_t carry_id = carry.get_id();
    const uint32_t decay_id = final_decay.get_id();
    const uint32_t k_decay_id = k_decay_transposed.get_id();
    const uint32_t value_id = corrected_value.get_id();
    pack_reconfig_data_format(state_id);
    pack_reconfig_l1_acc(1);
    for (uint32_t first_key = 0; first_key < Kt; first_key += rows) {
        for (uint32_t first_column = 0; first_column < Columns; first_column += columns) {
            tile_regs_acquire();
            reconfig_data_format(state_id, decay_id);
            mul_bcast_cols_init(state_id, decay_id);
            for (uint32_t row = 0; row < rows; ++row) {
                for (uint32_t column = 0; column < columns; ++column) {
                    const uint32_t key = first_key + row;
                    mul_tiles_bcast_cols(
                        state_id, decay_id, key * Columns + first_column + column, key, row * columns + column);
                }
            }  // f * S
            reconfig_data_format<SrcOrder::Reverse>(k_decay_id, value_id);
            matmul_block_init(k_decay_id, value_id, false, columns, 1, Ct);
            for (uint32_t row = 0; row < rows; ++row) {
                for (uint32_t chunk_tile = 0; chunk_tile < Ct; ++chunk_tile) {
                    matmul_block(
                        k_decay_id,
                        value_id,
                        (first_key + row) * Ct + chunk_tile,
                        chunk_tile * Columns + first_column,
                        row * columns,
                        false,
                        columns,
                        1,
                        Ct);  // += k_dec_t @ U
                }
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t row = 0; row < rows; ++row) {
                for (uint32_t column = 0; column < columns; ++column) {
                    const uint32_t tile = (first_key + row) * Columns + first_column + column;
                    pack_tile(row * columns + column, state_id, tile);
                    pack_tile(row * columns + column, carry_id, tile);
                }
            }
            tile_regs_release();
        }
    }
    pack_reconfig_l1_acc(0);
    state.push_back(key_state_tiles);
    carry.push_back(key_state_tiles);
    if constexpr (InputPolicy == ChunkInputPolicy::CONSUME) {
        k_decay_transposed.pop_front(key_chunk_tiles);
        final_decay.pop_front(Kt);
    }
    corrected_value.pop_front(chunk_state_tiles);
}

// The zero-seeded B and identity-seeded A + B recurrences advance together as one [Kt, 2 * Vt] state, so every
// operation covers both; each tile's arithmetic is the same as advancing them apart.
template <uint32_t Ct, uint32_t Kt, uint32_t Vt>
FORCE_INLINE void compute_summary(uint32_t num_chunks, uint32_t split_chunk, uint32_t value_block) {
    DataflowBuffer state(dfb::state);
    DataflowBuffer t_inv(dfb::t_inv);
    DataflowBuffer v_beta(dfb::v_beta);
    DataflowBuffer kd(dfb::kd);
    DataflowBuffer state_ring(dfb::state_ring);
    DataflowBuffer value_new(dfb::value_new);
    DataflowBuffer final_decay(dfb::final_decay);
    DataflowBuffer output(dfb::output);
    DataflowBuffer k_decay_transposed(dfb::k_decay_transposed);
    DataflowBuffer scratch(dfb::scratch);
    DataflowBuffer summary_head_output(dfb::summary_head_output);
    DataflowBuffer summary_head_state(dfb::summary_head_state);
    DataflowBuffer transport_state(dfb::transport_state);
    DataflowBuffer state_carry(dfb::state_carry);
    DataflowBuffer identity(dfb::identity);

    constexpr uint32_t chunk_chunk_tiles = Ct * Ct;
    constexpr uint32_t chunk_key_tiles = Ct * Kt;
    constexpr uint32_t chunk_value_tiles = Ct * Vt;
    constexpr uint32_t pair_columns = 2 * Vt;
    constexpr uint32_t key_pair_tiles = Kt * pair_columns;
    constexpr uint32_t key_chunk_tiles = Kt * Ct;

    identity.wait_front(2);
    seed_state<key_pair_tiles>(state, state_ring, state_carry);
    pack_reconfig_data_format(dfb::state_ring);
    for (uint32_t chunk = 0; chunk < num_chunks; chunk++) {
        auto& current = state_ring;

        kd.wait_front(chunk_key_tiles);
        v_beta.wait_front(chunk_value_tiles);
        t_inv.wait_front(chunk_chunk_tiles);
        k_decay_transposed.wait_front(key_chunk_tiles);
        final_decay.wait_front(Kt);
        compute_value_new<ChunkInputPolicy::RETAIN, Ct, Kt, Vt, 2>(
            current, kd, v_beta, t_inv, scratch, value_new, scratch);
        update_state<ChunkInputPolicy::RETAIN, Ct, Kt, pair_columns>(
            current, state_carry, scratch, k_decay_transposed, final_decay);
        if (split_chunk != 0 && chunk + 1 == split_chunk) {
            state_carry.wait_front(key_pair_tiles);
            pack_reconfig_data_format(summary_head_output.get_id());
            extract_paths<true, Kt, Vt, 1>(state_carry, summary_head_output, &identity, value_block);
            extract_paths<false, Kt, Vt, 1>(state_carry, summary_head_state);
            // Restart from zero and identity so the tail summary is independent
            // of the head transition just saved above.
            state_ring.wait_front(key_pair_tiles);
            state_ring.pop_front(key_pair_tiles);
            state_carry.pop_front(key_pair_tiles);
            seed_state<key_pair_tiles>(state, state_ring, state_carry);
            pack_reconfig_data_format(dfb::state_ring);
        }
        kd.pop_front(chunk_key_tiles);
        v_beta.pop_front(chunk_value_tiles);
        t_inv.pop_front(chunk_chunk_tiles);
        k_decay_transposed.pop_front(key_chunk_tiles);
        final_decay.pop_front(Kt);
    }
    state_ring.wait_front(key_pair_tiles);
    state_carry.wait_front(key_pair_tiles);
    pack_reconfig_data_format(output.get_id());
    constexpr uint32_t packet_rows = kda::largest_divisor_at_most(
        Kt, ckernel::get_dest_max_tiles<DST_SYNC_MODE, DST_ACCUM_MODE, ckernel::DstTileShape::Tile32x32>() / Vt);
    extract_paths<true, Kt, Vt, packet_rows>(state_carry, output, &identity, value_block);
    pack_reconfig_data_format(transport_state.get_id());
    extract_paths<false, Kt, Vt, packet_rows>(state_carry, transport_state);
    state_ring.pop_front(key_pair_tiles);
    state_carry.pop_front(key_pair_tiles);
}

template <uint32_t Ct, uint32_t Kt, uint32_t Vt>
FORCE_INLINE void compute_recurrent(uint32_t num_chunks, uint32_t reset_chunk) {
    DataflowBuffer state(dfb::state);
    DataflowBuffer t_inv(dfb::t_inv);
    DataflowBuffer v_beta(dfb::v_beta);
    DataflowBuffer kd(dfb::kd);
    DataflowBuffer q_decay(dfb::q_decay);
    DataflowBuffer intra(dfb::intra);
    DataflowBuffer state_ring(dfb::state_ring);
    DataflowBuffer value_new(dfb::value_new);
    DataflowBuffer final_decay(dfb::final_decay);
    DataflowBuffer output(dfb::output);
    DataflowBuffer output_intermediate(dfb::output_intermediate);
    DataflowBuffer k_decay_transposed(dfb::k_decay_transposed);
    DataflowBuffer final_state(dfb::final_state);
    DataflowBuffer scratch(dfb::scratch);
    DataflowBuffer tail_entry_states(dfb::tail_entry_states);
    DataflowBuffer state_carry(dfb::state_carry);
    DataflowBuffer identity(dfb::identity);

    constexpr uint32_t key_value_tiles = Kt * Vt;

    identity.wait_front(2);
    seed_state<key_value_tiles>(state, state_ring, state_carry);
    pack_reconfig_data_format(dfb::scratch);
    for (uint32_t chunk = 0; chunk < num_chunks; chunk++) {
        // A chronological split restarts the causal stream mid-group. The recurrence is affine in
        // the state, so no per-chunk term changes -- only where the carry comes
        // from. reset_chunk 0 means never, which is exact rather than a sentinel:
        // r == 0 means no group straddles, and chunk 0 always seeds from `state`.
        auto& current_state = state_ring;
        if (reset_chunk != 0 && chunk == reset_chunk) {
            state_ring.wait_front(key_value_tiles);
            state_carry.wait_front(key_value_tiles);
            state_ring.pop_front(key_value_tiles);
            state_carry.pop_front(key_value_tiles);
            seed_state<key_value_tiles>(tail_entry_states, state_ring, state_carry);
            pack_reconfig_data_format(dfb::scratch);
        }
        compute_value_new<ChunkInputPolicy::CONSUME, Ct, Kt, Vt>(
            current_state, kd, v_beta, t_inv, scratch, output_intermediate, value_new);
        compute_chunk_output<Ct, Kt, Vt>(
            current_state, value_new, q_decay, intra, output_intermediate, scratch, output);

        pack_reconfig_data_format(dfb::state_ring);
        update_state<ChunkInputPolicy::CONSUME, Ct, Kt, Vt>(
            current_state, state_carry, value_new, k_decay_transposed, final_decay);
        pack_reconfig_data_format(dfb::scratch);
    }
    // The final state leaves exactly: copied from the FP32 carry straight into DST.
    state_ring.wait_front(key_value_tiles);
    state_ring.pop_front(key_value_tiles);
    pack_reconfig_data_format(final_state.get_id());
    seed_state<key_value_tiles>(state_carry, final_state);
}

template <uint32_t Ct, uint32_t Kt, uint32_t Vt, uint32_t summary>
TT_KERNEL void compute(uint32_t num_chunks, uint32_t group, uint32_t value_block) {
    compute_kernel_hw_startup<SrcOrder::Reverse>(dfb::kd, dfb::v_beta, dfb::output);

    kda_chronology::Topology topology{};
    {
        DataflowBuffer chronology(dfb::chronology_compute);
        topology = kda_chronology::receive(chronology);
    }
    const uint32_t groups = topology.local_rows / tt::constants::TILE_HEIGHT / num_chunks;
    const uint32_t reset_chunk = topology.reset_chunk(group, groups);
    const uint32_t valid_chunks = topology.valid_chunks(group, groups);
    if (valid_chunks == 0) {
        return;
    }
    if constexpr (summary) {
        compute_summary<Ct, Kt, Vt>(valid_chunks, reset_chunk, value_block);
    } else {
        compute_recurrent<Ct, Kt, Vt>(valid_chunks, reset_chunk);
    }
}
