// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/operations/experimental/kda/chronological_selections/device/kernels/chronology.hpp"
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

template <uint32_t Mt, uint32_t Kt, uint32_t Nt>
FORCE_INLINE void matrix_multiply(DataflowBuffer& a, DataflowBuffer& b, DataflowBuffer& output) {
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
template <ElementwiseOperation Operation, uint32_t Count, uint32_t PacketTiles>
FORCE_INLINE void elementwise(DataflowBuffer& a, DataflowBuffer& b, DataflowBuffer& output) {
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

template <uint32_t Count, uint32_t PacketTiles>
FORCE_INLINE void copy(DataflowBuffer& input, DataflowBuffer& output) {
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
template <uint32_t Vt, uint32_t Paths>
FORCE_INLINE void subtract_from_values(DataflowBuffer& values, DataflowBuffer& projection, DataflowBuffer& difference) {
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
// state whose first Vt columns hold B. Publish (A + B) - B, or copy B, one tile row of Vt tiles per packet, from the
// FP32 carry (UnpackToDest) with an SFPU subtraction, so A is exact in FP32 before it is packed.
template <bool Difference, uint32_t Kt, uint32_t Vt, uint32_t PacketRows>
FORCE_INLINE void extract_paths(DataflowBuffer& carry, DataflowBuffer& output) {
    static_assert(Kt % PacketRows == 0);
    const uint32_t carry_id = carry.get_id();
    const uint32_t output_id = output.get_id();
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
                    sub_binary_tile_init();
                    sub_binary_tile(0, 1, 0);
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

// Copy an UnpackToDest seed exactly into both the ring (the matmul view of the state) and the FP32 carry.
template <uint32_t Count>
FORCE_INLINE void seed_state(DataflowBuffer& seed, DataflowBuffer& ring, DataflowBuffer& carry) {
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

// decay_diagonal[k] = diag(final_decay[k]): the identity tile with each row scaled by its key's decay complement.
template <uint32_t Kt>
FORCE_INLINE void build_decay_diagonal(
    DataflowBuffer& identity, DataflowBuffer& final_decay, DataflowBuffer& decay_diagonal) {
    const uint32_t identity_id = identity.get_id();
    const uint32_t decay_id = final_decay.get_id();
    decay_diagonal.reserve_back(Kt);
    pack_reconfig_data_format(decay_diagonal.get_id());
    reconfig_data_format(identity_id, decay_id);
    mul_bcast_cols_init(identity_id, decay_id);
    for (uint32_t key = 0; key < Kt; ++key) {
        tile_regs_acquire();
        mul_tiles_bcast_cols(identity_id, decay_id, 0, key, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, decay_diagonal.get_id(), key);
        tile_regs_release();
    }
    decay_diagonal.push_back(Kt);
}

template <ChunkInputPolicy InputPolicy, uint32_t Ct, uint32_t Kt, uint32_t Vt, uint32_t Paths = 1>
FORCE_INLINE void compute_value_new(
    DataflowBuffer& current_state,
    DataflowBuffer& kd,
    DataflowBuffer& v_beta,
    DataflowBuffer& t_inv,
    DataflowBuffer& state_projection,
    DataflowBuffer& difference,
    DataflowBuffer& corrected_value) {
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

template <uint32_t Ct, uint32_t Kt, uint32_t Vt>
FORCE_INLINE void compute_chunk_output(
    DataflowBuffer& current_state,
    DataflowBuffer& corrected_value,
    DataflowBuffer& q_decay,
    DataflowBuffer& intra,
    DataflowBuffer& state_projection,
    DataflowBuffer& value_projection,
    DataflowBuffer& output) {
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

// S_{n+1} = S_n + (diag(final_decay) S_n + k_dec_t @ U_n) with final_decay = expm1(G_last) (complement form). Each
// output tile starts from the FP32 carry copied straight into DST and both products accumulate onto it there, so the
// small per-chunk change of a long-memory channel is never rounded to a source register's precision. The new state is
// packed to the destination (the ring, or final_state) and back to the carry, one key row at a time.
template <ChunkInputPolicy InputPolicy, uint32_t Ct, uint32_t Kt, uint32_t Columns>
FORCE_INLINE void update_state(
    DataflowBuffer& current_state,
    DataflowBuffer& carry,
    DataflowBuffer& destination,
    DataflowBuffer& corrected_value,
    DataflowBuffer& k_decay_transposed,
    DataflowBuffer& final_decay,
    DataflowBuffer& identity,
    DataflowBuffer& decay_diagonal) {
    constexpr uint32_t dst_tiles =
        ckernel::get_dest_max_tiles<DST_SYNC_MODE, DST_ACCUM_MODE, ckernel::DstTileShape::Tile32x32>();
    constexpr uint32_t block = Columns < dst_tiles ? Columns : dst_tiles;
    static_assert(Columns % block == 0, "state columns must split into equal DST blocks");
    constexpr uint32_t chunk_state_tiles = Ct * Columns;
    constexpr uint32_t key_state_tiles = Kt * Columns;
    constexpr uint32_t key_chunk_tiles = Kt * Ct;

    if constexpr (InputPolicy == ChunkInputPolicy::CONSUME) {
        k_decay_transposed.wait_front(key_chunk_tiles);
        final_decay.wait_front(Kt);
    }
    build_decay_diagonal<Kt>(identity, final_decay, decay_diagonal);
    decay_diagonal.wait_front(Kt);
    current_state.wait_front(key_state_tiles);
    carry.wait_front(key_state_tiles);

    const uint32_t carry_id = carry.get_id();
    const uint32_t k_decay_id = k_decay_transposed.get_id();
    const uint32_t value_id = corrected_value.get_id();
    const uint32_t diagonal_id = decay_diagonal.get_id();
    const uint32_t state_id = current_state.get_id();
    const uint32_t destination_id = destination.get_id();
    destination.reserve_back(key_state_tiles);
    pack_reconfig_data_format(destination_id);
    for (uint32_t key = 0; key < Kt; ++key) {
        carry.reserve_back(Columns);
        for (uint32_t first = 0; first < Columns; first += block) {
            tile_regs_acquire();
            reconfig_data_format_srca(carry_id);
            copy_init(carry_id);
            for (uint32_t column = 0; column < block; ++column) {
                copy_tile(carry_id, first + column, column);  // FP32 S_n, this key row (front-relative)
            }
            reconfig_data_format<SrcOrder::Reverse>(k_decay_id, value_id);
            matmul_block_init(k_decay_id, value_id, false, block, 1, Ct);
            for (uint32_t chunk_tile = 0; chunk_tile < Ct; ++chunk_tile) {
                matmul_block(
                    k_decay_id, value_id, key * Ct + chunk_tile, chunk_tile * Columns + first, 0, false, block, 1, Ct);
            }
            reconfig_data_format<SrcOrder::Reverse>(diagonal_id, state_id);
            matmul_block_init(diagonal_id, state_id, false, block, 1, 1);
            matmul_block(diagonal_id, state_id, key, key * Columns + first, 0, false, block, 1, 1);
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t column = 0; column < block; ++column) {
                pack_tile(column, destination_id, key * Columns + first + column);
                pack_tile(column, carry_id, first + column);
            }
            tile_regs_release();
        }
        carry.pop_front(Columns);
        carry.push_back(Columns);
    }
    destination.push_back(key_state_tiles);
    decay_diagonal.pop_front(Kt);
    if constexpr (InputPolicy == ChunkInputPolicy::CONSUME) {
        k_decay_transposed.pop_front(key_chunk_tiles);
        final_decay.pop_front(Kt);
    }
    corrected_value.pop_front(chunk_state_tiles);
    current_state.pop_front(key_state_tiles);
}

// The zero-seeded B and identity-seeded A + B recurrences advance together as one [Kt, 2 * Vt] state, so every
// operation covers both; each tile's arithmetic is the same as advancing them apart.
template <uint32_t Ct, uint32_t Kt, uint32_t Vt>
FORCE_INLINE void compute_summary(uint32_t num_chunks, uint32_t split_chunk) {
    DataflowBuffer state(dfb::state);
    DataflowBuffer t_inv(dfb::t_inv);
    DataflowBuffer v_beta(dfb::v_beta);
    DataflowBuffer kd(dfb::kd);
    DataflowBuffer state_ring(dfb::state_ring);
    DataflowBuffer value_new(dfb::value_new);
    DataflowBuffer final_decay(dfb::final_decay);
    DataflowBuffer output(dfb::output);
    DataflowBuffer k_decay_transposed(dfb::k_decay_transposed);
    DataflowBuffer final_state(dfb::final_state);
    DataflowBuffer scratch(dfb::scratch);
    DataflowBuffer summary_head_output(dfb::summary_head_output);
    DataflowBuffer summary_head_state(dfb::summary_head_state);
    DataflowBuffer transport_state(dfb::transport_state);
    DataflowBuffer state_carry(dfb::state_carry);
    DataflowBuffer identity(dfb::identity);
    DataflowBuffer decay_diagonal(dfb::decay_diagonal);

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
        DataflowBuffer& current = state_ring;
        const bool last = chunk == num_chunks - 1;

        kd.wait_front(chunk_key_tiles);
        v_beta.wait_front(chunk_value_tiles);
        t_inv.wait_front(chunk_chunk_tiles);
        k_decay_transposed.wait_front(key_chunk_tiles);
        final_decay.wait_front(Kt);
        compute_value_new<ChunkInputPolicy::RETAIN, Ct, Kt, Vt, 2>(
            current, kd, v_beta, t_inv, scratch, value_new, scratch);
        update_state<ChunkInputPolicy::RETAIN, Ct, Kt, pair_columns>(
            current,
            state_carry,
            last ? final_state : state_ring,
            scratch,
            k_decay_transposed,
            final_decay,
            identity,
            decay_diagonal);
        if (split_chunk != 0 && chunk + 1 == split_chunk) {
            state_carry.wait_front(key_pair_tiles);
            pack_reconfig_data_format(summary_head_output.get_id());
            extract_paths<true, Kt, Vt, 1>(state_carry, summary_head_output);
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
    final_state.wait_front(key_pair_tiles);
    state_carry.wait_front(key_pair_tiles);
    pack_reconfig_data_format(output.get_id());
    constexpr uint32_t packet_rows = kda::largest_divisor_at_most(
        Kt, ckernel::get_dest_max_tiles<DST_SYNC_MODE, DST_ACCUM_MODE, ckernel::DstTileShape::Tile32x32>() / Vt);
    extract_paths<true, Kt, Vt, packet_rows>(state_carry, output);
    pack_reconfig_data_format(transport_state.get_id());
    extract_paths<false, Kt, Vt, packet_rows>(state_carry, transport_state);
    final_state.pop_front(key_pair_tiles);
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
    DataflowBuffer decay_diagonal(dfb::decay_diagonal);

    constexpr uint32_t key_value_tiles = Kt * Vt;

    identity.wait_front(2);
    seed_state<key_value_tiles>(state, state_ring, state_carry);
    pack_reconfig_data_format(dfb::scratch);
    for (uint32_t chunk = 0; chunk < num_chunks; chunk++) {
        // A chronological split restarts the causal stream mid-group. The recurrence is affine in
        // the state, so no per-chunk term changes -- only where the carry comes
        // from. reset_chunk 0 means never, which is exact rather than a sentinel:
        // r == 0 means no group straddles, and chunk 0 always seeds from `state`.
        DataflowBuffer& current_state = state_ring;
        if (reset_chunk != 0 && chunk == reset_chunk) {
            state_ring.wait_front(key_value_tiles);
            state_carry.wait_front(key_value_tiles);
            state_ring.pop_front(key_value_tiles);
            state_carry.pop_front(key_value_tiles);
            seed_state<key_value_tiles>(tail_entry_states, state_ring, state_carry);
            pack_reconfig_data_format(dfb::scratch);
        }
        DataflowBuffer& destination = chunk == num_chunks - 1 ? final_state : state_ring;

        compute_value_new<ChunkInputPolicy::CONSUME, Ct, Kt, Vt>(
            current_state, kd, v_beta, t_inv, scratch, output_intermediate, value_new);
        compute_chunk_output<Ct, Kt, Vt>(
            current_state, value_new, q_decay, intra, output_intermediate, scratch, output);

        pack_reconfig_data_format(dfb::state_ring);
        update_state<ChunkInputPolicy::CONSUME, Ct, Kt, Vt>(
            current_state,
            state_carry,
            destination,
            value_new,
            k_decay_transposed,
            final_decay,
            identity,
            decay_diagonal);
        pack_reconfig_data_format(dfb::scratch);
    }
    state_carry.wait_front(key_value_tiles);
    state_carry.pop_front(key_value_tiles);
}

template <uint32_t Ct, uint32_t Kt, uint32_t Vt, uint32_t summary>
TT_KERNEL void compute(uint32_t num_chunks, uint32_t group) {
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
        compute_summary<Ct, Kt, Vt>(valid_chunks, reset_chunk);
    } else {
        compute_recurrent<Ct, Kt, Vt>(valid_chunks, reset_chunk);
    }
}
