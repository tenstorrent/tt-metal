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
// state whose first Vt columns hold B. Publish (A + B) - B, or copy B, one tile row of Vt tiles per packet.
template <bool Difference, uint32_t Kt, uint32_t Vt, uint32_t PacketRows>
FORCE_INLINE void extract_paths(DataflowBuffer& pair, DataflowBuffer& output) {
    constexpr uint32_t dst_tiles =
        ckernel::get_dest_max_tiles<DST_SYNC_MODE, DST_ACCUM_MODE, ckernel::DstTileShape::Tile32x32>();
    static_assert(Kt % PacketRows == 0 && PacketRows * Vt <= dst_tiles);
    const uint32_t pair_id = pair.get_id();
    const uint32_t output_id = output.get_id();
    if constexpr (Difference) {
        reconfig_data_format(pair_id, pair_id);
        sub_init(pair_id, pair_id);
    } else {
        reconfig_data_format_srca(pair_id);
        copy_init(pair_id);
    }
    for (uint32_t first_row = 0; first_row < Kt; first_row += PacketRows) {
        output.reserve_back(PacketRows * Vt);
        tile_regs_acquire();
        for (uint32_t row = 0; row < PacketRows; ++row) {
            for (uint32_t value = 0; value < Vt; ++value) {
                const uint32_t b_tile = (first_row + row) * 2 * Vt + value;
                if constexpr (Difference) {
                    sub_tiles(pair_id, pair_id, b_tile + Vt, b_tile, row * Vt + value);
                } else {
                    copy_tile(pair_id, b_tile, row * Vt + value);
                }
            }
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t tile = 0; tile < PacketRows * Vt; ++tile) {
            pack_tile(tile, output_id, tile);
        }
        tile_regs_release();
        output.push_back(PacketRows * Vt);
    }
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
    }
    if constexpr (InputPolicy == ChunkInputPolicy::CONSUME) {
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

// destination = state * decay (each key row scaled by its decay) + k_decay_transposed @ corrected: the decayed state
// lands in DST and the matmul accumulates the update onto it, so neither term leaves DST before the sum.
template <uint32_t Ct, uint32_t Kt, uint32_t Vt>
FORCE_INLINE void decay_and_accumulate_update(
    DataflowBuffer& state,
    DataflowBuffer& decay,
    DataflowBuffer& k_decay_transposed,
    DataflowBuffer& corrected,
    DataflowBuffer& destination) {
    constexpr uint32_t dst_tiles =
        ckernel::get_dest_max_tiles<DST_SYNC_MODE, DST_ACCUM_MODE, ckernel::DstTileShape::Tile32x32>();
    // Each pass covers a block of pass_rows key rows by pass_columns value columns.
    constexpr uint32_t pass_columns = kda::largest_divisor_at_most(Vt, dst_tiles);
    constexpr uint32_t pass_rows = kda::largest_divisor_at_most(Kt, dst_tiles / pass_columns);
    const uint32_t state_id = state.get_id();
    const uint32_t decay_id = decay.get_id();
    const uint32_t k_id = k_decay_transposed.get_id();
    const uint32_t corrected_id = corrected.get_id();
    const uint32_t destination_id = destination.get_id();

    destination.reserve_back(Kt * Vt);
    for (uint32_t first_row = 0; first_row < Kt; first_row += pass_rows) {
        for (uint32_t first_column = 0; first_column < Vt; first_column += pass_columns) {
            tile_regs_acquire();
            reconfig_data_format(state_id, decay_id);
            mul_bcast_cols_init(state_id, decay_id);
            for (uint32_t row = 0; row < pass_rows; ++row) {
                for (uint32_t column = 0; column < pass_columns; ++column) {
                    const uint32_t index = (first_row + row) * Vt + first_column + column;
                    mul_tiles_bcast_cols(state_id, decay_id, index, first_row + row, row * pass_columns + column);
                }
            }
            reconfig_data_format<SrcOrder::Reverse>(k_id, corrected_id);
            matmul_block_init(k_id, corrected_id, false, pass_columns, pass_rows, Ct);
            for (uint32_t k = 0; k < Ct; ++k) {
                matmul_block(
                    k_id,
                    corrected_id,
                    first_row * Ct + k,
                    k * Vt + first_column,
                    0,
                    false,
                    pass_columns,
                    pass_rows,
                    Ct);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t row = 0; row < pass_rows; ++row) {
                for (uint32_t column = 0; column < pass_columns; ++column) {
                    pack_tile(
                        row * pass_columns + column, destination_id, (first_row + row) * Vt + first_column + column);
                }
            }
            tile_regs_release();
        }
    }
    destination.push_back(Kt * Vt);
}

template <ChunkInputPolicy InputPolicy, uint32_t Ct, uint32_t Kt, uint32_t Vt>
FORCE_INLINE void update_state(
    DataflowBuffer& current_state,
    DataflowBuffer& destination,
    DataflowBuffer& corrected_value,
    DataflowBuffer& k_decay_transposed,
    DataflowBuffer& final_decay) {
    constexpr uint32_t chunk_value_tiles = Ct * Vt;
    constexpr uint32_t key_value_tiles = Kt * Vt;
    constexpr uint32_t key_chunk_tiles = Kt * Ct;

    if constexpr (InputPolicy == ChunkInputPolicy::CONSUME) {
        k_decay_transposed.wait_front(key_chunk_tiles);
        final_decay.wait_front(Kt);
    }
    decay_and_accumulate_update<Ct, Kt, Vt>(
        current_state, final_decay, k_decay_transposed, corrected_value, destination);
    if constexpr (InputPolicy == ChunkInputPolicy::CONSUME) {
        k_decay_transposed.pop_front(key_chunk_tiles);
        final_decay.pop_front(Kt);
    }
    corrected_value.pop_front(chunk_value_tiles);
    current_state.pop_front(key_value_tiles);
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

    constexpr uint32_t chunk_chunk_tiles = Ct * Ct;
    constexpr uint32_t chunk_key_tiles = Ct * Kt;
    constexpr uint32_t chunk_value_tiles = Ct * Vt;
    constexpr uint32_t pair_columns = 2 * Vt;
    constexpr uint32_t key_pair_tiles = Kt * pair_columns;
    constexpr uint32_t key_chunk_tiles = Kt * Ct;

    pack_reconfig_data_format(dfb::state_ring);
    for (uint32_t chunk = 0; chunk < num_chunks; chunk++) {
        DataflowBuffer& current = chunk == 0 ? state : state_ring;
        const bool last = chunk == num_chunks - 1;

        kd.wait_front(chunk_key_tiles);
        v_beta.wait_front(chunk_value_tiles);
        t_inv.wait_front(chunk_chunk_tiles);
        k_decay_transposed.wait_front(key_chunk_tiles);
        final_decay.wait_front(Kt);
        compute_value_new<ChunkInputPolicy::RETAIN, Ct, Kt, Vt, 2>(
            current, kd, v_beta, t_inv, scratch, value_new, scratch);
        update_state<ChunkInputPolicy::RETAIN, Ct, Kt, pair_columns>(
            current, last ? final_state : state_ring, scratch, k_decay_transposed, final_decay);
        if (split_chunk != 0 && chunk + 1 == split_chunk) {
            state_ring.wait_front(key_pair_tiles);
            pack_reconfig_data_format(summary_head_output.get_id());
            extract_paths<true, Kt, Vt, 1>(state_ring, summary_head_output);
            extract_paths<false, Kt, Vt, 1>(state_ring, summary_head_state);
            pack_reconfig_data_format(dfb::state_ring);
            // Restart from zero and identity so the tail summary is independent
            // of the head transition just saved above.
            state.wait_front(key_pair_tiles);
            copy<key_pair_tiles, key_pair_tiles>(state, state_ring);
            state_ring.pop_front(key_pair_tiles);
            state.pop_front(key_pair_tiles);
        }
        kd.pop_front(chunk_key_tiles);
        v_beta.pop_front(chunk_value_tiles);
        t_inv.pop_front(chunk_chunk_tiles);
        k_decay_transposed.pop_front(key_chunk_tiles);
        final_decay.pop_front(Kt);
    }
    final_state.wait_front(key_pair_tiles);
    pack_reconfig_data_format(output.get_id());
    constexpr uint32_t packet_rows = kda::largest_divisor_at_most(
        Kt, ckernel::get_dest_max_tiles<DST_SYNC_MODE, DST_ACCUM_MODE, ckernel::DstTileShape::Tile32x32>() / Vt);
    extract_paths<true, Kt, Vt, packet_rows>(final_state, output);
    extract_paths<false, Kt, Vt, packet_rows>(final_state, transport_state);
    final_state.pop_front(key_pair_tiles);
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

    constexpr uint32_t key_value_tiles = Kt * Vt;

    pack_reconfig_data_format(dfb::scratch);
    for (uint32_t chunk = 0; chunk < num_chunks; chunk++) {
        // A chronological split restarts the causal stream mid-group. The recurrence is affine in
        // the state, so no per-chunk term changes -- only where the carry comes
        // from. reset_chunk 0 means never, which is exact rather than a sentinel:
        // r == 0 means no group straddles, and chunk 0 always seeds from `state`.
        DataflowBuffer& current_state = chunk == 0 ? state : state_ring;
        if (reset_chunk != 0 && chunk == reset_chunk) {
            state_ring.wait_front(key_value_tiles);
            tail_entry_states.wait_front(key_value_tiles);
            copy<key_value_tiles, key_value_tiles>(tail_entry_states, state_ring);
            state_ring.pop_front(key_value_tiles);
            tail_entry_states.pop_front(key_value_tiles);
        }
        DataflowBuffer& destination = chunk == num_chunks - 1 ? final_state : state_ring;

        compute_value_new<ChunkInputPolicy::CONSUME, Ct, Kt, Vt>(
            current_state, kd, v_beta, t_inv, scratch, output_intermediate, value_new);
        compute_chunk_output<Ct, Kt, Vt>(
            current_state, value_new, q_decay, intra, output_intermediate, scratch, output);

        pack_reconfig_data_format(dfb::state_ring);
        update_state<ChunkInputPolicy::CONSUME, Ct, Kt, Vt>(
            current_state, destination, value_new, k_decay_transposed, final_decay);
    }
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
