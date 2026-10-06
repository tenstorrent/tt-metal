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

// Exact-state helpers. The running state S is kept twice: the ring copies feed matmuls through SrcA/SrcB (TF32),
// while the carry copies (UnpackToDest DFBs) hold S in FP32 for the state update. Any value read through a source
// register loses its low mantissa bits; the long-memory channels of the state change by less than that per chunk,
// so the update must add to the FP32 carry, never to a source-register copy (tt_metal_tracker-g1b.7).

// Copy an UnpackToDest seed exactly into both the ring (matmul view) and the carry (FP32 view).
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

// decay_diagonal[kt] = diag(final_decay[kt]) as one 32x32 tile per key tile.
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

template <ChunkInputPolicy InputPolicy, uint32_t Ct, uint32_t Kt, uint32_t Vt>
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
    constexpr uint32_t key_value_tiles = Kt * Vt;

    if constexpr (InputPolicy == ChunkInputPolicy::CONSUME) {
        kd.wait_front(chunk_key_tiles);
    }
    current_state.wait_front(key_value_tiles);
    matrix_multiply<Ct, Kt, Vt>(kd, current_state, state_projection);
    state_projection.wait_front(chunk_value_tiles);
    if constexpr (InputPolicy == ChunkInputPolicy::CONSUME) {
        kd.pop_front(chunk_key_tiles);
    }
    if constexpr (InputPolicy == ChunkInputPolicy::CONSUME) {
        v_beta.wait_front(chunk_value_tiles);
    }
    elementwise<ElementwiseOperation::SUBTRACT, chunk_value_tiles, chunk_value_tiles>(
        v_beta, state_projection, difference);
    difference.wait_front(chunk_value_tiles);
    if constexpr (InputPolicy == ChunkInputPolicy::CONSUME) {
        v_beta.pop_front(chunk_value_tiles);
    }
    state_projection.pop_front(chunk_value_tiles);
    if constexpr (InputPolicy == ChunkInputPolicy::CONSUME) {
        t_inv.wait_front(chunk_chunk_tiles);
    }
    matrix_multiply<Ct, Ct, Vt>(t_inv, difference, corrected_value);
    corrected_value.wait_front(chunk_value_tiles);
    if constexpr (InputPolicy == ChunkInputPolicy::CONSUME) {
        t_inv.pop_front(chunk_chunk_tiles);
    }
    difference.pop_front(chunk_value_tiles);
}

// Homogeneous counterpart of compute_value_new for the transition chain: U = t_inv @ (0 - kd @ S). The zero tile is
// the identity DFB's second tile. Chunk inputs stay resident (the summary's chains share them).
template <uint32_t Ct, uint32_t Kt, uint32_t Vt>
FORCE_INLINE void compute_homogeneous_value(
    DataflowBuffer& current_state,
    DataflowBuffer& kd,
    DataflowBuffer& identity,
    DataflowBuffer& t_inv,
    DataflowBuffer& state_projection,
    DataflowBuffer& difference,
    DataflowBuffer& corrected_value) {
    constexpr uint32_t chunk_value_tiles = Ct * Vt;
    constexpr uint32_t key_value_tiles = Kt * Vt;
    constexpr uint32_t dst_tiles =
        ckernel::get_dest_max_tiles<DST_SYNC_MODE, DST_ACCUM_MODE, ckernel::DstTileShape::Tile32x32>();

    current_state.wait_front(key_value_tiles);
    matrix_multiply<Ct, Kt, Vt>(kd, current_state, state_projection);
    state_projection.wait_front(chunk_value_tiles);
    const uint32_t zero_id = identity.get_id();
    const uint32_t projection_id = state_projection.get_id();
    const uint32_t difference_id = difference.get_id();
    reconfig_data_format(zero_id, projection_id);
    sub_init(zero_id, projection_id);
    difference.reserve_back(chunk_value_tiles);
    for (uint32_t first = 0; first < chunk_value_tiles; first += dst_tiles) {
        const uint32_t count = first + dst_tiles <= chunk_value_tiles ? dst_tiles : chunk_value_tiles - first;
        tile_regs_acquire();
        for (uint32_t tile = 0; tile < count; ++tile) {
            sub_tiles(zero_id, projection_id, 1, first + tile, tile);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t tile = 0; tile < count; ++tile) {
            pack_tile(tile, difference_id, first + tile);
        }
        tile_regs_release();
    }
    difference.push_back(chunk_value_tiles);
    difference.wait_front(chunk_value_tiles);
    state_projection.pop_front(chunk_value_tiles);
    matrix_multiply<Ct, Ct, Vt>(t_inv, difference, corrected_value);
    corrected_value.wait_front(chunk_value_tiles);
    difference.pop_front(chunk_value_tiles);
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

// S_{n+1} = S_n + (diag(final_decay) S_n + k_dec_t @ U_n), final_decay = expm1(G_last) (complement form).
// Each output tile starts from the FP32 carry copied straight into DST; both products accumulate onto it in DST.
// The new state is packed to the ring (matmul view) and back to the carry, one key row at a time.
template <ChunkInputPolicy InputPolicy, uint32_t Ct, uint32_t Kt, uint32_t Vt>
FORCE_INLINE void update_state(
    DataflowBuffer& current_state,
    DataflowBuffer& carry,
    DataflowBuffer& destination,
    DataflowBuffer& corrected_value,
    DataflowBuffer& k_decay_transposed,
    DataflowBuffer& final_decay,
    DataflowBuffer& identity,
    DataflowBuffer& decay_diagonal) {
    constexpr uint32_t chunk_value_tiles = Ct * Vt;
    constexpr uint32_t key_value_tiles = Kt * Vt;
    constexpr uint32_t key_chunk_tiles = Kt * Ct;
    static_assert(
        Vt <= ckernel::get_dest_max_tiles<DST_SYNC_MODE, DST_ACCUM_MODE, ckernel::DstTileShape::Tile32x32>(),
        "one key row of the state must fit in DST");

    if constexpr (InputPolicy == ChunkInputPolicy::CONSUME) {
        k_decay_transposed.wait_front(key_chunk_tiles);
        final_decay.wait_front(Kt);
    }
    build_decay_diagonal<Kt>(identity, final_decay, decay_diagonal);
    decay_diagonal.wait_front(Kt);
    if constexpr (InputPolicy == ChunkInputPolicy::CONSUME) {
        final_decay.pop_front(Kt);
    }
    current_state.wait_front(key_value_tiles);
    carry.wait_front(key_value_tiles);

    const uint32_t carry_id = carry.get_id();
    const uint32_t k_dec_id = k_decay_transposed.get_id();
    const uint32_t value_id = corrected_value.get_id();
    const uint32_t diagonal_id = decay_diagonal.get_id();
    const uint32_t state_id = current_state.get_id();
    const uint32_t destination_id = destination.get_id();
    destination.reserve_back(key_value_tiles);
    pack_reconfig_data_format(destination_id);
    for (uint32_t key = 0; key < Kt; ++key) {
        tile_regs_acquire();
        reconfig_data_format_srca(carry_id);
        copy_init(carry_id);
        for (uint32_t value = 0; value < Vt; ++value) {
            copy_tile(carry_id, value, value);  // FP32 S_n row, front-relative
        }
        reconfig_data_format<SrcOrder::Reverse>(k_dec_id, value_id);
        matmul_block_init(k_dec_id, value_id, false, Vt, 1, Ct);
        for (uint32_t chunk_tile = 0; chunk_tile < Ct; ++chunk_tile) {
            matmul_block(k_dec_id, value_id, key * Ct + chunk_tile, chunk_tile * Vt, 0, false, Vt, 1, Ct);
        }
        reconfig_data_format<SrcOrder::Reverse>(diagonal_id, state_id);
        matmul_block_init(diagonal_id, state_id, false, Vt, 1, 1);
        matmul_block(diagonal_id, state_id, key, key * Vt, 0, false, Vt, 1, 1);
        tile_regs_commit();
        carry.pop_front(Vt);
        carry.reserve_back(Vt);
        tile_regs_wait();
        for (uint32_t value = 0; value < Vt; ++value) {
            pack_tile(value, destination_id, key * Vt + value);
            pack_tile(value, carry_id, value);
        }
        tile_regs_release();
        carry.push_back(Vt);
    }
    destination.push_back(key_value_tiles);
    decay_diagonal.pop_front(Kt);
    current_state.pop_front(key_value_tiles);
    corrected_value.pop_front(chunk_value_tiles);
    if constexpr (InputPolicy == ChunkInputPolicy::CONSUME) {
        k_decay_transposed.pop_front(key_chunk_tiles);
    }
}

// Affine summary in complement form: A - I = S_a - I from the FP32 carry of the identity-seeded homogeneous chain
// (v_beta = 0), and B = S_b from the zero-seeded chain. A must not be formed as F(I) - F(0) - I: both chains carry B,
// so an outlier value channel (|B| ~ 100 in a long-memory head) cancels out of A's same column and leaves the chains'
// rounding there (~0.2 against true entries ~1e-2), which E @ carry then multiplies by the state
// (tt_metal_tracker-g1b.5.17). All of A is emitted before B, in packets of PacketTiles: the writer drains the two
// outputs in that order, so interleaving them would deadlock once a packet buffer fills. The carries stay at the front.
template <uint32_t Kt, uint32_t Vt, uint32_t PacketTiles>
FORCE_INLINE void emit_complement_summary(
    DataflowBuffer& carry_a,
    DataflowBuffer& carry_b,
    DataflowBuffer& identity,
    DataflowBuffer& out_a,
    DataflowBuffer& out_b) {
    constexpr uint32_t key_value_tiles = Kt * Vt;
    static_assert(key_value_tiles % PacketTiles == 0);
    const uint32_t a_id = carry_a.get_id();
    const uint32_t b_id = carry_b.get_id();
    const uint32_t identity_id = identity.get_id();
    carry_a.wait_front(key_value_tiles);
    carry_b.wait_front(key_value_tiles);
    pack_reconfig_data_format(out_a.get_id());
    // One SFPU init for the whole emission: the interleaved copies reprogram only the datacopy address modifiers.
    sub_binary_tile_init();
    for (uint32_t packet = 0; packet < key_value_tiles; packet += PacketTiles) {
        out_a.reserve_back(PacketTiles);
        for (uint32_t offset = 0; offset < PacketTiles; ++offset) {
            const uint32_t tile = packet + offset;
            const bool diagonal = tile / Vt == tile % Vt;
            tile_regs_acquire();
            reconfig_data_format_srca(a_id);
            copy_init(a_id);
            copy_tile(a_id, tile, 0);
            // DST 1 holds I on diagonal tiles and stays zero otherwise (subtracting +0 is exact).
            reconfig_data_format_srca(identity_id);
            copy_init(identity_id);
            copy_tile(identity_id, diagonal ? 0 : 1, 1);
            sub_binary_tile(0, 1, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, out_a.get_id(), offset);
            tile_regs_release();
        }
        out_a.push_back(PacketTiles);
    }
    reconfig_data_format_srca(b_id);
    copy_init(b_id);
    for (uint32_t packet = 0; packet < key_value_tiles; packet += PacketTiles) {
        out_b.reserve_back(PacketTiles);
        for (uint32_t offset = 0; offset < PacketTiles; ++offset) {
            tile_regs_acquire();
            copy_tile(b_id, packet + offset, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, out_b.get_id(), offset);
            tile_regs_release();
        }
        out_b.push_back(PacketTiles);
    }
}

template <uint32_t Ct, uint32_t Kt, uint32_t Vt>
FORCE_INLINE void compute_summary(uint32_t num_chunks, uint32_t split_chunk) {
    DataflowBuffer state(dfb::state);
    DataflowBuffer t_inv(dfb::t_inv);
    DataflowBuffer v_beta(dfb::v_beta);
    DataflowBuffer kd(dfb::kd);
    DataflowBuffer state_ring(dfb::state_ring);
    DataflowBuffer state_carry(dfb::state_carry);
    DataflowBuffer value_new(dfb::value_new);
    DataflowBuffer final_decay(dfb::final_decay);
    DataflowBuffer output(dfb::output);
    DataflowBuffer k_decay_transposed(dfb::k_decay_transposed);
    DataflowBuffer scratch(dfb::scratch);
    DataflowBuffer transport_state(dfb::transport_state);
    DataflowBuffer summary_seed(dfb::summary_seed);
    DataflowBuffer summary_ring(dfb::summary_ring);
    DataflowBuffer summary_carry(dfb::summary_carry);
    DataflowBuffer summary_head_output(dfb::summary_head_output);
    DataflowBuffer summary_head_state(dfb::summary_head_state);
    DataflowBuffer identity(dfb::identity);
    DataflowBuffer decay_diagonal(dfb::decay_diagonal);

    constexpr uint32_t chunk_chunk_tiles = Ct * Ct;
    constexpr uint32_t chunk_key_tiles = Ct * Kt;
    constexpr uint32_t chunk_value_tiles = Ct * Vt;
    constexpr uint32_t key_value_tiles = Kt * Vt;
    constexpr uint32_t key_chunk_tiles = Kt * Ct;

    identity.wait_front(2);
    seed_state<key_value_tiles>(state, state_ring, state_carry);
    seed_state<key_value_tiles>(summary_seed, summary_ring, summary_carry);
    for (uint32_t chunk = 0; chunk < num_chunks; chunk++) {
        kd.wait_front(chunk_key_tiles);
        v_beta.wait_front(chunk_value_tiles);
        t_inv.wait_front(chunk_chunk_tiles);
        k_decay_transposed.wait_front(key_chunk_tiles);
        final_decay.wait_front(Kt);
        pack_reconfig_data_format(dfb::scratch);
        compute_value_new<ChunkInputPolicy::RETAIN, Ct, Kt, Vt>(
            state_ring, kd, v_beta, t_inv, scratch, value_new, scratch);
        update_state<ChunkInputPolicy::RETAIN, Ct, Kt, Vt>(
            state_ring, state_carry, state_ring, scratch, k_decay_transposed, final_decay, identity, decay_diagonal);
        pack_reconfig_data_format(dfb::scratch);
        compute_homogeneous_value<Ct, Kt, Vt>(summary_ring, kd, identity, t_inv, scratch, value_new, scratch);
        update_state<ChunkInputPolicy::RETAIN, Ct, Kt, Vt>(
            summary_ring,
            summary_carry,
            summary_ring,
            scratch,
            k_decay_transposed,
            final_decay,
            identity,
            decay_diagonal);
        if (split_chunk != 0 && chunk + 1 == split_chunk) {
            emit_complement_summary<Kt, Vt, Vt>(
                summary_carry, state_carry, identity, summary_head_output, summary_head_state);
            // Restart from zero and identity so the tail summary is independent of the head transition saved above.
            state_carry.pop_front(key_value_tiles);
            summary_carry.pop_front(key_value_tiles);
            state_ring.wait_front(key_value_tiles);
            summary_ring.wait_front(key_value_tiles);
            seed_state<key_value_tiles>(state, state_ring, state_carry);
            seed_state<key_value_tiles>(summary_seed, summary_ring, summary_carry);
            state_ring.pop_front(key_value_tiles);
            summary_ring.pop_front(key_value_tiles);
        }
        kd.pop_front(chunk_key_tiles);
        v_beta.pop_front(chunk_value_tiles);
        t_inv.pop_front(chunk_chunk_tiles);
        k_decay_transposed.pop_front(key_chunk_tiles);
        final_decay.pop_front(Kt);
    }
    emit_complement_summary<Kt, Vt, key_value_tiles>(summary_carry, state_carry, identity, output, transport_state);
    state_carry.pop_front(key_value_tiles);
    summary_carry.pop_front(key_value_tiles);
    state_ring.wait_front(key_value_tiles);
    summary_ring.wait_front(key_value_tiles);
    state_ring.pop_front(key_value_tiles);
    summary_ring.pop_front(key_value_tiles);
    identity.pop_front(2);
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
    DataflowBuffer state_carry(dfb::state_carry);
    DataflowBuffer value_new(dfb::value_new);
    DataflowBuffer final_decay(dfb::final_decay);
    DataflowBuffer output(dfb::output);
    DataflowBuffer output_intermediate(dfb::output_intermediate);
    DataflowBuffer k_decay_transposed(dfb::k_decay_transposed);
    DataflowBuffer final_state(dfb::final_state);
    DataflowBuffer scratch(dfb::scratch);
    DataflowBuffer tail_entry_states(dfb::tail_entry_states);
    DataflowBuffer identity(dfb::identity);
    DataflowBuffer decay_diagonal(dfb::decay_diagonal);

    constexpr uint32_t key_value_tiles = Kt * Vt;

    identity.wait_front(2);
    seed_state<key_value_tiles>(state, state_ring, state_carry);
    for (uint32_t chunk = 0; chunk < num_chunks; chunk++) {
        // A chronological split restarts the causal stream mid-group. The recurrence is affine in
        // the state, so no per-chunk term changes -- only where the carry comes
        // from. reset_chunk 0 means never, which is exact rather than a sentinel:
        // r == 0 means no group straddles, and chunk 0 always seeds from `state`.
        if (reset_chunk != 0 && chunk == reset_chunk) {
            state_ring.wait_front(key_value_tiles);
            state_carry.wait_front(key_value_tiles);
            state_carry.pop_front(key_value_tiles);
            seed_state<key_value_tiles>(tail_entry_states, state_ring, state_carry);
            state_ring.pop_front(key_value_tiles);
        }
        DataflowBuffer& destination = chunk == num_chunks - 1 ? final_state : state_ring;

        pack_reconfig_data_format(dfb::scratch);
        compute_value_new<ChunkInputPolicy::CONSUME, Ct, Kt, Vt>(
            state_ring, kd, v_beta, t_inv, scratch, output_intermediate, value_new);
        compute_chunk_output<Ct, Kt, Vt>(state_ring, value_new, q_decay, intra, output_intermediate, scratch, output);
        update_state<ChunkInputPolicy::CONSUME, Ct, Kt, Vt>(
            state_ring, state_carry, destination, value_new, k_decay_transposed, final_decay, identity, decay_diagonal);
    }
    state_carry.wait_front(key_value_tiles);
    state_carry.pop_front(key_value_tiles);
    identity.pop_front(2);
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
