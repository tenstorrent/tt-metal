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
#include "api/compute/matmul.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/operations/experimental/kda/device/kernels/compute/matmul_subblock.hpp"

enum class ElementwiseOperation { ADD, SUBTRACT };

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

// output = operand OP (a @ b), with the product kept in DST: each row group's matmul accumulates into DST and the
// destination-reuse binary op takes it from there, so the product never round-trips through a buffer. The operand
// takes source A and the product source B, as in the two-operand op on a packed product. The carried state is
// unchanged; the chunk output's last bits can differ.
// The operand's tile for output tile t is t % OperandTiles (the summary's two paths share one value row).
template <
    EltwiseBinaryType Op,
    uint32_t Mt,
    uint32_t Kt,
    uint32_t Nt,
    uint32_t OperandTiles,
    typename DFBA,
    typename DFBB,
    typename DFBOperand,
    typename DFBOut>
FORCE_INLINE void matrix_multiply_then(DFBA& a, DFBB& b, DFBOperand& operand, DFBOut& output) {
    constexpr uint32_t dst_tiles =
        ckernel::get_dest_max_tiles<DST_SYNC_MODE, DST_ACCUM_MODE, ckernel::DstTileShape::Tile32x32>();
    // Whole output rows per destination acquire when they fit, else column groups of one row (the summary's
    // two-path rows are twice the state width).
    constexpr uint32_t group_columns = Nt <= dst_tiles ? Nt : dst_tiles;
    static_assert(Nt % group_columns == 0, "column groups must tile an output row");
    constexpr uint32_t group_rows = Nt <= dst_tiles ? (dst_tiles / Nt < Mt ? dst_tiles / Nt : Mt) : 1;
    static_assert(Mt % group_rows == 0);
    const uint32_t a_id = a.get_id();
    const uint32_t b_id = b.get_id();
    const uint32_t operand_id = operand.get_id();
    const uint32_t output_id = output.get_id();

    output.reserve_back(Mt * Nt);
    for (uint32_t first_row = 0; first_row < Mt; first_row += group_rows) {
        for (uint32_t first_column = 0; first_column < Nt; first_column += group_columns) {
            reconfig_data_format<SrcOrder::Reverse>(a_id, b_id);
            matmul_block_init(a_id, b_id, false, group_columns, 1, Kt);
            tile_regs_acquire();
            for (uint32_t row = 0; row < group_rows; ++row) {
                for (uint32_t k = 0; k < Kt; ++k) {
                    matmul_block(
                        a_id,
                        b_id,
                        (first_row + row) * Kt + k,
                        k * Nt + first_column,
                        row * group_columns,
                        false,
                        group_columns,
                        1,
                        Kt);
                }
            }
            // The operand unpacks to srcA; any BF16 output mask leaves it in a different format than the state. The
            // product moves from DST to srcB in srcB's format, kept FP32 (state_ring's, under every mask) so a BF16
            // matmul input does not round it.
            reconfig_data_format_srca(operand_id);
            reconfig_data_format_srcb(dfb::state_ring);
            if constexpr (Op == EltwiseBinaryType::ELWADD) {
                add_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCB>(operand_id);
            } else {
                sub_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCB>(operand_id);
            }
            for (uint32_t row = 0; row < group_rows; ++row) {
                for (uint32_t column = 0; column < group_columns; ++column) {
                    const uint32_t tile = row * group_columns + column;
                    const uint32_t operand_tile = ((first_row + row) * Nt + first_column + column) % OperandTiles;
                    if constexpr (Op == EltwiseBinaryType::ELWADD) {
                        add_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCB>(operand_id, operand_tile, tile);
                    } else {
                        sub_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCB>(operand_id, operand_tile, tile);
                    }
                }
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t row = 0; row < group_rows; ++row) {
                for (uint32_t column = 0; column < group_columns; ++column) {
                    pack_tile(
                        row * group_columns + column,
                        output_id,
                        (first_row + row) * Nt + first_column + column);
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

// The summary carries the zero-seeded state B and the identity-seeded state A + B side by side: a [Kt, 2 * Vt]
// state whose first Vt columns hold B. Publish (A + B) - B, or copy B, one tile row of Vt tiles per packet.
template <bool Difference, uint32_t Kt, uint32_t Vt, uint32_t PacketRows, typename DFBPair, typename DFBOut>
FORCE_INLINE void extract_paths(DFBPair& pair, DFBOut& output) {
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

template <typename DFBState, typename DFBDecay, typename DFBOut>
FORCE_INLINE void multiply_by_decay(
    DFBState& state, DFBDecay& decay, DFBOut& output, uint32_t key_tiles, uint32_t value_tiles) {
    constexpr uint32_t dst_tiles =
        ckernel::get_dest_max_tiles<DST_SYNC_MODE, DST_ACCUM_MODE, ckernel::DstTileShape::Tile32x32>();
    const uint32_t count = key_tiles * value_tiles;
    const uint32_t state_id = state.get_id();
    const uint32_t decay_id = decay.get_id();
    const uint32_t output_id = output.get_id();

    output.reserve_back(count);
    reconfig_data_format(state_id, decay_id);
    mul_bcast_cols_init(state_id, decay_id);
    // Batch independent tiles to amortize destination-register lifecycle overhead.
    for (uint32_t block_start = 0; block_start < count; block_start += dst_tiles) {
        const uint32_t remaining = count - block_start;
        const uint32_t block_tiles = remaining < dst_tiles ? remaining : dst_tiles;
        tile_regs_acquire();
        for (uint32_t tile = 0; tile < block_tiles; ++tile) {
            const uint32_t index = block_start + tile;
            const uint32_t key = index / value_tiles;
            mul_tiles_bcast_cols(state_id, decay_id, index, key, tile);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t tile = 0; tile < block_tiles; ++tile) {
            pack_tile(tile, output_id, block_start + tile);
        }
        tile_regs_release();
    }
    output.push_back(count);
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
    DataflowBuffer state_update(dfb::state_update);
    DataflowBuffer state_temporary(dfb::state_temporary);

    constexpr uint32_t chunk_chunk_tiles = Ct * Ct;
    constexpr uint32_t chunk_key_tiles = Ct * Kt;
    constexpr uint32_t chunk_value_tiles = Ct * Vt;
    constexpr uint32_t pair_columns = 2 * Vt;
    constexpr uint32_t key_pair_tiles = Kt * pair_columns;
    constexpr uint32_t key_chunk_tiles = Kt * Ct;

    pack_reconfig_data_format(dfb::state_ring);
    for (uint32_t chunk = 0; chunk < num_chunks; chunk++) {
        auto& current = chunk == 0 ? state : state_ring;
        const bool last = chunk == num_chunks - 1;

        kd.wait_front(chunk_key_tiles);
        v_beta.wait_front(chunk_value_tiles);
        t_inv.wait_front(chunk_chunk_tiles);
        k_decay_transposed.wait_front(key_chunk_tiles);
        final_decay.wait_front(Kt);
        // The decayed state reads only the incoming state, so it is issued between kd S and the subtraction that
        // waits on it. The operations and their operand order are those of compute_value_new and update_state.
        current.wait_front(key_pair_tiles);
        matrix_multiply_then<EltwiseBinaryType::ELWSUB, Ct, Kt, pair_columns, Vt>(
            kd, current, v_beta, value_new);  // v_beta - kd S
        multiply_by_decay(current, final_decay, state_temporary, Kt, pair_columns);
        value_new.wait_front(Ct * pair_columns);
        matrix_multiply<Ct, Ct, pair_columns>(t_inv, value_new, scratch);  // v_new
        scratch.wait_front(Ct * pair_columns);
        value_new.pop_front(Ct * pair_columns);
        state_temporary.wait_front(key_pair_tiles);
        matrix_multiply_then<EltwiseBinaryType::ELWADD, Kt, Ct, pair_columns, key_pair_tiles>(
            k_decay_transposed, scratch, state_temporary, last ? final_state : state_ring);  // dl S + k_dec_t v_new
        state_temporary.pop_front(key_pair_tiles);
        scratch.pop_front(Ct * pair_columns);
        current.pop_front(key_pair_tiles);
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
    DataflowBuffer query_projection(dfb::query_projection);
    DataflowBuffer k_decay_transposed(dfb::k_decay_transposed);
    DataflowBuffer final_state(dfb::final_state);
    DataflowBuffer scratch(dfb::scratch);
    DataflowBuffer state_update(dfb::state_update);
    DataflowBuffer state_temporary(dfb::state_temporary);
    DataflowBuffer tail_entry_states(dfb::tail_entry_states);

    constexpr uint32_t chunk_chunk_tiles = Ct * Ct;
    constexpr uint32_t chunk_key_tiles = Ct * Kt;
    constexpr uint32_t chunk_value_tiles = Ct * Vt;
    constexpr uint32_t key_value_tiles = Kt * Vt;
    constexpr uint32_t key_chunk_tiles = Kt * Ct;

    pack_reconfig_data_format(dfb::scratch);
    for (uint32_t chunk = 0; chunk < num_chunks; chunk++) {
        // A chronological split restarts the causal stream mid-group. The recurrence is affine in
        // the state, so no per-chunk term changes -- only where the carry comes
        // from. reset_chunk 0 means never, which is exact rather than a sentinel:
        // r == 0 means no group straddles, and chunk 0 always seeds from `state`.
        auto& current_state = chunk == 0 ? state : state_ring;
        if (reset_chunk != 0 && chunk == reset_chunk) {
            state_ring.wait_front(key_value_tiles);
            tail_entry_states.wait_front(key_value_tiles);
            copy<key_value_tiles, key_value_tiles>(tail_entry_states, state_ring);
            state_ring.pop_front(key_value_tiles);
            tail_entry_states.pop_front(key_value_tiles);
        }
        auto& destination = chunk == num_chunks - 1 ? final_state : state_ring;

        // Each stage waits for the previous one's packed result. Every stage that reads only the incoming state
        // runs first, and each dependent stage is issued after independent work, so the math thread rarely waits on
        // a pack. The operations and their operand order are those of the straight-line recurrence.
        current_state.wait_front(key_value_tiles);
        kd.wait_front(chunk_key_tiles);
        v_beta.wait_front(chunk_value_tiles);
        matrix_multiply_then<EltwiseBinaryType::ELWSUB, Ct, Kt, Vt, chunk_value_tiles>(
            kd, current_state, v_beta, output_intermediate);  // v_beta - kd S
        q_decay.wait_front(chunk_key_tiles);
        matrix_multiply<Ct, Kt, Vt>(q_decay, current_state, query_projection);  // q_decay S
        final_decay.wait_front(Kt);
        multiply_by_decay(current_state, final_decay, state_temporary, Kt, Vt);  // dl * S
        kd.pop_front(chunk_key_tiles);
        q_decay.pop_front(chunk_key_tiles);
        final_decay.pop_front(Kt);

        // v_new = T_inv (v_beta - kd S).
        output_intermediate.wait_front(chunk_value_tiles);
        v_beta.pop_front(chunk_value_tiles);
        t_inv.wait_front(chunk_chunk_tiles);
        matrix_multiply<Ct, Ct, Vt>(t_inv, output_intermediate, value_new);
        value_new.wait_front(chunk_value_tiles);
        t_inv.pop_front(chunk_chunk_tiles);
        output_intermediate.pop_front(chunk_value_tiles);

        // The state update and the output's intra-chunk term both read v_new. The update and the decayed state are
        // each packed to FP32 and then added: accumulating the matmul onto the decayed state in DST instead changes
        // the result enough that Kimi-K3's chunked prefill drifts from the reference over consecutive calls.
        k_decay_transposed.wait_front(key_chunk_tiles);
        state_temporary.wait_front(key_value_tiles);
        matrix_multiply_then<EltwiseBinaryType::ELWADD, Kt, Ct, Vt, key_value_tiles>(
            k_decay_transposed, value_new, state_temporary, destination);  // dl S + k_dec_t v_new
        intra.wait_front(chunk_chunk_tiles);
        matrix_multiply<Ct, Ct, Vt>(intra, value_new, scratch);  // intra v_new
        state_temporary.pop_front(key_value_tiles);
        k_decay_transposed.pop_front(key_chunk_tiles);
        intra.pop_front(chunk_chunk_tiles);
        value_new.pop_front(chunk_value_tiles);
        current_state.pop_front(key_value_tiles);

        // The chunk output, issued after the next state so the next chunk's first stage never waits on it.
        scratch.wait_front(chunk_value_tiles);
        query_projection.wait_front(chunk_value_tiles);
        pack_reconfig_data_format(output.get_id());
        elementwise<ElementwiseOperation::ADD, chunk_value_tiles, chunk_value_tiles>(query_projection, scratch, output);
        query_projection.pop_front(chunk_value_tiles);
        scratch.pop_front(chunk_value_tiles);
        pack_reconfig_data_format(dfb::state_ring);
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
