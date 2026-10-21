// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/operations/experimental/kda/device/kernels/chronology.hpp"
//
// Phase B (scan) compute kernel: the sequential-over-chunk recurrence for one
// head.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/matmul.h"
#include "api/compute/reconfig_data_format.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/kda/device/kernels/compute/matmul_subblock.hpp"

enum class ElementwiseOperation { ADD, SUBTRACT };
enum class ChunkInputPolicy { RETAIN, CONSUME };

namespace ckl = compute_kernel_lib;

// Inputs are waited for and popped by the caller; each helper reserves and publishes its whole output, and the packer
// keeps the format configured before the helper.
constexpr ckl::InputSpec resident(uint32_t dfb_id) {
    return ckl::input(dfb_id, ckl::WaitPolicy::None, ckl::PopPolicy::None, ckl::InputTileMapping::Block);
}

constexpr ckl::OutputSpec published(uint32_t dfb_id) {
    return ckl::output(dfb_id, ckl::ReservePolicy::Upfront, ckl::PushPolicy::AtEnd, ckl::DataFormatReconfig::Disabled);
}

// A resident operand read from a runtime tile offset.
constexpr ckl::InputSpec resident_from(uint32_t dfb_id) {
    return ckl::input(
        dfb_id,
        ckl::WaitPolicy::None,
        ckl::PopPolicy::None,
        ckl::InputTileMapping::Block,
        ckl::DataFormatReconfig::Enabled,
        ckl::TileAddressing::Offset);
}

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
    uint32_t ADfb,
    uint32_t BDfb,
    uint32_t OutputDfb,
    typename DFBA,
    typename DFBB,
    typename DFBOut>
FORCE_INLINE void elementwise(DFBA& a, DFBB& b, DFBOut& output) {
    static_assert(PacketTiles > 0 && Count % PacketTiles == 0);
    constexpr uint32_t dst_tiles =
        ckernel::get_dest_max_tiles<DST_SYNC_MODE, DST_ACCUM_MODE, ckernel::DstTileShape::Tile32x32>();
    constexpr ckl::BinaryFpuOp op =
        Operation == ElementwiseOperation::ADD ? ckl::BinaryFpuOp::Add : ckl::BinaryFpuOp::Sub;
    for (uint32_t packet = 0; packet < Count; packet += PacketTiles) {
        ckl::eltwise_chain(
            ckl::IterationShape::tiles(PacketTiles).block_size(dst_tiles),
            ckl::BinaryFpu<op, resident_from(ADfb), resident_from(BDfb)>{a, b, packet, packet},
            ckl::PackTile<published(OutputDfb)>{output});
    }
}

// Out of line: called only on a chronological split, it would otherwise grow the inlined kernel past the point where
// GCC stops inlining the unpacker setup at kernel start.
template <uint32_t Count, uint32_t PacketTiles, uint32_t InputDfb, uint32_t OutputDfb, typename DFBIn, typename DFBOut>
__attribute__((noinline)) void copy(DFBIn& input, DFBOut& output) {
    static_assert(PacketTiles > 0 && Count % PacketTiles == 0);
    constexpr uint32_t dst_tiles =
        ckernel::get_dest_max_tiles<DST_SYNC_MODE, DST_ACCUM_MODE, ckernel::DstTileShape::Tile32x32>();
    for (uint32_t packet = 0; packet < Count; packet += PacketTiles) {
        ckl::eltwise_chain(
            ckl::IterationShape::tiles(PacketTiles).block_size(dst_tiles),
            ckl::CopyTile<resident_from(InputDfb)>{input, packet},
            ckl::PackTile<published(OutputDfb)>{output});
    }
}

// difference[path * Vt + v] = values[v] - projection[path * Vt + v] for one tile row of Paths value groups.
template <
    uint32_t Vt,
    uint32_t Paths,
    uint32_t ValuesDfb,
    uint32_t ProjectionDfb,
    uint32_t DifferenceDfb,
    typename DFBValues,
    typename DFBProjection,
    typename DFBDifference>
FORCE_INLINE void subtract_from_values(DFBValues& values, DFBProjection& projection, DFBDifference& difference) {
    constexpr uint32_t dst_tiles =
        ckernel::get_dest_max_tiles<DST_SYNC_MODE, DST_ACCUM_MODE, ckernel::DstTileShape::Tile32x32>();
    // The values row is re-read for every path.
    ckl::sub<
        ckl::input(ValuesDfb, ckl::WaitPolicy::None, ckl::PopPolicy::None, ckl::InputTileMapping::Row),
        resident(ProjectionDfb),
        published(DifferenceDfb)>(
        ckl::IterationShape::grid(Paths, Vt).block_size(dst_tiles), values, projection, difference);
}

// The summary carries the zero-seeded state B and the identity-seeded state A + B side by side: a [Kt, 2 * Vt]
// state whose first Vt columns hold B. Publish (A + B) - B, or copy B, one tile row of Vt tiles per packet.
template <
    bool Difference,
    uint32_t Kt,
    uint32_t Vt,
    uint32_t PacketRows,
    uint32_t PairDfb,
    uint32_t OutputDfb,
    typename DFBPair,
    typename DFBOut>
FORCE_INLINE void extract_paths(DFBPair& pair, DFBOut& output) {
    constexpr uint32_t dst_tiles =
        ckernel::get_dest_max_tiles<DST_SYNC_MODE, DST_ACCUM_MODE, ckernel::DstTileShape::Tile32x32>();
    static_assert(Kt % PacketRows == 0 && PacketRows * Vt <= dst_tiles);
    constexpr uint32_t pair_columns = 2 * Vt;
    // A packet's PacketRows x Vt tiles of one path; the other path sits Vt tiles to the right in each row.
    constexpr auto packet = ckl::IterationShape::grid(PacketRows, Vt).block_size(Vt);
    constexpr ckl::InputSpec pair_rows = ckl::input(
        PairDfb,
        ckl::WaitPolicy::None,
        ckl::PopPolicy::None,
        ckl::InputTileMapping::Block,
        ckl::DataFormatReconfig::Enabled,
        ckl::TileAddressing::Strided);
    // The packet window is reserved and published around the chain.
    using PackPacket = ckl::PackTile<ckl::output(
        OutputDfb, ckl::ReservePolicy::None, ckl::PushPolicy::None, ckl::DataFormatReconfig::Disabled)>;
    for (uint32_t first_row = 0; first_row < Kt; first_row += PacketRows) {
        const uint32_t b_tile = first_row * pair_columns;
        output.reserve_back(PacketRows * Vt);
        if constexpr (Difference) {
            ckl::eltwise_chain(
                packet,
                ckl::BinaryFpu<ckl::BinaryFpuOp::Sub, pair_rows, pair_rows>{
                    pair,
                    pair,
                    ckl::StridedTileRange{b_tile + Vt, pair_columns},
                    ckl::StridedTileRange{b_tile, pair_columns}},
                PackPacket{output});
        } else {
            ckl::eltwise_chain(
                packet,
                ckl::CopyTile<pair_rows>{pair, ckl::StridedTileRange{b_tile, pair_columns}},
                PackPacket{output});
        }
        output.push_back(PacketRows * Vt);
    }
}

template <
    ChunkInputPolicy InputPolicy,
    uint32_t Ct,
    uint32_t Kt,
    uint32_t Vt,
    uint32_t ProjectionDfb,
    uint32_t DifferenceDfb,
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
        elementwise<
            ElementwiseOperation::SUBTRACT,
            chunk_value_tiles,
            chunk_value_tiles,
            dfb::v_beta,
            ProjectionDfb,
            DifferenceDfb>(v_beta, state_projection, difference);
    } else {
        static_assert(Ct == 1, "paired summaries take one tile row per chunk");
        subtract_from_values<Vt, Paths, dfb::v_beta, ProjectionDfb, DifferenceDfb>(
            v_beta, state_projection, difference);
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
    uint32_t StateProjectionDfb,
    uint32_t ValueProjectionDfb,
    uint32_t OutputDfb,
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
    elementwise<
        ElementwiseOperation::ADD,
        chunk_value_tiles,
        chunk_value_tiles,
        StateProjectionDfb,
        ValueProjectionDfb,
        OutputDfb>(state_projection, value_projection, output);
    state_projection.pop_front(chunk_value_tiles);
    value_projection.pop_front(chunk_value_tiles);
}

// Out of line: update_state dispatches to one of two buffer specializations per chunk, and inlining both into the
// chunk loop makes small recurrences about 4% slower.
template <
    uint32_t StateDfb,
    uint32_t DecayDfb,
    uint32_t OutputDfb,
    typename DFBState,
    typename DFBDecay,
    typename DFBOut>
__attribute__((noinline)) void multiply_by_decay(
    DFBState& state, DFBDecay& decay, DFBOut& output, uint32_t key_tiles, uint32_t value_tiles) {
    constexpr uint32_t dst_tiles =
        ckernel::get_dest_max_tiles<DST_SYNC_MODE, DST_ACCUM_MODE, ckernel::DstTileShape::Tile32x32>();
    // Batch independent tiles to amortize destination-register lifecycle overhead.
    ckl::mul<
        resident(StateDfb),
        ckl::input(
            DecayDfb, ckl::BroadcastDim::Col, ckl::WaitPolicy::None, ckl::PopPolicy::None, ckl::InputTileMapping::Col),
        published(OutputDfb)>(
        ckl::IterationShape::grid(key_tiles, value_tiles).block_size(dst_tiles), state, decay, output);
}

template <
    ChunkInputPolicy InputPolicy,
    uint32_t Ct,
    uint32_t Kt,
    uint32_t Vt,
    typename DFBState,
    typename DFBDestination,
    typename DFBCorrected,
    typename DFBKDecayT,
    typename DFBFinalDecay>
FORCE_INLINE void update_state(
    DFBState& current_state,
    DFBDestination& destination,
    DFBCorrected& corrected_value,
    DFBKDecayT& k_decay_transposed,
    DFBFinalDecay& final_decay) {
    constexpr uint32_t chunk_value_tiles = Ct * Vt;
    constexpr uint32_t key_value_tiles = Kt * Vt;
    constexpr uint32_t key_chunk_tiles = Kt * Ct;

    if constexpr (InputPolicy == ChunkInputPolicy::CONSUME) {
        k_decay_transposed.wait_front(key_chunk_tiles);
        final_decay.wait_front(Kt);
    }
    // The update and the decayed state are each packed to FP32 and then added. Accumulating the matmul onto the
    // decayed state in DST instead changes the result enough that Kimi-K3's chunked prefill drifts from the
    // reference over consecutive calls (L24 worst-layer PCC 0.981 -> 0.953 at 56k tokens).
    DataflowBuffer state_update(dfb::state_update);
    DataflowBuffer state_temporary(dfb::state_temporary);
    matrix_multiply<Kt, Ct, Vt>(k_decay_transposed, corrected_value, state_update);
    state_update.wait_front(key_value_tiles);
    // The chunk reads either the seed state or the state ring and writes either the ring or the final state; each
    // elementwise step dispatches on the buffer selected for this chunk.
    if (current_state.get_id() == dfb::state) {
        multiply_by_decay<dfb::state, dfb::final_decay, dfb::state_temporary>(
            current_state, final_decay, state_temporary, Kt, Vt);
    } else {
        multiply_by_decay<dfb::state_ring, dfb::final_decay, dfb::state_temporary>(
            current_state, final_decay, state_temporary, Kt, Vt);
    }
    state_temporary.wait_front(key_value_tiles);
    if (destination.get_id() == dfb::final_state) {
        elementwise<
            ElementwiseOperation::ADD,
            key_value_tiles,
            key_value_tiles,
            dfb::state_temporary,
            dfb::state_update,
            dfb::final_state>(state_temporary, state_update, destination);
    } else {
        elementwise<
            ElementwiseOperation::ADD,
            key_value_tiles,
            key_value_tiles,
            dfb::state_temporary,
            dfb::state_update,
            dfb::state_ring>(state_temporary, state_update, destination);
    }
    state_temporary.pop_front(key_value_tiles);
    state_update.pop_front(key_value_tiles);
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
        auto& current = chunk == 0 ? state : state_ring;
        const bool last = chunk == num_chunks - 1;

        kd.wait_front(chunk_key_tiles);
        v_beta.wait_front(chunk_value_tiles);
        t_inv.wait_front(chunk_chunk_tiles);
        k_decay_transposed.wait_front(key_chunk_tiles);
        final_decay.wait_front(Kt);
        compute_value_new<ChunkInputPolicy::RETAIN, Ct, Kt, Vt, dfb::scratch, dfb::value_new, 2>(
            current, kd, v_beta, t_inv, scratch, value_new, scratch);
        update_state<ChunkInputPolicy::RETAIN, Ct, Kt, pair_columns>(
            current, last ? final_state : state_ring, scratch, k_decay_transposed, final_decay);
        if (split_chunk != 0 && chunk + 1 == split_chunk) {
            state_ring.wait_front(key_pair_tiles);
            pack_reconfig_data_format(summary_head_output.get_id());
            extract_paths<true, Kt, Vt, 1, dfb::state_ring, dfb::summary_head_output>(state_ring, summary_head_output);
            extract_paths<false, Kt, Vt, 1, dfb::state_ring, dfb::summary_head_state>(state_ring, summary_head_state);
            pack_reconfig_data_format(dfb::state_ring);
            // Restart from zero and identity so the tail summary is independent
            // of the head transition just saved above.
            state.wait_front(key_pair_tiles);
            copy<key_pair_tiles, key_pair_tiles, dfb::state, dfb::state_ring>(state, state_ring);
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
    extract_paths<true, Kt, Vt, packet_rows, dfb::final_state, dfb::output>(final_state, output);
    extract_paths<false, Kt, Vt, packet_rows, dfb::final_state, dfb::transport_state>(final_state, transport_state);
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
        auto& current_state = chunk == 0 ? state : state_ring;
        if (reset_chunk != 0 && chunk == reset_chunk) {
            state_ring.wait_front(key_value_tiles);
            tail_entry_states.wait_front(key_value_tiles);
            copy<key_value_tiles, key_value_tiles, dfb::tail_entry_states, dfb::state_ring>(
                tail_entry_states, state_ring);
            state_ring.pop_front(key_value_tiles);
            tail_entry_states.pop_front(key_value_tiles);
        }
        auto& destination = chunk == num_chunks - 1 ? final_state : state_ring;

        compute_value_new<ChunkInputPolicy::CONSUME, Ct, Kt, Vt, dfb::scratch, dfb::output_intermediate>(
            current_state, kd, v_beta, t_inv, scratch, output_intermediate, value_new);
        compute_chunk_output<Ct, Kt, Vt, dfb::output_intermediate, dfb::scratch, dfb::output>(
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
