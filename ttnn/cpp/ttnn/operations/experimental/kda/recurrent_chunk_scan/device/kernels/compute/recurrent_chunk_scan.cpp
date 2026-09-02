// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Phase B (scan) compute kernel: the sequential-over-chunk recurrence for one
// head.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/matmul.h"
#include "api/compute/reconfig_data_format.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/operations/experimental/kda/device/kernels/compute/matmul_subblock.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"

namespace ckl = compute_kernel_lib;

enum class ChunkInputPolicy { RETAIN, CONSUME };

constexpr uint32_t max_dst_tiles =
    ckernel::get_dest_max_tiles<DST_SYNC_MODE, DST_ACCUM_MODE, ckernel::DstTileShape::Tile32x32>();

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

template <
    ChunkInputPolicy InputPolicy,
    uint32_t Ct,
    uint32_t Kt,
    uint32_t Vt,
    uint32_t StateProjectionDfb,
    uint32_t DifferenceDfb>
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
    if constexpr (InputPolicy == ChunkInputPolicy::CONSUME) {
        kd.pop_front(chunk_key_tiles);
    }
    if constexpr (InputPolicy == ChunkInputPolicy::CONSUME) {
        v_beta.wait_front(chunk_value_tiles);
    }
    ckl::sub<
        ckl::input(dfb::v_beta, ckl::WaitPolicy::None, ckl::PopPolicy::None, ckl::InputTileMapping::Block),
        ckl::input(StateProjectionDfb, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd, ckl::InputTileMapping::Block),
        ckl::output(DifferenceDfb, ckl::ReservePolicy::Upfront, ckl::PushPolicy::AtEnd)>(
        ckl::IterationShape::tiles(chunk_value_tiles).block_size(max_dst_tiles));
    difference.wait_front(chunk_value_tiles);
    if constexpr (InputPolicy == ChunkInputPolicy::CONSUME) {
        v_beta.pop_front(chunk_value_tiles);
    }
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

template <
    uint32_t Ct,
    uint32_t Kt,
    uint32_t Vt,
    uint32_t StateProjectionDfb,
    uint32_t ValueProjectionDfb,
    uint32_t OutputDfb>
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
    q_decay.pop_front(chunk_key_tiles);
    intra.wait_front(chunk_chunk_tiles);
    matrix_multiply<Ct, Ct, Vt>(intra, corrected_value, value_projection);
    intra.pop_front(chunk_chunk_tiles);
    ckl::add<
        ckl::input(StateProjectionDfb, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd, ckl::InputTileMapping::Block),
        ckl::input(ValueProjectionDfb, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd, ckl::InputTileMapping::Block),
        ckl::output(OutputDfb, ckl::ReservePolicy::Upfront, ckl::PushPolicy::AtEnd)>(
        ckl::IterationShape::tiles(chunk_value_tiles).block_size(max_dst_tiles));
}

template <ChunkInputPolicy InputPolicy, uint32_t Ct, uint32_t Kt, uint32_t Vt>
FORCE_INLINE void compute_state_update(
    DataflowBuffer& corrected_value, DataflowBuffer& k_decay_transposed, DataflowBuffer& state_update) {
    constexpr uint32_t chunk_value_tiles = Ct * Vt;
    constexpr uint32_t key_value_tiles = Kt * Vt;
    constexpr uint32_t key_chunk_tiles = Kt * Ct;

    if constexpr (InputPolicy == ChunkInputPolicy::CONSUME) {
        k_decay_transposed.wait_front(key_chunk_tiles);
    }
    matrix_multiply<Kt, Ct, Vt>(k_decay_transposed, corrected_value, state_update);
    if constexpr (InputPolicy == ChunkInputPolicy::CONSUME) {
        k_decay_transposed.pop_front(key_chunk_tiles);
    }
    corrected_value.pop_front(chunk_value_tiles);
}

template <ChunkInputPolicy InputPolicy, uint32_t Kt, uint32_t Vt, uint32_t CurrentStateDfb, uint32_t DestinationDfb>
FORCE_INLINE void finish_state_update(DataflowBuffer& final_decay, DataflowBuffer& state_update) {
    constexpr uint32_t key_value_tiles = Kt * Vt;

    if constexpr (InputPolicy == ChunkInputPolicy::CONSUME) {
        final_decay.wait_front(Kt);
    }
    ckl::mul<
        ckl::input(CurrentStateDfb, ckl::WaitPolicy::None, ckl::PopPolicy::AtEnd, ckl::InputTileMapping::Block),
        ckl::input(
            dfb::final_decay,
            ckl::BroadcastDim::Col,
            ckl::WaitPolicy::None,
            ckl::PopPolicy::None,
            ckl::InputTileMapping::Col),
        ckl::output(dfb::state_temporary, ckl::ReservePolicy::Upfront, ckl::PushPolicy::AtEnd)>(
        ckl::IterationShape::grid(Kt, Vt).block_size(max_dst_tiles));
    if constexpr (InputPolicy == ChunkInputPolicy::CONSUME) {
        final_decay.pop_front(Kt);
    }
    ckl::add<
        ckl::input(dfb::state_temporary, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd, ckl::InputTileMapping::Block),
        ckl::input(dfb::state_update, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd, ckl::InputTileMapping::Block),
        ckl::output(DestinationDfb, ckl::ReservePolicy::Upfront, ckl::PushPolicy::AtEnd)>(
        ckl::IterationShape::tiles(key_value_tiles).block_size(max_dst_tiles));
}

template <uint32_t Ct, uint32_t Kt, uint32_t Vt>
FORCE_INLINE void compute_summary(uint32_t num_chunks) {
    DataflowBuffer state(dfb::state);
    DataflowBuffer t_inv(dfb::t_inv);
    DataflowBuffer v_beta(dfb::v_beta);
    DataflowBuffer kd(dfb::kd);
    DataflowBuffer state_ring(dfb::state_ring);
    DataflowBuffer value_new(dfb::value_new);
    DataflowBuffer final_decay(dfb::final_decay);
    DataflowBuffer output(dfb::output);
    DataflowBuffer k_decay_transposed(dfb::k_decay_transposed);
    DataflowBuffer state_update(dfb::state_update);
    DataflowBuffer final_state(dfb::final_state);
    DataflowBuffer scratch(dfb::scratch);
    DataflowBuffer summary_raw(dfb::summary_raw);
    DataflowBuffer summary_seed(dfb::summary_seed);
    DataflowBuffer summary_ring(dfb::summary_ring);

    constexpr uint32_t chunk_chunk_tiles = Ct * Ct;
    constexpr uint32_t chunk_key_tiles = Ct * Kt;
    constexpr uint32_t chunk_value_tiles = Ct * Vt;
    constexpr uint32_t key_value_tiles = Kt * Vt;
    constexpr uint32_t key_chunk_tiles = Kt * Ct;

    compute_kernel_hw_startup<SrcOrder::Reverse>(kd.get_id(), v_beta.get_id(), output.get_id());
    for (uint32_t chunk = 0; chunk < num_chunks; chunk++) {
        DataflowBuffer& current_b = chunk == 0 ? state : state_ring;
        DataflowBuffer& current_ab = chunk == 0 ? summary_seed : summary_ring;
        const bool last = chunk == num_chunks - 1;

        kd.wait_front(chunk_key_tiles);
        v_beta.wait_front(chunk_value_tiles);
        t_inv.wait_front(chunk_chunk_tiles);
        k_decay_transposed.wait_front(key_chunk_tiles);
        final_decay.wait_front(Kt);
        compute_value_new<ChunkInputPolicy::RETAIN, Ct, Kt, Vt, dfb::scratch, dfb::value_new>(
            current_b, kd, v_beta, t_inv, scratch, value_new, scratch);
        compute_state_update<ChunkInputPolicy::RETAIN, Ct, Kt, Vt>(scratch, k_decay_transposed, state_update);
        if (chunk == 0) {
            if (last) {
                finish_state_update<ChunkInputPolicy::RETAIN, Kt, Vt, dfb::state, dfb::final_state>(
                    final_decay, state_update);
            } else {
                finish_state_update<ChunkInputPolicy::RETAIN, Kt, Vt, dfb::state, dfb::state_ring>(
                    final_decay, state_update);
            }
        } else if (last) {
            finish_state_update<ChunkInputPolicy::RETAIN, Kt, Vt, dfb::state_ring, dfb::final_state>(
                final_decay, state_update);
        } else {
            finish_state_update<ChunkInputPolicy::RETAIN, Kt, Vt, dfb::state_ring, dfb::state_ring>(
                final_decay, state_update);
        }
        compute_value_new<ChunkInputPolicy::RETAIN, Ct, Kt, Vt, dfb::scratch, dfb::value_new>(
            current_ab, kd, v_beta, t_inv, scratch, value_new, scratch);
        compute_state_update<ChunkInputPolicy::RETAIN, Ct, Kt, Vt>(scratch, k_decay_transposed, state_update);
        if (chunk == 0) {
            if (last) {
                finish_state_update<ChunkInputPolicy::RETAIN, Kt, Vt, dfb::summary_seed, dfb::summary_raw>(
                    final_decay, state_update);
            } else {
                finish_state_update<ChunkInputPolicy::RETAIN, Kt, Vt, dfb::summary_seed, dfb::summary_ring>(
                    final_decay, state_update);
            }
        } else if (last) {
            finish_state_update<ChunkInputPolicy::RETAIN, Kt, Vt, dfb::summary_ring, dfb::summary_raw>(
                final_decay, state_update);
        } else {
            finish_state_update<ChunkInputPolicy::RETAIN, Kt, Vt, dfb::summary_ring, dfb::summary_ring>(
                final_decay, state_update);
        }
        kd.pop_front(chunk_key_tiles);
        v_beta.pop_front(chunk_value_tiles);
        t_inv.pop_front(chunk_chunk_tiles);
        k_decay_transposed.pop_front(key_chunk_tiles);
        final_decay.pop_front(Kt);
    }
    summary_raw.wait_front(key_value_tiles);
    final_state.wait_front(key_value_tiles);
    ckl::sub<
        ckl::input(dfb::summary_raw, ckl::WaitPolicy::None, ckl::PopPolicy::AtEnd, ckl::InputTileMapping::Block),
        ckl::input(dfb::final_state, ckl::WaitPolicy::None, ckl::PopPolicy::None, ckl::InputTileMapping::Block),
        ckl::output(dfb::output, ckl::ReservePolicy::Upfront, ckl::PushPolicy::AtEnd)>(
        ckl::IterationShape::tiles(key_value_tiles).block_size(max_dst_tiles));
}

template <uint32_t Ct, uint32_t Kt, uint32_t Vt>
FORCE_INLINE void compute_recurrent(uint32_t num_chunks) {
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
    DataflowBuffer state_update(dfb::state_update);
    DataflowBuffer final_state(dfb::final_state);
    DataflowBuffer scratch(dfb::scratch);

    compute_kernel_hw_startup<SrcOrder::Reverse>(kd.get_id(), v_beta.get_id(), output.get_id());
    pack_reconfig_data_format(dfb::scratch);
    for (uint32_t chunk = 0; chunk < num_chunks; chunk++) {
        DataflowBuffer& current_state = chunk == 0 ? state : state_ring;
        compute_value_new<ChunkInputPolicy::CONSUME, Ct, Kt, Vt, dfb::scratch, dfb::output_intermediate>(
            current_state, kd, v_beta, t_inv, scratch, output_intermediate, value_new);
        compute_chunk_output<Ct, Kt, Vt, dfb::output_intermediate, dfb::scratch, dfb::output>(
            current_state, value_new, q_decay, intra, output_intermediate, scratch, output);

        pack_reconfig_data_format(dfb::state_update);
        compute_state_update<ChunkInputPolicy::CONSUME, Ct, Kt, Vt>(value_new, k_decay_transposed, state_update);
        if (chunk == 0) {
            if (chunk + 1 == num_chunks) {
                finish_state_update<ChunkInputPolicy::CONSUME, Kt, Vt, dfb::state, dfb::final_state>(
                    final_decay, state_update);
            } else {
                finish_state_update<ChunkInputPolicy::CONSUME, Kt, Vt, dfb::state, dfb::state_ring>(
                    final_decay, state_update);
            }
        } else if (chunk + 1 == num_chunks) {
            finish_state_update<ChunkInputPolicy::CONSUME, Kt, Vt, dfb::state_ring, dfb::final_state>(
                final_decay, state_update);
        } else {
            finish_state_update<ChunkInputPolicy::CONSUME, Kt, Vt, dfb::state_ring, dfb::state_ring>(
                final_decay, state_update);
        }
    }
}

template <uint32_t Ct, uint32_t Kt, uint32_t Vt, uint32_t summary_pair>
TT_KERNEL void compute(uint32_t num_chunks) {
    if constexpr (summary_pair) {
        compute_summary<Ct, Kt, Vt>(num_chunks);
    } else {
        compute_recurrent<Ct, Kt, Vt>(num_chunks);
    }
}
