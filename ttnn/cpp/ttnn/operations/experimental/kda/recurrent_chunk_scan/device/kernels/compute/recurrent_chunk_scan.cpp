// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
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

enum class ElementwiseOperation { ADD, SUBTRACT, MULTIPLY };

FORCE_INLINE constexpr uint32_t largest_divisor_at_most(uint32_t value, uint32_t limit) {
    for (uint32_t divisor = limit; divisor > 1; --divisor) {
        if (value % divisor == 0) {
            return divisor;
        }
    }
    return 1;
}

template <uint32_t Mt, uint32_t Kt, uint32_t Nt, bool Tr>
FORCE_INLINE void matrix_multiply(uint32_t a_id, uint32_t b_id, uint32_t output_id, DataflowBuffer& output) {
    constexpr uint32_t dst_tiles =
        ckernel::get_dest_max_tiles<DST_SYNC_MODE, DST_ACCUM_MODE, ckernel::DstTileShape::Tile32x32>();
    constexpr uint32_t subblock_columns = largest_divisor_at_most(Nt, dst_tiles);
    constexpr uint32_t subblock_rows = largest_divisor_at_most(Mt, dst_tiles / subblock_columns);
    static_assert(subblock_rows * subblock_columns <= dst_tiles);

    output.reserve_back(Mt * Nt);
    reconfig_data_format(b_id, a_id);
    matmul_block_init(a_id, b_id, Tr, subblock_columns, subblock_rows, Kt);
    for (uint32_t row_start = 0; row_start < Mt; row_start += subblock_rows) {
        for (uint32_t column_start = 0; column_start < Nt; column_start += subblock_columns) {
            tile_regs_acquire();
            for (uint32_t k = 0; k < Kt; ++k) {
                const uint32_t b_index = Tr ? column_start * Kt + k : k * Nt + column_start;
                matmul_block(a_id, b_id, row_start * Kt + k, b_index, 0, Tr, subblock_columns, subblock_rows, Kt);
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

template <ElementwiseOperation Operation, uint32_t Count>
FORCE_INLINE void elementwise(uint32_t a_id, uint32_t b_id, uint32_t output_id, DataflowBuffer& output) {
    constexpr uint32_t dst_tiles =
        ckernel::get_dest_max_tiles<DST_SYNC_MODE, DST_ACCUM_MODE, ckernel::DstTileShape::Tile32x32>();

    output.reserve_back(Count);
    reconfig_data_format(a_id, b_id);
    if constexpr (Operation == ElementwiseOperation::ADD) {
        add_init(a_id, b_id);
    } else if constexpr (Operation == ElementwiseOperation::SUBTRACT) {
        sub_init(a_id, b_id);
    } else {
        mul_init(a_id, b_id);
    }
    for (uint32_t block_start = 0; block_start < Count; block_start += dst_tiles) {
        const uint32_t remaining = Count - block_start;
        const uint32_t block_tiles = remaining < dst_tiles ? remaining : dst_tiles;
        tile_regs_acquire();
        if constexpr (Operation == ElementwiseOperation::ADD) {
            add_block(a_id, b_id, block_start, block_start, 0, block_tiles);
        } else if constexpr (Operation == ElementwiseOperation::SUBTRACT) {
            sub_block(a_id, b_id, block_start, block_start, 0, block_tiles);
        } else {
            mul_block(a_id, b_id, block_start, block_start, 0, block_tiles);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t tile = 0; tile < block_tiles; ++tile) {
            pack_tile(tile, output_id, block_start + tile);
        }
        tile_regs_release();
    }
    output.push_back(Count);
}

FORCE_INLINE void multiply_by_decay(
    uint32_t state_id,
    uint32_t decay_id,
    uint32_t output_id,
    DataflowBuffer& output,
    uint32_t key_tiles,
    uint32_t value_tiles) {
    output.reserve_back(key_tiles * value_tiles);
    reconfig_data_format(state_id, decay_id);
    mul_bcast_cols_init(state_id, decay_id);
    for (uint32_t key = 0; key < key_tiles; key++) {
        for (uint32_t value = 0; value < value_tiles; value++) {
            const uint32_t index = key * value_tiles + value;
            tile_regs_acquire();
            mul_tiles_bcast_cols(state_id, decay_id, index, key, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, output_id, index);
            tile_regs_release();
        }
    }
    output.push_back(key_tiles * value_tiles);
}

template <uint32_t Ct, uint32_t Kt, uint32_t Vt>
FORCE_INLINE void summary_step(
    DataflowBuffer& current_state,
    DataflowBuffer& destination,
    DataflowBuffer& kd,
    DataflowBuffer& v_beta,
    DataflowBuffer& t_inv,
    DataflowBuffer& k_decay_transposed,
    DataflowBuffer& final_decay,
    DataflowBuffer& scratch,
    DataflowBuffer& value_new,
    DataflowBuffer& state_update,
    DataflowBuffer& state_temporary) {
    constexpr uint32_t cv = Ct * Vt;
    constexpr uint32_t kv = Kt * Vt;
    current_state.wait_front(kv);
    matrix_multiply<Ct, Kt, Vt, false>(dfb::kd, current_state.get_id(), dfb::scratch, scratch);
    scratch.wait_front(cv);
    elementwise<ElementwiseOperation::SUBTRACT, cv>(dfb::v_beta, dfb::scratch, dfb::value_new, value_new);
    value_new.wait_front(cv);
    scratch.pop_front(cv);
    matrix_multiply<Ct, Ct, Vt, false>(dfb::t_inv, dfb::value_new, dfb::scratch, scratch);
    scratch.wait_front(cv);
    value_new.pop_front(cv);
    matrix_multiply<Kt, Ct, Vt, false>(dfb::k_decay_transposed, dfb::scratch, dfb::state_update, state_update);
    state_update.wait_front(kv);
    scratch.pop_front(cv);
    multiply_by_decay(current_state.get_id(), dfb::final_decay, dfb::state_temporary, state_temporary, Kt, Vt);
    state_temporary.wait_front(kv);
    elementwise<ElementwiseOperation::ADD, kv>(
        dfb::state_temporary, dfb::state_update, destination.get_id(), destination);
    current_state.pop_front(kv);
    state_temporary.pop_front(kv);
    state_update.pop_front(kv);
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
    DataflowBuffer state_temporary(dfb::state_temporary);
    DataflowBuffer final_state(dfb::final_state);
    DataflowBuffer scratch(dfb::scratch);
    DataflowBuffer summary_raw(dfb::summary_raw);
    DataflowBuffer summary_seed(dfb::summary_seed);
    DataflowBuffer summary_ring(dfb::summary_ring);

    constexpr uint32_t cc = Ct * Ct;
    constexpr uint32_t ck = Ct * Kt;
    constexpr uint32_t cv = Ct * Vt;
    constexpr uint32_t kv = Kt * Vt;
    constexpr uint32_t kc = Kt * Ct;

    compute_kernel_hw_startup(kd.get_id(), v_beta.get_id(), output.get_id());
    for (uint32_t chunk = 0; chunk < num_chunks; chunk++) {
        DataflowBuffer& current_b = chunk == 0 ? state : state_ring;
        DataflowBuffer& current_ab = chunk == 0 ? summary_seed : summary_ring;
        const bool last = chunk == num_chunks - 1;

        kd.wait_front(ck);
        v_beta.wait_front(cv);
        t_inv.wait_front(cc);
        k_decay_transposed.wait_front(kc);
        final_decay.wait_front(Kt);
        summary_step<Ct, Kt, Vt>(
            current_b,
            last ? final_state : state_ring,
            kd,
            v_beta,
            t_inv,
            k_decay_transposed,
            final_decay,
            scratch,
            value_new,
            state_update,
            state_temporary);
        summary_step<Ct, Kt, Vt>(
            current_ab,
            last ? summary_raw : summary_ring,
            kd,
            v_beta,
            t_inv,
            k_decay_transposed,
            final_decay,
            scratch,
            value_new,
            state_update,
            state_temporary);
        kd.pop_front(ck);
        v_beta.pop_front(cv);
        t_inv.pop_front(cc);
        k_decay_transposed.pop_front(kc);
        final_decay.pop_front(Kt);
    }
    summary_raw.wait_front(kv);
    final_state.wait_front(kv);
    elementwise<ElementwiseOperation::SUBTRACT, kv>(dfb::summary_raw, dfb::final_state, dfb::output, output);
    summary_raw.pop_front(kv);
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
    DataflowBuffer state_temporary(dfb::state_temporary);
    DataflowBuffer final_state(dfb::final_state);
    DataflowBuffer scratch(dfb::scratch);

    constexpr uint32_t cc = Ct * Ct;
    constexpr uint32_t ck = Ct * Kt;
    constexpr uint32_t cv = Ct * Vt;
    constexpr uint32_t kv = Kt * Vt;
    constexpr uint32_t kc = Kt * Ct;

    compute_kernel_hw_startup(kd.get_id(), v_beta.get_id(), output.get_id());
    pack_reconfig_data_format(dfb::scratch);
    for (uint32_t chunk = 0; chunk < num_chunks; chunk++) {
        DataflowBuffer& current_state = chunk == 0 ? state : state_ring;
        DataflowBuffer& destination = chunk == num_chunks - 1 ? final_state : state_ring;

        kd.wait_front(ck);
        current_state.wait_front(kv);
        matrix_multiply<Ct, Kt, Vt, false>(dfb::kd, current_state.get_id(), dfb::scratch, scratch);
        scratch.wait_front(cv);
        kd.pop_front(ck);
        v_beta.wait_front(cv);
        elementwise<ElementwiseOperation::SUBTRACT, cv>(
            dfb::v_beta, dfb::scratch, dfb::output_intermediate, output_intermediate);
        output_intermediate.wait_front(cv);
        v_beta.pop_front(cv);
        scratch.pop_front(cv);
        t_inv.wait_front(cc);
        matrix_multiply<Ct, Ct, Vt, false>(dfb::t_inv, dfb::output_intermediate, dfb::value_new, value_new);
        value_new.wait_front(cv);
        t_inv.pop_front(cc);
        output_intermediate.pop_front(cv);

        q_decay.wait_front(ck);
        matrix_multiply<Ct, Kt, Vt, false>(
            dfb::q_decay, current_state.get_id(), dfb::output_intermediate, output_intermediate);
        output_intermediate.wait_front(cv);
        q_decay.pop_front(ck);
        intra.wait_front(cc);
        matrix_multiply<Ct, Ct, Vt, false>(dfb::intra, dfb::value_new, dfb::scratch, scratch);
        scratch.wait_front(cv);
        intra.pop_front(cc);
        pack_reconfig_data_format(dfb::output);
        elementwise<ElementwiseOperation::ADD, cv>(dfb::output_intermediate, dfb::scratch, dfb::output, output);
        output_intermediate.pop_front(cv);
        scratch.pop_front(cv);

        k_decay_transposed.wait_front(kc);
        pack_reconfig_data_format(dfb::state_update);
        matrix_multiply<Kt, Ct, Vt, false>(dfb::k_decay_transposed, dfb::value_new, dfb::state_update, state_update);
        state_update.wait_front(kv);
        k_decay_transposed.pop_front(kc);
        value_new.pop_front(cv);

        final_decay.wait_front(Kt);
        multiply_by_decay(current_state.get_id(), dfb::final_decay, dfb::state_temporary, state_temporary, Kt, Vt);
        state_temporary.wait_front(kv);
        final_decay.pop_front(Kt);
        elementwise<ElementwiseOperation::ADD, kv>(
            dfb::state_temporary, dfb::state_update, destination.get_id(), destination);
        current_state.pop_front(kv);
        state_temporary.pop_front(kv);
        state_update.pop_front(kv);
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
