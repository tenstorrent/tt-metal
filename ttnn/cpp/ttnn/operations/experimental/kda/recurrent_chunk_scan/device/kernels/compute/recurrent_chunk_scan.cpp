// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Phase B (scan) compute kernel: the sequential-over-chunk recurrence for one head.

#include <cstdint>

#include "api/compute/bcast.h"
#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/matmul.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

inline __attribute__((always_inline)) void matrix_multiply(
    uint32_t a_id,
    uint32_t b_id,
    uint32_t output_id,
    DataflowBuffer& output,
    uint32_t rows,
    uint32_t inner,
    uint32_t columns,
    bool transpose_b) {
    output.reserve_back(rows * columns);
    pack_reconfig_data_format(output_id);
    reconfig_data_format(b_id, a_id);
    matmul_init(a_id, b_id, transpose_b ? 1 : 0);
    for (uint32_t row = 0; row < rows; row++) {
        for (uint32_t column = 0; column < columns; column++) {
            tile_regs_acquire();
            for (uint32_t index = 0; index < inner; index++) {
                const uint32_t b_index = transpose_b ? (column * inner + index) : (index * columns + column);
                matmul_tiles(a_id, b_id, row * inner + index, b_index, 0);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, output_id, row * columns + column);
            tile_regs_release();
        }
    }
    output.push_back(rows * columns);
}

inline __attribute__((always_inline)) void elementwise(
    uint32_t a_id, uint32_t b_id, uint32_t output_id, DataflowBuffer& output, uint32_t count, uint32_t operation) {
    output.reserve_back(count);
    pack_reconfig_data_format(output_id);
    reconfig_data_format(a_id, b_id);
    if (operation == 0) {
        add_init(a_id, b_id);
    } else if (operation == 1) {
        sub_init(a_id, b_id);
    } else {
        mul_init(a_id, b_id);
    }
    for (uint32_t index = 0; index < count; index++) {
        tile_regs_acquire();
        if (operation == 0) {
            add_tiles(a_id, b_id, index, index, 0);
        } else if (operation == 1) {
            sub_tiles(a_id, b_id, index, index, 0);
        } else {
            mul_tiles(a_id, b_id, index, index, 0);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, output_id, index);
        tile_regs_release();
    }
    output.push_back(count);
}

inline __attribute__((always_inline)) void multiply_by_decay(
    uint32_t state_id,
    uint32_t decay_id,
    uint32_t output_id,
    DataflowBuffer& output,
    uint32_t key_tiles,
    uint32_t value_tiles) {
    output.reserve_back(key_tiles * value_tiles);
    pack_reconfig_data_format(output_id);
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

inline __attribute__((always_inline)) void summary_step(
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
    DataflowBuffer& state_temporary,
    uint32_t chunk_tiles,
    uint32_t key_tiles,
    uint32_t value_tiles) {
    const uint32_t cv = chunk_tiles * value_tiles;
    const uint32_t kv = key_tiles * value_tiles;
    current_state.wait_front(kv);
    matrix_multiply(dfb::kd, current_state.get_id(), dfb::scratch, scratch, chunk_tiles, key_tiles, value_tiles, false);
    scratch.wait_front(cv);
    elementwise(dfb::v_beta, dfb::scratch, dfb::value_new, value_new, cv, 1);
    value_new.wait_front(cv);
    scratch.pop_front(cv);
    matrix_multiply(dfb::t_inv, dfb::value_new, dfb::scratch, scratch, chunk_tiles, chunk_tiles, value_tiles, false);
    scratch.wait_front(cv);
    value_new.pop_front(cv);
    matrix_multiply(
        dfb::k_decay_transposed,
        dfb::scratch,
        dfb::state_update,
        state_update,
        key_tiles,
        chunk_tiles,
        value_tiles,
        false);
    state_update.wait_front(kv);
    scratch.pop_front(cv);
    multiply_by_decay(
        current_state.get_id(), dfb::final_decay, dfb::state_temporary, state_temporary, key_tiles, value_tiles);
    state_temporary.wait_front(kv);
    current_state.pop_front(kv);
    elementwise(dfb::state_temporary, dfb::state_update, destination.get_id(), destination, kv, 0);
    state_temporary.pop_front(kv);
    state_update.pop_front(kv);
}

template <uint32_t Ct, uint32_t Kt, uint32_t Vt, uint32_t summary_pair>
TT_KERNEL void compute(uint32_t num_chunks) {
    DataflowBuffer state(dfb::state);
    DataflowBuffer t_inv(dfb::t_inv);
    DataflowBuffer v_beta(dfb::v_beta);
    DataflowBuffer kd(dfb::kd);
    DataflowBuffer q_decay(dfb::q_decay);
    DataflowBuffer intra(dfb::intra);
    DataflowBuffer state_two(dfb::state_two);
    DataflowBuffer value_new(dfb::value_new);
    DataflowBuffer final_decay(dfb::final_decay);
    DataflowBuffer output(dfb::output);
    DataflowBuffer output_intermediate(dfb::output_intermediate);
    DataflowBuffer k_decay_transposed(dfb::k_decay_transposed);
    DataflowBuffer state_update(dfb::state_update);
    DataflowBuffer state_temporary(dfb::state_temporary);
    DataflowBuffer final_state(dfb::final_state);
    DataflowBuffer scratch(dfb::scratch);
    DataflowBuffer summary_raw(dfb::summary_raw);
    DataflowBuffer state_three(dfb::state_three);

    constexpr uint32_t cc = Ct * Ct;
    constexpr uint32_t ck = Ct * Kt;
    constexpr uint32_t cv = Ct * Vt;
    constexpr uint32_t kv = Kt * Vt;
    constexpr uint32_t kc = Kt * Ct;

    compute_kernel_hw_startup(kd.get_id(), v_beta.get_id(), output.get_id());

    if constexpr (summary_pair) {
        for (uint32_t chunk = 0; chunk < num_chunks; chunk++) {
            DataflowBuffer* current_b = chunk == 0 ? &state : ((chunk & 1U) ? &state_two : &state_three);
            DataflowBuffer* next_b = (chunk & 1U) ? &state_three : &state_two;
            DataflowBuffer* current_ab = chunk == 0 ? &q_decay : ((chunk & 1U) ? &intra : &output_intermediate);
            DataflowBuffer* next_ab = (chunk & 1U) ? &output_intermediate : &intra;
            const bool last = chunk == num_chunks - 1;

            kd.wait_front(ck);
            v_beta.wait_front(cv);
            t_inv.wait_front(cc);
            k_decay_transposed.wait_front(kc);
            final_decay.wait_front(Kt);
            summary_step(
                *current_b,
                last ? final_state : *next_b,
                kd,
                v_beta,
                t_inv,
                k_decay_transposed,
                final_decay,
                scratch,
                value_new,
                state_update,
                state_temporary,
                Ct,
                Kt,
                Vt);
            summary_step(
                *current_ab,
                last ? summary_raw : *next_ab,
                kd,
                v_beta,
                t_inv,
                k_decay_transposed,
                final_decay,
                scratch,
                value_new,
                state_update,
                state_temporary,
                Ct,
                Kt,
                Vt);
            kd.pop_front(ck);
            v_beta.pop_front(cv);
            t_inv.pop_front(cc);
            k_decay_transposed.pop_front(kc);
            final_decay.pop_front(Kt);
        }
        summary_raw.wait_front(kv);
        final_state.wait_front(kv);
        elementwise(dfb::summary_raw, dfb::final_state, dfb::output, output, kv, 1);
        summary_raw.pop_front(kv);
    } else {
        for (uint32_t chunk = 0; chunk < num_chunks; chunk++) {
            DataflowBuffer* current_state = chunk == 0 ? &state : ((chunk & 1U) ? &state_two : &state_three);
            DataflowBuffer* next_state = (chunk & 1U) ? &state_three : &state_two;
            DataflowBuffer& destination = chunk == num_chunks - 1 ? final_state : *next_state;

            kd.wait_front(ck);
            current_state->wait_front(kv);
            matrix_multiply(dfb::kd, current_state->get_id(), dfb::scratch, scratch, Ct, Kt, Vt, false);
            scratch.wait_front(cv);
            kd.pop_front(ck);
            v_beta.wait_front(cv);
            elementwise(dfb::v_beta, dfb::scratch, dfb::output_intermediate, output_intermediate, cv, 1);
            output_intermediate.wait_front(cv);
            v_beta.pop_front(cv);
            scratch.pop_front(cv);
            t_inv.wait_front(cc);
            matrix_multiply(dfb::t_inv, dfb::output_intermediate, dfb::value_new, value_new, Ct, Ct, Vt, false);
            value_new.wait_front(cv);
            t_inv.pop_front(cc);
            output_intermediate.pop_front(cv);

            q_decay.wait_front(ck);
            matrix_multiply(
                dfb::q_decay,
                current_state->get_id(),
                dfb::output_intermediate,
                output_intermediate,
                Ct,
                Kt,
                Vt,
                false);
            output_intermediate.wait_front(cv);
            q_decay.pop_front(ck);
            intra.wait_front(cc);
            matrix_multiply(dfb::intra, dfb::value_new, dfb::scratch, scratch, Ct, Ct, Vt, false);
            scratch.wait_front(cv);
            intra.pop_front(cc);
            elementwise(dfb::output_intermediate, dfb::scratch, dfb::output, output, cv, 0);
            output_intermediate.pop_front(cv);
            scratch.pop_front(cv);

            k_decay_transposed.wait_front(kc);
            matrix_multiply(
                dfb::k_decay_transposed, dfb::value_new, dfb::state_update, state_update, Kt, Ct, Vt, false);
            state_update.wait_front(kv);
            k_decay_transposed.pop_front(kc);
            value_new.pop_front(cv);

            final_decay.wait_front(Kt);
            multiply_by_decay(current_state->get_id(), dfb::final_decay, dfb::state_temporary, state_temporary, Kt, Vt);
            state_temporary.wait_front(kv);
            final_decay.pop_front(Kt);
            current_state->pop_front(kv);
            elementwise(dfb::state_temporary, dfb::state_update, destination.get_id(), destination, kv, 0);
            state_temporary.pop_front(kv);
            state_update.pop_front(kv);
        }
    }
}
