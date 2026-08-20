// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// KDA scan reader: the initial state S [K,V] once, then vector-decay prep
// intermediates v_beta, kd, q_decay, intra, k_dec_t, dl[K,1], t_inv. FP32 by default; selected intermediates may be
// BF16.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

template <uint32_t Ct, uint32_t Kt, uint32_t Vt, uint32_t Vt_full, uint32_t summary_pair>
TT_KERNEL void reader(uint32_t head, uint32_t value_block, uint32_t num_chunks) {
    const auto v_beta_accessor = TensorAccessor(tensor::v_beta);
    const auto kd_accessor = TensorAccessor(tensor::kd);
    const auto k_decay_transposed_accessor = TensorAccessor(tensor::k_decay_transposed);
    const auto final_decay_accessor = TensorAccessor(tensor::final_decay);
    const auto t_inv_accessor = TensorAccessor(tensor::t_inv);

    DataflowBuffer state(dfb::state);
    DataflowBuffer t_inv(dfb::t_inv);
    DataflowBuffer v_beta(dfb::v_beta);
    DataflowBuffer kd(dfb::kd);
    DataflowBuffer q_decay(dfb::q_decay);
    DataflowBuffer intra(dfb::intra);
    DataflowBuffer summary_seed(dfb::summary_seed);
    DataflowBuffer k_decay_transposed(dfb::k_decay_transposed);
    DataflowBuffer final_decay(dfb::final_decay);
    Noc noc;

    constexpr uint32_t cc = Ct * Ct;
    constexpr uint32_t ck = Ct * Kt;
    constexpr uint32_t kc = Kt * Ct;
    constexpr uint32_t kv = Kt * Vt;

    auto read_into = [&](const auto& accessor, DataflowBuffer& buffer, uint32_t base, uint32_t count) {
        buffer.reserve_back(count);
        const uint32_t entry_size = buffer.get_entry_size();
        for (uint32_t tile = 0; tile < count; tile++) {
            noc.async_read(accessor, buffer, entry_size, {.page_id = base + tile}, {.offset_bytes = tile * entry_size});
        }
        noc.async_read_barrier();
        buffer.push_back(count);
    };

    auto read_value_slice = [&](const auto& accessor, DataflowBuffer& buffer, uint32_t row_base, uint32_t rows) {
        buffer.reserve_back(rows * Vt);
        const uint32_t entry_size = buffer.get_entry_size();
        for (uint32_t row = 0; row < rows; row++) {
            const uint32_t source = row_base + row * Vt_full + value_block * Vt;
            const uint32_t destination = row * Vt;
            for (uint32_t value_tile = 0; value_tile < Vt; value_tile++) {
                noc.async_read(
                    accessor,
                    buffer,
                    entry_size,
                    {.page_id = source + value_tile},
                    {.offset_bytes = (destination + value_tile) * entry_size});
            }
        }
        noc.async_read_barrier();
        buffer.push_back(rows * Vt);
    };

    if constexpr (summary_pair) {
        auto seed_identity = [&](DataflowBuffer& buffer) {
            constexpr uint32_t one_fp32 = 0x3F800000;
            constexpr uint32_t face_elements = 16 * 16;
            constexpr uint32_t tile_elements = 4 * face_elements;
            buffer.reserve_back(kv);
            {
                auto lock = buffer.scoped_write_lock(kv);
                auto state_ptr = lock.get_ptr<volatile uint32_t>();
                for (uint32_t index = 0; index < kv * tile_elements; ++index) {
                    state_ptr[index] = 0;
                }
                for (uint32_t local_col = 0; local_col < Vt; local_col++) {
                    const uint32_t global_col = value_block * Vt + local_col;
                    if (global_col < Kt) {
                        auto tile = state_ptr + (global_col * Vt + local_col) * tile_elements;
                        for (uint32_t row = 0; row < 16; ++row) {
                            tile[row * 16 + row] = one_fp32;
                            tile[3 * face_elements + row * 16 + row] = one_fp32;
                        }
                    }
                }
            }
            buffer.push_back(kv);
        };

        state.reserve_back(kv);
        noc.async_write_zeros(state, kv * state.get_entry_size());
        noc.write_zeros_l1_barrier();
        state.push_back(kv);
        seed_identity(summary_seed);
    } else {
        const auto initial_state_accessor = TensorAccessor(tensor::initial_state);
        read_value_slice(initial_state_accessor, state, head * Kt * Vt_full, Kt);
    }

    for (uint32_t chunk = 0; chunk < num_chunks; chunk++) {
        const uint32_t head_chunk = head * num_chunks + chunk;
        read_value_slice(v_beta_accessor, v_beta, head_chunk * Ct * Vt_full, Ct);
        read_into(kd_accessor, kd, head_chunk * ck, ck);
        if constexpr (!summary_pair) {
            const auto q_decay_accessor = TensorAccessor(tensor::q_decay);
            const auto intra_accessor = TensorAccessor(tensor::intra);
            read_into(q_decay_accessor, q_decay, head_chunk * ck, ck);
            read_into(intra_accessor, intra, head_chunk * cc, cc);
        }
        read_into(k_decay_transposed_accessor, k_decay_transposed, head_chunk * kc, kc);
        read_into(final_decay_accessor, final_decay, head_chunk * Kt, Kt);
        read_into(t_inv_accessor, t_inv, head_chunk * cc, cc);
    }
}
