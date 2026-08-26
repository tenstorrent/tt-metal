// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

template <uint32_t Ct, uint32_t Kt, uint32_t Vt>
TT_KERNEL void writer(uint32_t work_item_start, uint32_t work_item_count) {
    constexpr uint32_t cc = Ct * Ct;
    constexpr uint32_t ck = Ct * Kt;
    constexpr uint32_t cv = Ct * Vt;
    constexpr uint32_t kc = Kt * Ct;

    const auto v_beta_accessor = TensorAccessor(tensor::v_beta_output);
    const auto kd_accessor = TensorAccessor(tensor::kd_output);
    const auto q_decay_accessor = TensorAccessor(tensor::q_decay_output);
    const auto intra_accessor = TensorAccessor(tensor::intra_output);
    const auto k_decay_transposed_accessor = TensorAccessor(tensor::k_decay_transposed_output);
    const auto final_decay_accessor = TensorAccessor(tensor::final_decay_output);
    const auto t_inv_accessor = TensorAccessor(tensor::t_inv_output);
    DataflowBuffer v_beta(dfb::v_beta);
    DataflowBuffer kd(dfb::w);
    DataflowBuffer q_decay(dfb::q_decay);
    DataflowBuffer intra(dfb::intra);
    DataflowBuffer k_decay_transposed(dfb::k_decay_transposed);
    DataflowBuffer final_decay(dfb::v_new);
    DataflowBuffer t_inv(dfb::t_inv);
    Noc noc;

    auto drain = [&](DataflowBuffer& buffer, const auto& accessor, uint32_t tiles, uint32_t base) {
        buffer.wait_front(tiles);
        for (uint32_t tile = 0; tile < tiles; ++tile) {
            noc.async_write(
                buffer,
                accessor,
                buffer.get_entry_size(),
                {.offset_bytes = tile * buffer.get_entry_size()},
                {.page_id = base + tile});
        }
        noc.async_write_barrier();
        buffer.pop_front(tiles);
    };
    for (uint32_t index = 0; index < work_item_count; ++index) {
        const uint32_t work_item = work_item_start + index;
        drain(v_beta, v_beta_accessor, cv, work_item * cv);
        drain(t_inv, t_inv_accessor, cc, work_item * cc);
        drain(kd, kd_accessor, ck, work_item * ck);
        drain(intra, intra_accessor, cc, work_item * cc);
        drain(q_decay, q_decay_accessor, ck, work_item * ck);
        drain(k_decay_transposed, k_decay_transposed_accessor, kc, work_item * kc);
        drain(final_decay, final_decay_accessor, Kt, work_item * Kt);
    }
}
