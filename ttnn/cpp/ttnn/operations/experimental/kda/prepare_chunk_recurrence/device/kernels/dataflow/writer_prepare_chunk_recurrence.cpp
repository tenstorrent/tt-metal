// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/operations/experimental/kda/device/kernels/chronology.hpp"

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

template <uint32_t Ct, uint32_t Kt, uint32_t Vt, uint32_t max_items, uint32_t batch_items>
TT_KERNEL void writer(uint32_t work_item_start, uint32_t work_item_count, uint32_t num_chunks) {
    DataflowBuffer control(dfb::chronology_writer);
    control.wait_front(1);
    const uint32_t valid_chunks =
        kda_chronology::load(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(control.get_read_ptr())).chunk_count();
    control.pop_front(1);
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
    DataflowBuffer kd(dfb::kd);
    DataflowBuffer q_decay(dfb::q_decay);
    DataflowBuffer intra(dfb::intra);
    DataflowBuffer k_decay_transposed(dfb::k_decay_transposed);
    DataflowBuffer final_decay(dfb::final_decay);
    DataflowBuffer t_inv(dfb::t_inv);
    Noc noc;

    // Write `items` work items' tiles, `tiles` each, published together, to their pages.
    auto drain_items =
        [&](auto& buffer, const auto& accessor, uint32_t tiles, const uint32_t* work_items, uint32_t items) {
            buffer.wait_front(items * tiles);
            for (uint32_t item = 0; item < items; ++item) {
                for (uint32_t tile = 0; tile < tiles; ++tile) {
                    noc.async_write(
                        buffer,
                        accessor,
                        buffer.get_entry_size(),
                        {.offset_bytes = (item * tiles + tile) * buffer.get_entry_size()},
                        {.page_id = work_items[item] * tiles + tile});
                }
            }
            noc.async_write_barrier();
            buffer.pop_front(items * tiles);
        };
    auto drain = [&](auto& buffer, const auto& accessor, uint32_t tiles, uint32_t work_item) {
        drain_items(buffer, accessor, tiles, &work_item, 1);
    };
    // Compute runs the valid work items in batches of batch_items (those past the last whole batch one at a time) and
    // publishes each output for a whole batch, in this order.
    uint32_t valid_items = 0;
    for (uint32_t index = 0; index < work_item_count; ++index) {
        if ((work_item_start + index) % num_chunks < valid_chunks) {
            ++valid_items;
        }
    }
    uint32_t next = 0;
    for (uint32_t done = 0; done < valid_items;) {
        const uint32_t items = valid_items - done >= batch_items ? batch_items : 1;
        uint32_t batch[batch_items];
        for (uint32_t item = 0; item < items; ++next) {
            if ((work_item_start + next) % num_chunks < valid_chunks) {
                batch[item++] = work_item_start + next;
            }
        }
        drain_items(final_decay, final_decay_accessor, Kt, batch, items);
        drain_items(q_decay, q_decay_accessor, ck, batch, items);
        drain_items(v_beta, v_beta_accessor, cv, batch, items);
        drain_items(kd, kd_accessor, ck, batch, items);
        drain_items(intra, intra_accessor, cc, batch, items);
        drain_items(k_decay_transposed, k_decay_transposed_accessor, kc, batch, items);
        done += items;
    }
    // Compute inverts every item's Akk together after the per-item terms, padded to max_items.
    uint32_t inverse_items = 0;
    for (uint32_t index = 0; index < work_item_count; ++index) {
        const uint32_t work_item = work_item_start + index;
        if (work_item % num_chunks < valid_chunks) {
            drain(t_inv, t_inv_accessor, cc, work_item);
            ++inverse_items;
        }
    }
    if (inverse_items != 0 && inverse_items != max_items) {
        t_inv.wait_front((max_items - inverse_items) * cc);
        t_inv.pop_front((max_items - inverse_items) * cc);
    }
}
