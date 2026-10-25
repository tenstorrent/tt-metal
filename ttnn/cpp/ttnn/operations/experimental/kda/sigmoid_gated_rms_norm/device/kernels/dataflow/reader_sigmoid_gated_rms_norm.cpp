// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/cpp/ttnn/kernel/dataflow/generate_bcast_scalar_metal2.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/kda/sigmoid_gated_rms_norm/device/kernels/norm_items.hpp"

// Reads each group's inputs (the writer reads the gate on the other NoC), then the weight after the first group.
// With ACTIVATED_GATE the items of heads [0, Ha) then read their already activated FP32 gate ([B * Ha, T, V], laid out
// as the input): only the group's last stage reads it, so the inputs every core waits on go first.
template <uint32_t Vt, uint32_t epsilon_bits, uint32_t G, uint32_t H, uint32_t Ha, uint32_t Mt, uint32_t max_items>
TT_KERNEL void reader(uint32_t first_item, uint32_t item_stride, uint32_t item_count) {
    const auto x_acc = TensorAccessor(tensor::input);
    const auto w_acc = TensorAccessor(tensor::weight);
    DataflowBuffer x(dfb::x);
    DataflowBuffer weight(dfb::weight);
    DataflowBuffer epsilon(dfb::epsilon);
    Noc noc;
#ifdef ACTIVATED_GATE
    const auto a_acc = TensorAccessor(tensor::activated_gate);
    DataflowBuffer activated(dfb::activated);
#endif
    const NormItems<H, Ha, Mt, max_items> core_items(first_item, item_stride, item_count);
    const uint32_t wi_count = item_count;

    for (uint32_t first = 0; first < wi_count; first += G) {
        const uint32_t items = wi_count - first < G ? wi_count - first : G;
        [[maybe_unused]] const uint32_t raw = core_items.raw_in(first, items);
        // The compute kernel's first stage is the first item's gate sigmoid, which these reads overlap.
        x.reserve_back(items * Vt);
        for (uint32_t j = 0; j < items; j++) {
            const uint32_t x_base = core_items[first + j].wi() * Vt;
            for (uint32_t vt = 0; vt < Vt; vt++) {
                const uint32_t t = j * Vt + vt;
                noc.async_read(
                    x_acc, x, x.get_entry_size(), {.page_id = x_base + vt}, {.offset_bytes = t * x.get_entry_size()});
            }
        }
        if (first == 0) {
            // The reduction scaler and epsilon are first used after the inputs.
            dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
                dfb::scaler,
                ckernel::PoolType::AVG,
                ckernel::ReduceDim::REDUCE_ROW,
                Vt * tt::constants::TILE_WIDTH>();
            generate_bcast_col_scalar(epsilon, epsilon_bits);
        }
        noc.async_read_barrier();
        x.push_back(items * Vt);
        if (first == 0) {
            // The weight is first used once the group's items are scaled. Every core reads the same few weight pages:
            // read before the inputs, they would hold up every core's first item behind the banks that hold them.
            weight.reserve_back(Vt);
            for (uint32_t vt = 0; vt < Vt; vt++) {
                noc.async_read(
                    w_acc,
                    weight,
                    weight.get_entry_size(),
                    {.page_id = vt},
                    {.offset_bytes = vt * weight.get_entry_size()});
            }
            noc.async_read_barrier();
            weight.push_back(Vt);
        }
#ifdef ACTIVATED_GATE
        if (raw < items) {
            // Only the group's last stage, the output product, reads it.
            activated.reserve_back((items - raw) * Vt);
            for (uint32_t j = raw; j < items; j++) {
                const auto item = core_items[first + j];
                const uint32_t base = ((item.b * Ha + item.h) * Mt + item.mt) * Vt;
                for (uint32_t vt = 0; vt < Vt; vt++) {
                    noc.async_read(
                        a_acc,
                        activated,
                        activated.get_entry_size(),
                        {.page_id = base + vt},
                        {.offset_bytes = ((j - raw) * Vt + vt) * activated.get_entry_size()});
                }
            }
            noc.async_read_barrier();
            activated.push_back((items - raw) * Vt);
        }
#endif
    }
}
