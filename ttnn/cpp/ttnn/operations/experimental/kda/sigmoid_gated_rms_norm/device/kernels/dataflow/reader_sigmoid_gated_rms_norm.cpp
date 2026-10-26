// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/cpp/ttnn/kernel/dataflow/generate_bcast_scalar_metal2.hpp"
#ifdef L1_PREFETCH
#include "ttnn/cpp/ttnn/kernel_lib/l1_prefetch_dataflow.hpp"
#endif

// Reads each group's inputs (the writer reads the gate on the other NoC), then the weight after the first group.
// With L1_PREFETCH a core whose prefetch_job is nonzero then runs job prefetch_job - 1 of a later op's L1 prefetch
// (l1_prefetch_dataflow.hpp): this RISC is idle for the rest of the norm, which the compute bounds.
template <uint32_t Vt, uint32_t epsilon_bits, uint32_t G>
TT_KERNEL void reader(uint32_t wi_start, uint32_t wi_count, uint32_t prefetch_job) {
    const auto x_acc = TensorAccessor(tensor::input);
    const auto w_acc = TensorAccessor(tensor::weight);
    DataflowBuffer x(dfb::x);
    DataflowBuffer weight(dfb::weight);
    DataflowBuffer epsilon(dfb::epsilon);
    Noc noc;

    for (uint32_t first = 0; first < wi_count; first += G) {
        const uint32_t items = wi_count - first < G ? wi_count - first : G;
        // The compute kernel's first stage is the first item's gate sigmoid, which these reads overlap.
        x.reserve_back(items * Vt);
        const uint32_t x_base = (wi_start + first) * Vt;
        for (uint32_t t = 0; t < items * Vt; t++) {
            noc.async_read(
                x_acc, x, x.get_entry_size(), {.page_id = x_base + t}, {.offset_bytes = t * x.get_entry_size()});
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
    }
#ifdef L1_PREFETCH
    if (prefetch_job != 0) {
        dataflow_kernel_lib::l1_prefetch::run_job(
            TensorAccessor(tensor::prefetch_jobs),
            TensorAccessor(tensor::prefetch_source),
            prefetch_job - 1,
            DataflowBuffer(dfb::job_row).get_write_ptr());
    }
#else
    (void)prefetch_job;
#endif
}
