// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// reduce_scatter_average — reduce reader (NCRISC), core (0,2).
//
// Feeds the compute kernel N contributions of S slice tiles each, in ARRIVAL order,
// as g-tile granules through cb_contributions:
//   * contribution 0: this device's OWN slice, read directly from the input tensor
//     (base my_chip_id * slice_Wt) — no dependence on anything remote, so compute
//     pass 0 starts immediately (T7).
//   * contributions 1..N-1: the slice of each remote shard the moment its
//     double-inc lands in this core's sem_fwd / sem_bwd counter — a TWO-WAY POLL
//     (volatile reads + invalidate_l1_cache, the same spin noc_semaphore_wait_min
//     uses) so whichever direction lands first is consumed first, preserving the
//     accumulate-overlaps-flight property. A single-counter noc_semaphore_wait_min
//     would serialize the directions (op_design.md "Arrival poll").
//   Fwd arrival a carries the shard of chip (i - (1+a)) mod N; bwd arrival b that of
//   chip (i + (1+b)) mod N (T1/T2, nearest-first). Every contribution is walked in
//   the IDENTICAL SliceRowWalker order, which is what keeps add_tiles positionally
//   aligned across passes (R11).
//
// Also fills cb_scaler once with the 1/N broadcast scalar: bf16 via
// generate_bcast_unary_scalar (owns reserve/push); fp32 via the mirrored raw
// one-word store — the helper assumes 16-bit tile elements (op_design.md raw-API
// justification). Only element (0,0) is ever read by the SCALAR bcast.
//
// Cache-reuse re-arm (R1): reset BOTH sem counters after all arrivals are consumed.
//
// CT args: [cb_contributions, cb_scaler, my_chip_id, ring_size, fwd_arrivals,
//           bwd_arrivals, S, g, Wt, slice_Wt, P, dim, scaler_bits, scaler_is_fp32]
//          + input TensorAccessorArgs + gather TensorAccessorArgs
// RT args: [input_addr, gather_buffer_addr, page_size, sem_fwd_addr, sem_bwd_addr]

#include "api/dataflow/dataflow_api.h"
#include "ttnn/kernel/dataflow/generate_bcast_scalar.hpp"
#include "ttnn/operations/ccl/shared_with_host/ccl_helpers_schedule.hpp"

namespace sched = ttnn::ccl::schedule;

void kernel_main() {
    constexpr uint32_t cb_contributions = get_compile_time_arg_val(0);
    constexpr uint32_t cb_scaler = get_compile_time_arg_val(1);
    constexpr uint32_t my_chip_id = get_compile_time_arg_val(2);
    constexpr uint32_t ring_size = get_compile_time_arg_val(3);
    constexpr uint32_t fwd_arrivals = get_compile_time_arg_val(4);
    constexpr uint32_t bwd_arrivals = get_compile_time_arg_val(5);
    constexpr uint32_t S = get_compile_time_arg_val(6);         // output tiles per device
    constexpr uint32_t g = get_compile_time_arg_val(7);         // CB granule (divides S)
    constexpr uint32_t Wt = get_compile_time_arg_val(8);        // shard tile-columns
    constexpr uint32_t slice_Wt = get_compile_time_arg_val(9);  // output tile-columns
    constexpr uint32_t P = get_compile_time_arg_val(10);        // tiles per shard
    constexpr uint32_t dim = get_compile_time_arg_val(11);      // scatter dim
    constexpr uint32_t scaler_bits = get_compile_time_arg_val(12);
    constexpr uint32_t scaler_is_fp32 = get_compile_time_arg_val(13);
    constexpr auto input_args = TensorAccessorArgs<14>();
    constexpr auto gather_buffer_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();

    // R9: gate the scatter dim at compile time. The walk below (SliceRowWalker over an
    // Rt x Wt grid, base = my_chip_id * slice_Wt) is the dim=3 shape; dim=2 is
    // Refinement 2 and never reaches the kernel (host SUPPORTED gate).
    static_assert(sched::is_supported_scatter_dim(dim), "reduce_scatter_average: unsupported scatter dim");
    static_assert(dim == 3, "reduce_scatter_average: Phase-0 implements dim=3 only (Refinement 2 adds dim=2)");
    static_assert(g > 0 && S % g == 0, "reduce_scatter_average: granule must divide the slice tile count");
    static_assert(fwd_arrivals + bwd_arrivals + 1 == ring_size, "arrivals + own contribution must equal N");

    uint32_t ai = 0;
    const uint32_t input_addr = get_arg_val<uint32_t>(ai++);
    const uint32_t gather_buffer_addr = get_arg_val<uint32_t>(ai++);
    const uint32_t page_size = get_arg_val<uint32_t>(ai++);
    const uint32_t sem_fwd_addr = get_arg_val<uint32_t>(ai++);
    const uint32_t sem_bwd_addr = get_arg_val<uint32_t>(ai++);

    const auto input = TensorAccessor(input_args, input_addr, page_size);
    const auto gather_buffer = TensorAccessor(gather_buffer_args, gather_buffer_addr, page_size);
    auto sem_fwd_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_fwd_addr);
    auto sem_bwd_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_bwd_addr);

    // 1. Scaler tile: pushed once, never popped (persistent — compute waits count 1).
    if constexpr (scaler_is_fp32 != 0) {
        // Raw fill: full-width IEEE-754 store to element (0,0) of the Float32 page.
        // generate_bcast_unary_scalar assumes 16-bit elements, so it cannot be used.
        cb_reserve_back(cb_scaler, 1);
        auto scaler_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_scaler));
        scaler_ptr[0] = scaler_bits;
        cb_push_back(cb_scaler, 1);
    } else {
        generate_bcast_unary_scalar(cb_scaler, scaler_bits);  // owns reserve/push of 1 page
    }

    // First tile of this device's slice within any shard's Rt x Wt tile grid.
    constexpr uint32_t slice_base = sched::slice_tile_offset(dim, my_chip_id, 0, 0, slice_Wt);

    // One contribution = S tiles walked slice-row-major, streamed as S/g granules.
    // The walk order is identical for every contribution (R11).
    sched::SliceRowWalker walker(slice_Wt, Wt);
    auto push_contribution = [&](const auto& src, uint32_t base) {
        walker.set_base(base);
        walker.reset_offsets(0, 0);
        for (uint32_t chunk = 0; chunk < S / g; ++chunk) {
            cb_reserve_back(cb_contributions, g);
            uint32_t l1 = get_write_ptr(cb_contributions);
            for (uint32_t t = 0; t < g; ++t) {
                const uint32_t id = walker.next();  // returns AND advances — once per tile
                noc_async_read(src.get_noc_addr(id), l1, page_size);
                l1 += page_size;
            }
            noc_async_read_barrier();
            cb_push_back(cb_contributions, g);
        }
    };

    // 2. Own contribution first — straight from the input tensor (gather_buffer block
    //    my_chip_id is never written; the reference's serialized self-copy is deleted).
    push_contribution(input, slice_base);

    // 3. Remote contributions in ARRIVAL order: two-way monotonic-counter poll.
    //    sem_dir > consumed  <=>  at least one unconsumed arrival in that direction,
    //    and its data is fully landed (the inc is in-order behind the pages, R8).
    uint32_t fwd_consumed = 0;
    uint32_t bwd_consumed = 0;
    constexpr uint32_t total_arrivals = fwd_arrivals + bwd_arrivals;
    for (uint32_t done = 0; done < total_arrivals;) {
        invalidate_l1_cache();
        if (fwd_consumed < fwd_arrivals && *sem_fwd_ptr > fwd_consumed) {
            const uint32_t src_chip = (my_chip_id + ring_size - 1 - fwd_consumed) % ring_size;
            push_contribution(gather_buffer, src_chip * P + slice_base);
            ++fwd_consumed;
            ++done;
        } else if (bwd_consumed < bwd_arrivals && *sem_bwd_ptr > bwd_consumed) {
            const uint32_t src_chip = (my_chip_id + 1 + bwd_consumed) % ring_size;
            push_contribution(gather_buffer, src_chip * P + slice_base);
            ++bwd_consumed;
            ++done;
        }
    }

    // 4. Cache-reuse re-arm (R1): all fwd_arrivals + bwd_arrivals incs have been
    //    OBSERVED, so none can still be in flight — the reset cannot race a sender.
    noc_semaphore_set(sem_fwd_ptr, 0);
    noc_semaphore_set(sem_bwd_ptr, 0);
}
