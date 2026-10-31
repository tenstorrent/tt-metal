// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/operations/experimental/deepseek_prefill/attn_res_block_stats/device/kernels/attn_res_block_stats_items.hpp"

// Where a row is cut into runs, a run's partials go to the row's leader rather than to
// DRAM: into the leader's copy of `cb_received`, which sits at the same address on every
// core, followed by an increment of the count the leader waits on.
void kernel_main() {
    constexpr uint32_t Ht = get_compile_time_arg_val(0);
    constexpr uint32_t num_candidates = get_compile_time_arg_val(1);
    constexpr uint32_t received_sem_id = get_compile_time_arg_val(2);
    constexpr uint32_t kSplitArgs = 3;
    using Split = AttnResBlockStatsSplit<kSplitArgs>;
    constexpr uint32_t chunks_per_row = Split::chunks_per_row;
    constexpr auto out_args = TensorAccessorArgs<kSplitArgs + 5>();

    // Common args: the output, then where rows are cut, each row's leader as (x << 16) | y.
    constexpr uint32_t kLeaderArgs = 1;

    constexpr uint32_t cb_received = 5;
    constexpr uint32_t cb_out = 16;
    constexpr uint32_t tile_bytes = get_tile_size(cb_out);

    Noc noc;
    CircularBuffer cb_received_obj(cb_received);
    CircularBuffer cb_out_obj(cb_out);
    auto out_accessor = TensorAccessor(out_args, get_common_arg_val<uint32_t>(0));

    uint32_t first_unit, num_units;
    Split::core_units(first_unit, num_units);
    if (num_units == 0) {
        return;
    }
    const bool leads = Split::leads(first_unit);
    const uint32_t received_base = cb_received_obj.get_write_ptr();

    for (uint32_t u = first_unit; u < first_unit + num_units; ++u) {
        const auto unit = Split::unit(u);
        if (!leads) {
            // This run's partial for every candidate, into the leader's slots for the run.
            const uint32_t leader = get_common_arg_val<uint32_t>(kLeaderArgs + unit.row);
            const uint32_t leader_x = leader >> 16;
            const uint32_t leader_y = leader & 0xFFFF;
            const uint32_t slots = received_base + (unit.chunk - 1) * num_candidates * tile_bytes;
            for (uint32_t c = 0; c < num_candidates; ++c) {
                cb_out_obj.wait_front(1);
                noc_async_write(
                    cb_out_obj.get_read_ptr(), get_noc_addr(leader_x, leader_y, slots + c * tile_bytes), tile_bytes);
                noc_async_write_barrier();
                cb_out_obj.pop_front(1);
            }
            noc_semaphore_inc(get_noc_addr(leader_x, leader_y, get_semaphore(received_sem_id)), 1);
            continue;
        }
        if constexpr (chunks_per_row > 1) {
            // Hand compute the row's other runs once they have all landed.
            cb_received_obj.reserve_back((chunks_per_row - 1) * num_candidates);
            Semaphore<> received_sem(received_sem_id);
            received_sem.wait(chunks_per_row - 1);
            received_sem.set(0);
            cb_received_obj.push_back((chunks_per_row - 1) * num_candidates);
        }
        // The output is [1, C, N, 32], one tile per (candidate, row).
        for (uint32_t c = 0; c < num_candidates; ++c) {
            cb_out_obj.wait_front(1);
            noc.async_write(cb_out_obj, out_accessor, tile_bytes, {.offset_bytes = 0}, {.page_id = c * Ht + unit.row});
            noc.async_writes_flushed();
            cb_out_obj.pop_front(1);
        }
    }
    noc.async_write_barrier();
}
