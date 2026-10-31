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
// core, followed by an increment of the leader's count for that run. The leader parks each
// candidate's summed row as soon as it has it — and where the op gathers across ranks,
// tells the gather core, which sends a candidate's rows to the peers once all are parked.
void kernel_main() {
    constexpr uint32_t Ht = get_compile_time_arg_val(0);
    constexpr uint32_t num_candidates = get_compile_time_arg_val(1);
    constexpr uint32_t chunk_sem_id = get_compile_time_arg_val(2);
    // Where the op gathers: the first of the gather core's per-candidate count semaphores,
    // the ranks side by side in the output, and the counts' field width. Otherwise 0, 1, 0.
    constexpr uint32_t gather_sem_id = get_compile_time_arg_val(3);
    constexpr uint32_t ring_size = get_compile_time_arg_val(4);
    constexpr AttnResProgressFields row_fields{get_compile_time_arg_val(5)};
    constexpr AttnResProgressFields chunk_fields{get_compile_time_arg_val(6)};
    // One per fabric link, each counting the rows it sends.
    constexpr uint32_t num_gather_cores = get_compile_time_arg_val(7);
    constexpr uint32_t kSplitArgs = 8;
    using Split = AttnResBlockStatsSplit<kSplitArgs>;
    constexpr uint32_t chunks_per_row = Split::chunks_per_row;
    constexpr bool gathers = ring_size > 1;
    constexpr auto out_args = TensorAccessorArgs<kSplitArgs + 5>();

    // Common args: the output, this chip's rank, the gather cores as (x << 16) | y, then
    // where rows are cut, each row's leader the same way.
    constexpr uint32_t kGatherArgs = 2;
    constexpr uint32_t kLeaderArgs = kGatherArgs + num_gather_cores;
    const uint32_t rank = get_common_arg_val<uint32_t>(1);

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
    auto* chunk_counts = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(chunk_sem_id));
    if constexpr (chunks_per_row > 1) {
        if (leads) {
            cb_received_obj.reserve_back((chunks_per_row - 1) * num_candidates);
        }
    }

    for (uint32_t u = first_unit; u < first_unit + num_units; ++u) {
        const auto unit = Split::unit(u);
        if (!leads) {
            // Each candidate's partial over this run, into the leader's slot for it.
            const uint32_t leader = get_common_arg_val<uint32_t>(kLeaderArgs + unit.row);
            const uint32_t leader_x = leader >> 16;
            const uint32_t leader_y = leader & 0xFFFF;
            const uint64_t leader_counts = get_noc_addr(leader_x, leader_y, get_semaphore(chunk_sem_id));
            for (uint32_t c = 0; c < num_candidates; ++c) {
                const uint32_t slot = received_base + (c * (chunks_per_row - 1) + unit.chunk - 1) * tile_bytes;
                cb_out_obj.wait_front(1);
                noc_async_write(cb_out_obj.get_read_ptr(), get_noc_addr(leader_x, leader_y, slot), tile_bytes);
                noc_async_write_barrier();
                noc_semaphore_inc(leader_counts, chunk_fields.increment(unit.chunk - 1));
                cb_out_obj.pop_front(1);
            }
            continue;
        }
        for (uint32_t c = 0; c < num_candidates; ++c) {
            if constexpr (chunks_per_row > 1) {
                // Hand compute the row's other runs' partials for this candidate once all are in.
                for (uint32_t r = 0; r + 1 < chunks_per_row; ++r) {
                    while (chunk_fields.count(*chunk_counts, r) <= c) {
                        invalidate_l1_cache();
                    }
                }
                cb_received_obj.push_back(chunks_per_row - 1);
            }
            // The output is [1, C, N, 32 * P], one tile per (candidate, row, rank).
            cb_out_obj.wait_front(1);
            noc.async_write(
                cb_out_obj,
                out_accessor,
                tile_bytes,
                {.offset_bytes = 0},
                {.page_id = (c * Ht + unit.row) * ring_size + rank});
            if constexpr (gathers) {
                // Parked before it is announced: the gather cores read it back to send it.
                noc.async_write_barrier();
                for (uint32_t l = 0; l < num_gather_cores; ++l) {
                    const uint32_t gather_core = get_common_arg_val<uint32_t>(kGatherArgs + l);
                    noc_semaphore_inc(
                        get_noc_addr(
                            gather_core >> 16, gather_core & 0xFFFF, get_semaphore(gather_sem_id + row_fields.word(c))),
                        row_fields.increment(c));
                }
            } else {
                noc.async_writes_flushed();
            }
            cb_out_obj.pop_front(1);
        }
    }
    noc.async_write_barrier();
}
