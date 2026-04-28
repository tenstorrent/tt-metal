// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Writer for TurboQuant SDPA decode.
// Generates identity/scale tiles, writes output to DRAM, and (Tier 2A) handles
// the cross-core partial-state transfer between workers and the reducer:
//   - Worker (idx > 0): waits for compute to fill cb_partial_max/sum/out, NoC-
//     writes them to the reducer's matching cb_remote_* L1 slots, increments
//     the reducer's per-program semaphore, then exits without writing output.
//   - Reducer (idx == 0): waits for K-1 semaphore increments (one per worker),
//     then cb_push_back's its cb_remote_* CBs so the compute kernel's merge
//     loop unblocks. Output write at the end is unchanged.
//
// At K = 1 (current default) the new branches are skipped and behaviour is
// identical to the legacy writer.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel/dataflow/generate_reduce_scaler.hpp"
#include "ttnn/cpp/ttnn/kernel/dataflow/generate_bcast_scalar.hpp"
#include "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/dataflow_common.hpp"

void kernel_main() {
    constexpr uint32_t B = get_compile_time_arg_val(0);
    constexpr uint32_t NQH = get_compile_time_arg_val(1);
    constexpr uint32_t Sq_chunk_t = get_compile_time_arg_val(2);
    constexpr uint32_t vDHt = get_compile_time_arg_val(3);
    constexpr uint32_t num_cores = get_compile_time_arg_val(4);
    constexpr uint32_t identity_scalar_packed = get_compile_time_arg_val(5);
    constexpr auto out_args = TensorAccessorArgs<6>();

    uint32_t argidx = 0;
    const uint32_t out_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t core_id = get_arg_val<uint32_t>(argidx++);
    const uint32_t local_batch_start = get_arg_val<uint32_t>(argidx++);
    const uint32_t local_batch_end = get_arg_val<uint32_t>(argidx++);
    const uint32_t local_nh_start = get_arg_val<uint32_t>(argidx++);
    const uint32_t local_nh_end = get_arg_val<uint32_t>(argidx++);
    // Tier 2A Phase 2.3 — cross-core reduce args. Currently the program factory
    // sends (core_idx_in_group=0, cores_per_head=1, ...) so the worker / reducer
    // branches below are dead and behaviour is identical to before.
    const uint32_t core_idx_in_group_arg = get_arg_val<uint32_t>(argidx++);
    const uint32_t cores_per_head_arg = get_arg_val<uint32_t>(argidx++);
    const uint32_t reducer_noc_x_arg = get_arg_val<uint32_t>(argidx++);
    const uint32_t reducer_noc_y_arg = get_arg_val<uint32_t>(argidx++);
    const uint32_t reducer_semaphore_id_arg = get_arg_val<uint32_t>(argidx++);

    constexpr uint32_t out_chunk_tiles = Sq_chunk_t * vDHt;

    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    constexpr uint32_t cb_identity_scale = tt::CBIndex::c_5;
    constexpr uint32_t cb_col_identity = tt::CBIndex::c_7;
    // Tier 2A partial-state CBs.
    constexpr uint32_t cb_partial_max = tt::CBIndex::c_18;
    constexpr uint32_t cb_partial_sum = tt::CBIndex::c_19;
    constexpr uint32_t cb_partial_out = tt::CBIndex::c_20;
    constexpr uint32_t cb_remote_max = tt::CBIndex::c_21;
    constexpr uint32_t cb_remote_sum = tt::CBIndex::c_22;
    constexpr uint32_t cb_remote_out = tt::CBIndex::c_23;

    const uint32_t out_tile_bytes = get_tile_size(cb_out);
    const auto out_writer = TensorAccessor(out_args, out_addr, out_tile_bytes);

    // ── Generate identity/scale tile using proper helper ──
    generate_reduce_scaler(cb_identity_scale, identity_scalar_packed);

    // ── Generate column identity (same format as standard SDPA writer) ──
    generate_bcast_col_scalar(cb_col_identity, identity_scalar_packed);

    const bool is_reducer = (core_idx_in_group_arg == 0);
    const bool is_worker = (cores_per_head_arg > 1) && !is_reducer;

    const uint32_t partial_max_tile_bytes = get_tile_size(cb_partial_max);
    const uint32_t partial_sum_tile_bytes = get_tile_size(cb_partial_sum);
    const uint32_t partial_out_tile_bytes = get_tile_size(cb_partial_out);

    // Reducer-side cb_remote_* base L1 addresses — same on every core because
    // CB allocation is deterministic. cb_remote_* holds K slots (one per worker
    // including reducer slot 0 which is unused). Worker idx writes to slot idx,
    // offset by idx * (count * tile_bytes). Reducer compute reads slots 1..K-1.
    const uint32_t remote_max_base_addr = get_write_ptr(cb_remote_max);
    const uint32_t remote_sum_base_addr = get_write_ptr(cb_remote_sum);
    const uint32_t remote_out_base_addr = get_write_ptr(cb_remote_out);
    const uint32_t partial_max_chunk_bytes = partial_max_tile_bytes * Sq_chunk_t;
    const uint32_t partial_sum_chunk_bytes = partial_sum_tile_bytes * Sq_chunk_t;
    const uint32_t partial_out_chunk_bytes = partial_out_tile_bytes * out_chunk_tiles;
    const uint32_t worker_slot_max_offset = core_idx_in_group_arg * partial_max_chunk_bytes;
    const uint32_t worker_slot_sum_offset = core_idx_in_group_arg * partial_sum_chunk_bytes;
    const uint32_t worker_slot_out_offset = core_idx_in_group_arg * partial_out_chunk_bytes;

    const uint32_t reducer_sem_l1_addr = get_semaphore(reducer_semaphore_id_arg);
    volatile tt_l1_ptr uint32_t* local_sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(reducer_sem_l1_addr);

    // ── Per-(batch, head) iteration ──
    for (uint32_t nb = local_batch_start; nb < local_batch_end; ++nb) {
        for (uint32_t nq = local_nh_start; nq < local_nh_end; ++nq) {
            if (is_worker) {
                // Worker: NoC-write the three partial CBs to the reducer's
                // cb_remote_* slots (both cores have the same L1 layout, so
                // remote_*_local_addr is the same address on the reducer too),
                // then increment the reducer's semaphore. No output write.
                const uint64_t reducer_remote_max_noc =
                    get_noc_addr(reducer_noc_x_arg, reducer_noc_y_arg, remote_max_base_addr + worker_slot_max_offset);
                const uint64_t reducer_remote_sum_noc =
                    get_noc_addr(reducer_noc_x_arg, reducer_noc_y_arg, remote_sum_base_addr + worker_slot_sum_offset);
                const uint64_t reducer_remote_out_noc =
                    get_noc_addr(reducer_noc_x_arg, reducer_noc_y_arg, remote_out_base_addr + worker_slot_out_offset);
                const uint64_t reducer_sem_noc =
                    get_noc_addr(reducer_noc_x_arg, reducer_noc_y_arg, reducer_sem_l1_addr);

                cb_wait_front(cb_partial_max, Sq_chunk_t);
                noc_async_write(get_read_ptr(cb_partial_max), reducer_remote_max_noc, partial_max_chunk_bytes);

                cb_wait_front(cb_partial_sum, Sq_chunk_t);
                noc_async_write(get_read_ptr(cb_partial_sum), reducer_remote_sum_noc, partial_sum_chunk_bytes);

                cb_wait_front(cb_partial_out, out_chunk_tiles);
                noc_async_write(get_read_ptr(cb_partial_out), reducer_remote_out_noc, partial_out_chunk_bytes);

                noc_async_write_barrier();

                cb_pop_front(cb_partial_max, Sq_chunk_t);
                cb_pop_front(cb_partial_sum, Sq_chunk_t);
                cb_pop_front(cb_partial_out, out_chunk_tiles);

                // Signal reducer that this worker's partial state has landed.
                noc_semaphore_inc(reducer_sem_noc, 1);
                continue;  // worker writes no output
            }

            if (is_reducer && cores_per_head_arg > 1) {
                // Reducer: wait for K-1 worker increments, then push K worth of
                // tiles so the compute kernel's merge loop can cb_wait_front on
                // all peer slots in one go. (Slot 0 is the reducer's own and is
                // unused by the merge — but we still push it so the indexing
                // matches core_idx_in_group.) NoC writes have landed by the
                // time the semaphore hits K-1 (workers barrier before sema_inc).
                const uint32_t expected_workers = cores_per_head_arg - 1;
                noc_semaphore_wait(local_sem_ptr, expected_workers);
                noc_semaphore_set(local_sem_ptr, 0);  // reset for trace replay

                cb_reserve_back(cb_remote_max, cores_per_head_arg * Sq_chunk_t);
                cb_reserve_back(cb_remote_sum, cores_per_head_arg * Sq_chunk_t);
                cb_reserve_back(cb_remote_out, cores_per_head_arg * out_chunk_tiles);
                cb_push_back(cb_remote_max, cores_per_head_arg * Sq_chunk_t);
                cb_push_back(cb_remote_sum, cores_per_head_arg * Sq_chunk_t);
                cb_push_back(cb_remote_out, cores_per_head_arg * out_chunk_tiles);
            }

            // ── Reducer / single-core path: write output ──
            uint32_t out_tile_id = nb * NQH * Sq_chunk_t * vDHt + nq * Sq_chunk_t * vDHt;

            cb_wait_front(cb_out, out_chunk_tiles);
            uint32_t l1_read_addr = get_read_ptr(cb_out);
            for (uint32_t t = 0; t < out_chunk_tiles; t++) {
                noc_async_write_tile(out_tile_id + t, out_writer, l1_read_addr);
                l1_read_addr += out_tile_bytes;
            }
            noc_async_write_barrier();
            cb_pop_front(cb_out, out_chunk_tiles);
        }
    }
}
