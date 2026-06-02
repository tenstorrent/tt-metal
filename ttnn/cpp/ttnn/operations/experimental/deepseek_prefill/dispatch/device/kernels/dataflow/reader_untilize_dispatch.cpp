// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//
// Untilize core RISCV_1 — input prefetch + per-token routing.
//
// Each sender owns N untilize cores. They share the same expert subset (filtered via
// dispatch_core_idx & core_mask) and split work by batch, round-robin: batch i is handled
// by core i % total_workers.
//
// offsets[] is a SINGLE shared counter array living on the owner (core_id==0) of the
// sender group. All cores grow page_idx left→right from the same offsets[e]. Concurrent
// access is serialized by a baton that circulates in global batch order: a core waits on
// its own turn semaphore, assigns page_idx for its batch (pulling/pushing the owner's
// offsets[] for non-owners; owner edits in place), then signals the core that owns the
// next global batch. The baton is both the mutex and the ordering — no opposite-ends trick,
// no per-expert histogram needed.
//
// At startup: only the owner loads tt_expert_offsets[] into c_3; every core loads the
// expert dispatch_table[] into c_9 (read-only). The owner seeds its turn semaphore to 1
// so batch 0 (always owned by core 0) starts without waiting.
//
// Per batch:
//   1. Signal compute to untilize this batch.
//   2. Stream tiled input from DRAM → c_0.
//   3. Read this batch's indices and weights pages from DRAM → c_1, c_2.
//   4. Walk each token's top-k:
//        * filter out experts not handled by this sender (core_mask),
//        * filter out experts with dispatch_table == -1,
//        * compute page_idx from the local offsets[expert] counter,
//        * classify local vs cross-device, record route/distance for cross-device.
//      Build the per-batch route plan into c_14.
//   5. cb_push_back(c_14) — writer RISC (this same core, RISCV_0) drains the plan
//      and executes the data movement (NOC → DRAM for local, per-entry push into
//      this core's owning sender CB set — c_4/c_5/c_6 for u1, c_16/c_17/c_18 for
//      u2, writer_cb_size slots deep — for cross-device).
//
// After the last batch: push a single sentinel entry on c_14 so the writer
// knows to forward the final ROUTE_INFO_SENTINEL to the sender.
//

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/dprint.h"
#include "ttnn/operations/ccl/common/kernels/moe_utils.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"

#define ENABLE_DISPATCH_DEBUG 0
#if ENABLE_DISPATCH_DEBUG
#define DPRINT_DISPATCH(...) DPRINT(__VA_ARGS__)
#else
#define DPRINT_DISPATCH(...)
#endif

constexpr uint32_t ROUTE_INFO_SENTINEL = 0xFFFFFFFF;
constexpr uint32_t PLAN_FLAG_LOCAL = 0x1u;
constexpr uint32_t PLAN_FLAG_END = 0x80000000u;
constexpr uint32_t PLAN_ENTRY_U32S = 8;

// Plan entry layout (8 u32 = 32 bytes):
//   [0] flags (bit 0: is_local, bit 31: end-of-plan sentinel)
//   [1] token_t           (offset in c_11 batch — 0..read_batch_size-1)
//   [2] routed_expert
//   [3] page_idx          (DRAM page for local + remote)
//   [4] token_idx         (global token index, for metadata)
//   [5] (k << 16) | (weight & 0xFFFF)
//   [6] route             (cross-device only)
//   [7] distance          (cross-device only)
//
// Page layout: [entry_count u32][padding][entries...]

void kernel_main() {
    using namespace ttnn::operations::ccl::common;

    // ===== Compile-time args =====
    constexpr uint32_t cb_input_id = get_compile_time_arg_val(0);
    constexpr uint32_t cb_signal_id = get_compile_time_arg_val(1);
    constexpr uint32_t hidden_size = get_compile_time_arg_val(2);
    constexpr uint32_t aligned_input_page_size = get_compile_time_arg_val(3);
    constexpr uint32_t total_batches = get_compile_time_arg_val(4);
    constexpr uint32_t core_id = get_compile_time_arg_val(5);
    constexpr uint32_t total_workers = get_compile_time_arg_val(6);

    constexpr uint32_t cb_indices_id = get_compile_time_arg_val(7);
    constexpr uint32_t cb_weights_id = get_compile_time_arg_val(8);
    constexpr uint32_t cb_offsets_id = get_compile_time_arg_val(9);
    constexpr uint32_t cb_dispatch_table_id = get_compile_time_arg_val(10);
    constexpr uint32_t cb_plan_id = get_compile_time_arg_val(11);

    constexpr uint32_t read_batch_size = get_compile_time_arg_val(12);
    constexpr uint32_t aligned_indices_page_size = get_compile_time_arg_val(13);
    constexpr uint32_t aligned_weights_page_size = get_compile_time_arg_val(14);
    constexpr uint32_t aligned_offsets_page_size = get_compile_time_arg_val(15);
    constexpr uint32_t aligned_dispatch_table_page_size = get_compile_time_arg_val(16);

    constexpr uint32_t offsets_pages = get_compile_time_arg_val(17);
    constexpr uint32_t dispatch_table_pages = get_compile_time_arg_val(18);

    constexpr uint32_t num_experts_per_tok = get_compile_time_arg_val(19);
    constexpr uint32_t n_routed_experts = get_compile_time_arg_val(20);
    constexpr uint32_t max_dispatch_buffer_token_size = get_compile_time_arg_val(21);
    constexpr uint32_t dispatch_core_idx = get_compile_time_arg_val(22);
    constexpr uint32_t num_dispatch_cores = get_compile_time_arg_val(23);
    constexpr uint32_t core_mask = num_dispatch_cores - 1;
    // Batches are assigned round-robin (batch i -> core i % total_workers); all cores
    // grow offsets[] left-to-right from the single shared owner copy under the baton.

    constexpr uint32_t num_devices = get_compile_time_arg_val(24);
    constexpr uint32_t mesh_rows = get_compile_time_arg_val(25);
    constexpr uint32_t mesh_cols = get_compile_time_arg_val(26);
    constexpr uint32_t linearized_mesh_coord = get_compile_time_arg_val(27);
    constexpr tt::tt_fabric::Topology topology = (tt::tt_fabric::Topology)get_compile_time_arg_val(28);

    constexpr auto input_args = TensorAccessorArgs<29>();
    constexpr auto indices_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
    constexpr auto weights_args = TensorAccessorArgs<indices_args.next_compile_time_args_offset()>();
    constexpr auto offsets_args = TensorAccessorArgs<weights_args.next_compile_time_args_offset()>();
    constexpr auto dispatch_table_args = TensorAccessorArgs<offsets_args.next_compile_time_args_offset()>();

    constexpr uint32_t tiles_per_row = hidden_size / 32;
    constexpr uint32_t block_ct_dim = 8;
    constexpr uint32_t num_tile_blocks = tiles_per_row / block_ct_dim;

#ifdef AXIS
    constexpr ReplicateGroup axis = ReplicateGroup(AXIS);
    constexpr uint32_t row = linearized_mesh_coord / mesh_cols;
    constexpr uint32_t col = linearized_mesh_coord % mesh_cols;
    constexpr uint32_t device_begin_idx = axis == ReplicateGroup::COLS ? col : row * mesh_cols;
    constexpr uint32_t device_stride = axis == ReplicateGroup::COLS ? mesh_cols : 1;
#else
    constexpr ReplicateGroup axis = ReplicateGroup::NONE;
    constexpr uint32_t device_begin_idx = 0;
    constexpr uint32_t device_stride = 1;
#endif

    // ===== Runtime args =====
    uint32_t rt_idx = 0;
    uint32_t input_tensor_address = get_arg_val<uint32_t>(rt_idx++);
    uint32_t indices_tensor_address = get_arg_val<uint32_t>(rt_idx++);
    uint32_t weights_tensor_address = get_arg_val<uint32_t>(rt_idx++);
    uint32_t offsets_tensor_address = get_arg_val<uint32_t>(rt_idx++);
    uint32_t dispatch_table_tensor_address = get_arg_val<uint32_t>(rt_idx++);
    uint32_t token_start_idx = get_arg_val<uint32_t>(rt_idx++);
    uint32_t token_end_idx = get_arg_val<uint32_t>(rt_idx++);
    // Baton-ring offset sync: offsets[] live only on the owner (core_id==0) of this
    // sender group. Every core pulls/pushes the shared offsets[] under a baton that
    // circulates in global batch order (core (b+1)%W signaled after batch b).
    uint32_t owner_noc_x = get_arg_val<uint32_t>(rt_idx++);
    uint32_t owner_noc_y = get_arg_val<uint32_t>(rt_idx++);
    uint32_t next_noc_x = get_arg_val<uint32_t>(rt_idx++);
    uint32_t next_noc_y = get_arg_val<uint32_t>(rt_idx++);
    uint32_t turn_semaphore_id = get_arg_val<uint32_t>(rt_idx++);

    const auto input_addr_gen = TensorAccessor(input_args, input_tensor_address, aligned_input_page_size);
    const auto indices_addr_gen = TensorAccessor(indices_args, indices_tensor_address, aligned_indices_page_size);
    const auto weights_addr_gen = TensorAccessor(weights_args, weights_tensor_address, aligned_weights_page_size);
    const auto offsets_addr_gen = TensorAccessor(offsets_args, offsets_tensor_address);
    const auto dispatch_table_addr_gen = TensorAccessor(dispatch_table_args, dispatch_table_tensor_address);

    // ===== Startup: load offsets[] (owner only) and dispatch_table[] into local L1 =====
    // offsets[] is a single shared counter array living on the owner (core_id==0) of this
    // sender group. The owner loads tt_expert_offsets[] from DRAM; non-owners leave their
    // local copy uninitialized and pull the owner's copy under the baton (per-batch loop).
    // dispatch_table[] is read-only -> every core keeps its own copy.
    constexpr bool IS_OWNER = (core_id == 0);
    cb_reserve_back(cb_offsets_id, offsets_pages);
    uint32_t offsets_base_addr = get_write_ptr(cb_offsets_id);
    if constexpr (IS_OWNER) {
        for (uint32_t i = 0; i < offsets_pages; i++) {
            noc_async_read_page(i, offsets_addr_gen, offsets_base_addr + i * aligned_offsets_page_size);
        }
    }
    cb_reserve_back(cb_dispatch_table_id, dispatch_table_pages);
    uint32_t dispatch_table_base_addr = get_write_ptr(cb_dispatch_table_id);
    for (uint32_t i = 0; i < dispatch_table_pages; i++) {
        noc_async_read_page(
            i, dispatch_table_addr_gen, dispatch_table_base_addr + i * aligned_dispatch_table_page_size);
    }
    noc_async_read_barrier();
    tt_l1_ptr uint32_t* offsets = reinterpret_cast<tt_l1_ptr uint32_t*>(offsets_base_addr);
    tt_l1_ptr int32_t* expert_dispatch_table = reinterpret_cast<tt_l1_ptr int32_t*>(dispatch_table_base_addr);

    // ===== Baton-ring setup =====
    // Each core waits on its own turn semaphore (local poll) and signals the next core's
    // semaphore after finishing its batch. The owner seeds its own semaphore to 1 so the
    // very first batch (batch 0, always owned by core 0) does not block.
    const uint32_t offsets_bytes = offsets_pages * aligned_offsets_page_size;
    volatile tt_l1_ptr uint32_t* turn_sem_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(turn_semaphore_id));
    uint64_t owner_offsets_noc_addr = get_noc_addr(owner_noc_x, owner_noc_y, offsets_base_addr);
    uint64_t next_turn_sem_noc_addr = get_noc_addr(next_noc_x, next_noc_y, get_semaphore(turn_semaphore_id));
    if constexpr (IS_OWNER) {
        noc_semaphore_set(turn_sem_ptr, 1);
    }
    uint32_t turn_expected = 1;  // per-core baton counter; +1 for each batch this core handles
    DPRINT_DISPATCH(
        "[R s={} c={}] startup done; owner={} total_batches={} total_workers={}\n",
        (uint32_t)dispatch_core_idx,
        (uint32_t)core_id,
        (uint32_t)IS_OWNER,
        (uint32_t)total_batches,
        (uint32_t)total_workers);

    // ===== Indices / weights scratch (overwritten per batch, single page slot used) =====
    cb_reserve_back(cb_indices_id, read_batch_size);
    uint32_t indices_base = get_write_ptr(cb_indices_id);
    cb_reserve_back(cb_weights_id, read_batch_size);
    uint32_t weights_base = get_write_ptr(cb_weights_id);

    // ===== Per-batch loop — this core handles batches core_id, core_id+total_workers, ... =====
    for (uint32_t batch_idx = core_id; batch_idx < total_batches; batch_idx += total_workers) {
        uint32_t tile_base_page = batch_idx * tiles_per_row;
        uint32_t batch_start = batch_idx * read_batch_size;
        uint32_t batch_end =
            (batch_start + read_batch_size < token_end_idx) ? batch_start + read_batch_size : token_end_idx;
        uint32_t batch_count = batch_end - batch_start;

        // 1. Signal compute to start untilizing this batch
        cb_reserve_back(cb_signal_id, 1);
        volatile tt_l1_ptr uint32_t* signal_ptr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_signal_id));
        signal_ptr[0] = 0x00000000;
        cb_push_back(cb_signal_id, 1);

        // 2. Stream tiled input stripe from DRAM in blocks of 8 tiles
        for (uint32_t blk = 0; blk < num_tile_blocks; blk++) {
            cb_reserve_back(cb_input_id, block_ct_dim);
            uint32_t blk_write_ptr = get_write_ptr(cb_input_id);
            uint32_t blk_start = tile_base_page + blk * block_ct_dim;
            for (uint32_t col = 0; col < block_ct_dim; col++) {
                noc_async_read_page(blk_start + col, input_addr_gen, blk_write_ptr + col * aligned_input_page_size);
            }
            noc_async_read_barrier();
            cb_push_back(cb_input_id, block_ct_dim);
        }

        // 3. Read this batch's indices and weights pages
        for (uint32_t t = 0; t < batch_count; t++) {
            noc_async_read_page(batch_start + t, indices_addr_gen, indices_base + t * aligned_indices_page_size);
            noc_async_read_page(batch_start + t, weights_addr_gen, weights_base + t * aligned_weights_page_size);
        }
        noc_async_read_barrier();

        // 4. Build per-batch route plan into c_14.
        DPRINT_DISPATCH(
            "[R s={} c={}] b={} reserving plan slot (blocks on writer drain)\n",
            (uint32_t)dispatch_core_idx,
            (uint32_t)core_id,
            batch_idx);
        cb_reserve_back(cb_plan_id, 1);
        uint32_t plan_addr = get_write_ptr(cb_plan_id);
        volatile tt_l1_ptr uint32_t* plan = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(plan_addr);
        uint32_t entry_count = 0;
        uint32_t entry_off = 8;  // entries start at u32 offset 8 (32B header)

        // Baton acquire: wait our turn, then pull the shared offsets[] from the owner.
        // The offset assignment runs under the baton; the expensive DRAM reads above are
        // outside it. (Owner edits its copy in place — no pull needed.)
        DPRINT_DISPATCH(
            "[R s={} c={}] b={} WAIT baton (turn_sem>={}, have={})\n",
            (uint32_t)dispatch_core_idx,
            (uint32_t)core_id,
            batch_idx,
            turn_expected,
            (uint32_t)(*turn_sem_ptr));
        noc_semaphore_wait_min(turn_sem_ptr, turn_expected);
        DPRINT_DISPATCH("[R s={} c={}] b={} GOT baton\n", (uint32_t)dispatch_core_idx, (uint32_t)core_id, batch_idx);
        turn_expected++;
        if constexpr (!IS_OWNER) {
            noc_async_read(owner_offsets_noc_addr, offsets_base_addr, offsets_bytes);
            noc_async_read_barrier();
        }

        for (uint32_t t = 0; t < batch_count; t++) {
            tt_l1_ptr int32_t* indices_t =
                reinterpret_cast<tt_l1_ptr int32_t*>(indices_base + t * aligned_indices_page_size);
            tt_l1_ptr uint16_t* weights_t =
                reinterpret_cast<tt_l1_ptr uint16_t*>(weights_base + t * aligned_weights_page_size);
            uint32_t token_idx = batch_start + t;

            for (uint32_t k = 0; k < num_experts_per_tok; k++) {
                int32_t routed_expert = indices_t[k];
                if (((uint32_t)routed_expert & core_mask) != dispatch_core_idx) {
                    continue;
                }
                int32_t expert_chip_og = expert_dispatch_table[routed_expert];
                if (expert_chip_og == -1) {
                    continue;
                }

                // Single shared counter, all cores grow left-to-right from offsets[e].
                uint32_t& offset = offsets[routed_expert];
                if (offset >= max_dispatch_buffer_token_size) {
                    offset++;
                    continue;
                }
                uint32_t page_idx = offset++;

                uint32_t expert_chip = device_begin_idx + (uint32_t)expert_chip_og * device_stride;
                bool is_local = (expert_chip == linearized_mesh_coord);
                int16_t weight = (int16_t)weights_t[k];

                uint32_t base = entry_off;
                plan[base + 0] = is_local ? PLAN_FLAG_LOCAL : 0;
                plan[base + 1] = t;
                plan[base + 2] = (uint32_t)routed_expert;
                plan[base + 3] = page_idx;
                plan[base + 4] = token_idx;
                plan[base + 5] = (k << 16) | ((uint32_t)(uint16_t)weight);

                if (!is_local) {
                    if constexpr (is_1d_topology<topology>()) {
                        uint32_t route = get_route<topology, mesh_rows, mesh_cols>(linearized_mesh_coord, expert_chip);
                        uint32_t distance =
                            manhattan_distance<topology, mesh_rows, mesh_cols>(linearized_mesh_coord, expert_chip);
                        plan[base + 6] = route;
                        plan[base + 7] = distance;
                    } else {
                        plan[base + 6] = 0;
                        plan[base + 7] = 0;
                    }
                } else {
                    plan[base + 6] = 0;
                    plan[base + 7] = 0;
                }

                entry_count++;
                entry_off += PLAN_ENTRY_U32S;
            }
        }

        // Baton release: push updated offsets[] back to the owner (non-owner only; owner
        // edited in place), then hand the baton to the core owning the next global batch.
        if constexpr (!IS_OWNER) {
            noc_async_write(offsets_base_addr, owner_offsets_noc_addr, offsets_bytes);
            noc_async_write_barrier();
        }
        if (batch_idx + 1 < total_batches) {
            noc_semaphore_inc(next_turn_sem_noc_addr, 1);
            DPRINT_DISPATCH(
                "[R s={} c={}] b={} RELEASE baton -> signaled next (entries={})\n",
                (uint32_t)dispatch_core_idx,
                (uint32_t)core_id,
                batch_idx,
                entry_count);
        } else {
            DPRINT_DISPATCH(
                "[R s={} c={}] b={} RELEASE baton -> LAST batch, no signal (entries={})\n",
                (uint32_t)dispatch_core_idx,
                (uint32_t)core_id,
                batch_idx,
                entry_count);
        }

        plan[0] = entry_count;
        cb_push_back(cb_plan_id, 1);
    }

    DPRINT_DISPATCH("[R s={} c={}] loop DONE -> pushing sentinels\n", (uint32_t)dispatch_core_idx, (uint32_t)core_id);
    // Send sentinel to compute so it breaks out of its loop
    cb_reserve_back(cb_signal_id, 1);
    volatile tt_l1_ptr uint32_t* signal_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_signal_id));
    signal_ptr[0] = ROUTE_INFO_SENTINEL;
    cb_push_back(cb_signal_id, 1);

    // Push end-of-plan sentinel for the writer so it forwards ROUTE_INFO_SENTINEL to sender.
    cb_reserve_back(cb_plan_id, 1);
    uint32_t sentinel_addr = get_write_ptr(cb_plan_id);
    volatile tt_l1_ptr uint32_t* sentinel_plan = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sentinel_addr);
    sentinel_plan[0] = 0;  // entry_count
    sentinel_plan[8] = PLAN_FLAG_END;
    cb_push_back(cb_plan_id, 1);
}
