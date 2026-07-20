// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//
// Reader kernel for the untilize cores in the tile-layout dispatch path.
// Runs on the reader RISC of each untilize core, paired with
// writer_untilize_dispatch.cpp on the other data-movement RISC: this kernel
// streams tiled input for compute to untilize and builds the per-batch route
// plan, while the writer drains the previous batch's plan and issues the NOC
// writes.
//
// Token batches are distributed round-robin across total_workers untilize cores:
// core i processes batches i, i+total_workers, …
//
// Shared state — offsets[] (per-expert DRAM page allocators) is a single counter
// array owned by core 0 of each sender group.  Cores take turns mutating it under
// a baton ring (turn semaphore) that circulates in global batch order: a core
// pulls the owner's offsets[], appends its batch's allocations, writes them back,
// then hands the baton to the next core.  dispatch_table[] is read-only, so every
// core keeps its own copy.
//
// For each assigned batch:
//   1. Signal compute to start untilizing this batch (cb_signal_id).
//   2. Stream the tiled input stripe from DRAM → cb_input_id, block_ct_dim tiles
//      at a time, for compute to untilize.
//   3. Read this batch's indices pages from DRAM.
//   4. Take the baton, then build the route plan into cb_plan_id: for each
//      (token, top-k) routed to this dispatch core, allocate a DRAM page from
//      offsets[expert] (drop if the dispatch buffer is full), classify it local
//      vs cross-device (computing route/distance for the latter), and append a
//      PlanEntry.  Write offsets[] back and release the baton.
//
// After the last batch: push an end-of-plan sentinel — ROUTE_INFO_SENTINEL to
// compute (cb_signal_id) and a zero-entry sentinel plan page to the writer.

#include "tools/profiler/kernel_profiler.hpp"
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/debug/dprint.h"
#include "ttnn/operations/ccl/common/kernels/moe_utils.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "ttnn/operations/experimental/deepseek_prefill/dispatch/device/kernels/dataflow/dispatch_plan.hpp"

#define ENABLE_DISPATCH_DEBUG 0
#if ENABLE_DISPATCH_DEBUG
#define DPRINT_DISPATCH(...) DPRINT(__VA_ARGS__)
#else
#define DPRINT_DISPATCH(...)
#endif

constexpr uint32_t ROUTE_INFO_SENTINEL = 0xFFFFFFFF;

// Plan page layout (PlanHeader + PlanEntry[]) is defined in dispatch_plan.hpp and shared
// with the writer kernel.

void kernel_main() {
    using namespace ttnn::operations::ccl::common;

    Noc noc;

    // ===== Compile-time args =====
    constexpr uint32_t cb_input_id = get_compile_time_arg_val(0);
    constexpr uint32_t cb_signal_id = get_compile_time_arg_val(1);
    constexpr uint32_t hidden_size = get_compile_time_arg_val(2);
    constexpr uint32_t aligned_input_page_size = get_compile_time_arg_val(3);
    constexpr uint32_t total_batches = get_compile_time_arg_val(4);
    constexpr uint32_t core_id = get_compile_time_arg_val(5);
    constexpr uint32_t total_workers = get_compile_time_arg_val(6);

    constexpr uint32_t cb_indices_id = get_compile_time_arg_val(7);
    constexpr uint32_t cb_offsets_id = get_compile_time_arg_val(8);
    constexpr uint32_t cb_dispatch_table_id = get_compile_time_arg_val(9);
    constexpr uint32_t cb_plan_id = get_compile_time_arg_val(10);

    constexpr uint32_t read_batch_size = get_compile_time_arg_val(11);
    constexpr uint32_t aligned_indices_page_size = get_compile_time_arg_val(12);
    constexpr uint32_t aligned_offsets_page_size = get_compile_time_arg_val(13);
    constexpr uint32_t aligned_dispatch_table_page_size = get_compile_time_arg_val(14);

    constexpr uint32_t offsets_pages = get_compile_time_arg_val(15);
    constexpr uint32_t dispatch_table_pages = get_compile_time_arg_val(16);

    constexpr uint32_t num_experts_per_tok = get_compile_time_arg_val(17);
    constexpr uint32_t n_routed_experts = get_compile_time_arg_val(18);
    constexpr uint32_t max_dispatch_buffer_token_size = get_compile_time_arg_val(19);
    constexpr uint32_t dispatch_core_idx = get_compile_time_arg_val(20);
    constexpr uint32_t num_dispatch_cores = get_compile_time_arg_val(21);
    constexpr uint32_t core_mask = num_dispatch_cores - 1;
    // Batches are assigned round-robin (batch i -> core i % total_workers); all cores
    // grow offsets[] left-to-right from the single shared owner copy under the baton.

    constexpr uint32_t num_devices = get_compile_time_arg_val(22);
    constexpr uint32_t mesh_rows = get_compile_time_arg_val(23);
    constexpr uint32_t mesh_cols = get_compile_time_arg_val(24);
    constexpr uint32_t linearized_mesh_coord = get_compile_time_arg_val(25);
    constexpr tt::tt_fabric::Topology topology = (tt::tt_fabric::Topology)get_compile_time_arg_val(26);

    constexpr uint32_t block_ct_dim = get_compile_time_arg_val(27);

    constexpr auto input_args = TensorAccessorArgs<28>();
    constexpr auto indices_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
    constexpr auto offsets_args = TensorAccessorArgs<indices_args.next_compile_time_args_offset()>();
    constexpr auto dispatch_table_args = TensorAccessorArgs<offsets_args.next_compile_time_args_offset()>();

#ifdef HAS_PADDING_CONFIG
    // padding_config accessor + scratch CB id are appended LAST so the existing index layout is unchanged.
    constexpr auto padding_cfg_args = TensorAccessorArgs<dispatch_table_args.next_compile_time_args_offset()>();
    constexpr uint32_t cb_padding_config_id =
        get_compile_time_arg_val(padding_cfg_args.next_compile_time_args_offset());
#endif

    constexpr uint32_t tiles_per_row = hidden_size / 32;
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
#ifdef HAS_PADDING_CONFIG
    // padding_config base address appended after the 12 base runtime args.
    uint32_t padding_config_address = get_arg_val<uint32_t>(rt_idx++);
#endif

    const auto input_addr_gen = TensorAccessor(input_args, input_tensor_address, aligned_input_page_size);
    const auto indices_addr_gen = TensorAccessor(indices_args, indices_tensor_address, aligned_indices_page_size);
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
    Semaphore<> next_turn_sem(turn_semaphore_id);
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

    // ===== Indices scratch (overwritten per batch, single page slot used) =====
    cb_reserve_back(cb_indices_id, read_batch_size);
    uint32_t indices_base = get_write_ptr(cb_indices_id);

    // Shrink the batch loop to this device's real (unpadded) tokens when right-padded (pad_side == 0).
    // The writer applies the same reduction so they agree on the end-of-plan handshake. Padded tokens
    // in the trailing batch keep their sentinel expert index, so expert_dispatch_table[sentinel] == -1
    // drops them — making a coarse ceil(real/32) batch bound safe.
    uint32_t effective_total_batches = total_batches;
#ifdef HAS_PADDING_CONFIG
    {
        const auto padding_cfg_gen = TensorAccessor(padding_cfg_args, padding_config_address);
        cb_reserve_back(cb_padding_config_id, 1);
        uint32_t pc_l1 = get_write_ptr(cb_padding_config_id);
        noc_async_read_page(0, padding_cfg_gen, pc_l1);
        noc_async_read_barrier();
        tt_l1_ptr uint32_t* pc = reinterpret_cast<tt_l1_ptr uint32_t*>(pc_l1);
        uint32_t real_count = pc[0];
        uint32_t pad_side = pc[1];
        if (pad_side == 0) {
            uint32_t real_batches = (real_count + read_batch_size - 1) / read_batch_size;
            if (real_batches < effective_total_batches) {
                effective_total_batches = real_batches;
            }
        }
    }
#endif

    // ===== Per-batch loop — this core handles batches core_id, core_id+total_workers, ... =====
    for (uint32_t batch_idx = core_id; batch_idx < effective_total_batches; batch_idx += total_workers) {
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

        // 2. Stream tiled input stripe from DRAM in blocks of block_ct_dim tiles
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

        // 3. Read this batch's indices pages
        {
            // DeviceZoneScopedN("batch-DRAM-read-indices-weights");
            for (uint32_t t = 0; t < batch_count; t++) {
                noc_async_read_page(batch_start + t, indices_addr_gen, indices_base + t * aligned_indices_page_size);
            }
            noc_async_read_barrier();
        }

        // 4. Build per-batch route plan into c_14.
        DPRINT_DISPATCH(
            "[R s={} c={}] b={} reserving plan slot (blocks on writer drain)\n",
            (uint32_t)dispatch_core_idx,
            (uint32_t)core_id,
            batch_idx);
        cb_reserve_back(cb_plan_id, 1);
        uint32_t plan_addr = get_write_ptr(cb_plan_id);
        volatile tt_l1_ptr PlanHeader* plan = reinterpret_cast<volatile tt_l1_ptr PlanHeader*>(plan_addr);
        volatile tt_l1_ptr PlanEntry* entries =
            reinterpret_cast<volatile tt_l1_ptr PlanEntry*>(plan_addr + sizeof(PlanHeader));
        uint32_t entry_count = 0;

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
            noc.async_read(
                UnicastEndpoint{},
                CoreLocalMem<uint32_t>(offsets_base_addr),
                offsets_bytes,
                {.noc_x = owner_noc_x, .noc_y = owner_noc_y, .addr = offsets_base_addr},
                {});
            noc_async_read_barrier();
        }

        for (uint32_t t = 0; t < batch_count; t++) {
            tt_l1_ptr uint16_t* indices_t =
                reinterpret_cast<tt_l1_ptr uint16_t*>(indices_base + t * aligned_indices_page_size);
            uint32_t token_idx = batch_start + t;

            // Walk this token's top-k experts; emit a plan entry for each one routed to this
            // dispatch core (after the ownership / mapping / capacity filters below).
            for (uint32_t k = 0; k < num_experts_per_tok; k++) {
                int32_t routed_expert = indices_t[k];
                // Skip experts not owned by this dispatch core (low bits of the expert id
                // select the dispatch core), and experts the table maps nowhere (-1).
                if (((uint32_t)routed_expert & core_mask) != dispatch_core_idx) {
                    continue;
                }
                int32_t expert_chip_og = expert_dispatch_table[routed_expert];
                if (expert_chip_og == -1) {
                    continue;
                }

                // Allocate this token's destination DRAM page from the expert's counter.
                // Single shared counter; all cores grow it left-to-right from offsets[e].
                // If the dispatch buffer for this expert is full, still bump the counter
                // (so capacity accounting stays consistent) but drop the token — no entry.
                uint32_t& offset = offsets[routed_expert];
                if (offset >= max_dispatch_buffer_token_size) {
                    offset++;
                    continue;
                }
                uint32_t page_idx = offset++;

                uint32_t expert_chip = device_begin_idx + (uint32_t)expert_chip_og * device_stride;
                bool is_local = (expert_chip == linearized_mesh_coord);

                volatile tt_l1_ptr PlanEntry* entry = &entries[entry_count];
                entry->flags = is_local ? PLAN_FLAG_LOCAL : 0;
                entry->token_t = t;
                entry->routed_expert = (uint32_t)routed_expert;
                entry->page_idx = page_idx;
                entry->token_idx = token_idx;
                // Single aligned 32-bit store: baby-RISC sub-word L1 stores are unreliable on BH.
                entry->weight_k = pack_weight_k(0, (uint16_t)k);
                // Linearized destination device index. Under 1D it is unused by the fabric writer
                // (route/distance drive the send); under 2D it is the only routing input — the
                // writer recomputes the EDM direction and (mesh,chip) header from it.
                entry->dst_chip = expert_chip;

                if (!is_local) {
                    if constexpr (is_1d_topology<topology>()) {
                        entry->route = get_route<topology, mesh_rows, mesh_cols>(linearized_mesh_coord, expert_chip);
                        entry->distance =
                            manhattan_distance<topology, mesh_rows, mesh_cols>(linearized_mesh_coord, expert_chip);
                    } else {
                        entry->route = 0;
                        entry->distance = 0;
                    }
                } else {
                    entry->route = 0;
                    entry->distance = 0;
                }

                entry_count++;
            }
        }

        if constexpr (!IS_OWNER) {
            noc.async_write(
                CoreLocalMem<uint32_t>(offsets_base_addr),
                UnicastEndpoint{},
                offsets_bytes,
                {},
                {.noc_x = owner_noc_x, .noc_y = owner_noc_y, .addr = offsets_base_addr});
            noc_async_write_barrier();
        }
        if (batch_idx + 1 < effective_total_batches) {
            next_turn_sem.up(noc, next_noc_x, next_noc_y, 1);
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

        plan->entry_count = entry_count;
        cb_push_back(cb_plan_id, 1);
    }

    // Teardown: all batches done — push the two end-of-stream sentinels this core's
    // consumers wait on.
    DPRINT_DISPATCH("[R s={} c={}] loop DONE -> pushing sentinels\n", (uint32_t)dispatch_core_idx, (uint32_t)core_id);
    // (1) Sentinel value to compute so it breaks out of its untilize loop.
    cb_reserve_back(cb_signal_id, 1);
    volatile tt_l1_ptr uint32_t* signal_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_signal_id));
    signal_ptr[0] = ROUTE_INFO_SENTINEL;
    cb_push_back(cb_signal_id, 1);

    // (2) Zero-entry end-of-plan page to the writer (entry_count == 0, entries[0].flags ==
    //     PLAN_FLAG_END) so it stops draining and forwards ROUTE_INFO_SENTINEL to the sender.
    cb_reserve_back(cb_plan_id, 1);
    uint32_t sentinel_addr = get_write_ptr(cb_plan_id);
    volatile tt_l1_ptr PlanHeader* sentinel_plan = reinterpret_cast<volatile tt_l1_ptr PlanHeader*>(sentinel_addr);
    volatile tt_l1_ptr PlanEntry* sentinel_entries =
        reinterpret_cast<volatile tt_l1_ptr PlanEntry*>(sentinel_addr + sizeof(PlanHeader));
    sentinel_plan->entry_count = 0;
    sentinel_entries[0].flags = PLAN_FLAG_END;
    cb_push_back(cb_plan_id, 1);
}
