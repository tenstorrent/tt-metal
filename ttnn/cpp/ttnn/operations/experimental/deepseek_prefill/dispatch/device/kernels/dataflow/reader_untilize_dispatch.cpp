// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//
// Untilize core RISCV_1 — input prefetch + per-token routing.
//
// At startup: read the full expert offsets[] and expert dispatch_table[] tensors
// from DRAM into local L1 scratch (c_3, c_9). offsets[] is owned and mutated
// exclusively by this RISC — there is no cross-core synchronization because each
// sender (and thus its single owning untilize core) processes a disjoint set of
// experts via the dispatch_core_idx & core_mask filter.
//
// Per batch:
//   1. Signal compute to untilize this batch (existing).
//   2. Stream tiled input from DRAM → c_0 (existing).
//   3. Read this batch's indices and weights pages from DRAM → c_1, c_2.
//   4. Walk each token's top-k:
//        * filter out experts not handled by this sender (core_mask),
//        * filter out experts with dispatch_table == -1,
//        * compute page_idx from the local offsets[expert] counter (mirrors what
//          the old sender's main routing loop did),
//        * classify local vs cross-device, record route/distance for cross-device.
//      Build the per-batch route plan into c_14.
//   5. cb_push_back(c_14) — writer RISC drains the plan and executes the data
//      movement (NOC1 → DRAM for local, 2-slot push to sender c_4/c_5/c_6 for
//      cross-device).
//
// After the last batch: push a single sentinel entry on c_14 so the writer
// knows to forward the final ROUTE_INFO_SENTINEL to the sender.
//

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/dprint.h"
#include "ttnn/operations/ccl/common/kernels/moe_utils.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "tt_metal/tools/profiler/kernel_profiler.hpp"
#define ENABLE_DISPATCH_DEBUG 0
#if ENABLE_DISPATCH_DEBUG
#define DPRINT_DISPATCH DPRINT
#else
#define DPRINT_DISPATCH \
    if (0)              \
    DebugPrinter()
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
    // u1 (core_id=0): even batches, increments offset from start (left-to-right).
    // u2 (core_id=1): odd batches, decrements offset from end  (right-to-left).
    constexpr bool IS_RIGHT_UNTILIZER = (core_id == 1);

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
    constexpr auto end_offsets_args = TensorAccessorArgs<dispatch_table_args.next_compile_time_args_offset()>();

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
    uint32_t end_offsets_tensor_address = get_arg_val<uint32_t>(rt_idx++);
    uint32_t dispatch_table_tensor_address = get_arg_val<uint32_t>(rt_idx++);
    uint32_t token_start_idx = get_arg_val<uint32_t>(rt_idx++);
    uint32_t token_end_idx = get_arg_val<uint32_t>(rt_idx++);

    const auto input_addr_gen = TensorAccessor(input_args, input_tensor_address, aligned_input_page_size);
    const auto indices_addr_gen = TensorAccessor(indices_args, indices_tensor_address, aligned_indices_page_size);
    const auto weights_addr_gen = TensorAccessor(weights_args, weights_tensor_address, aligned_weights_page_size);
    const auto offsets_addr_gen = TensorAccessor(offsets_args, offsets_tensor_address);
    const auto dispatch_table_addr_gen = TensorAccessor(dispatch_table_args, dispatch_table_tensor_address);

    // ===== Startup: load offsets[] and dispatch_table[] into local L1 =====
    // u1 loads tt_expert_offsets (start, increments left-to-right).
    // u2 loads tt_end_offsets    (end,   decrements right-to-left).
    cb_reserve_back(cb_offsets_id, offsets_pages);
    uint32_t offsets_base_addr = get_write_ptr(cb_offsets_id);
    if constexpr (IS_RIGHT_UNTILIZER) {
        const auto end_offsets_addr_gen = TensorAccessor(end_offsets_args, end_offsets_tensor_address);
        for (uint32_t i = 0; i < offsets_pages; i++) {
            noc_async_read_page(i, end_offsets_addr_gen, offsets_base_addr + i * aligned_offsets_page_size);
        }
    } else {
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
    uint32_t* offsets = reinterpret_cast<uint32_t*>(offsets_base_addr);
    int32_t* expert_dispatch_table = reinterpret_cast<int32_t*>(dispatch_table_base_addr);

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
        {
            // DeviceZoneScopedN("read_input_for_current_batch")
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
        }

        // 3. Read this batch's indices and weights pages
        {
            // DeviceZoneScopedN("read_indices_and_weights")
            for (uint32_t t = 0; t < batch_count; t++) {
                noc_async_read_page(batch_start + t, indices_addr_gen, indices_base + t * aligned_indices_page_size);
                noc_async_read_page(batch_start + t, weights_addr_gen, weights_base + t * aligned_weights_page_size);
            }
            noc_async_read_barrier();
        }

        // 4. Build per-batch route plan into c_14.
        cb_reserve_back(cb_plan_id, 1);
        uint32_t plan_addr = get_write_ptr(cb_plan_id);
        volatile tt_l1_ptr uint32_t* plan = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(plan_addr);
        uint32_t entry_count = 0;
        uint32_t entry_off = 8;  // entries start at u32 offset 8 (32B header)

        {
            // DeviceZoneScopedN("build_plan_for_32_tokens")
            for (uint32_t t = 0; t < batch_count; t++) {
                int32_t* indices_t = reinterpret_cast<int32_t*>(indices_base + t * aligned_indices_page_size);
                uint16_t* weights_t = reinterpret_cast<uint16_t*>(weights_base + t * aligned_weights_page_size);
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

                    uint32_t& offset = offsets[routed_expert];
                    uint32_t page_idx;
                    if constexpr (IS_RIGHT_UNTILIZER) {
                        // Decrement before use: end is exclusive, so first write is at end-1.
                        // Guard: if offset is 0 it would wrap to UINT32_MAX; if the result
                        // exceeds the buffer it is out-of-bounds — both are skipped.
                        if (offset == 0) {
                            continue;
                        }
                        page_idx = --offset;
                        if (page_idx >= max_dispatch_buffer_token_size) {
                            offset++;
                            continue;
                        }
                    } else {
                        if (offset >= max_dispatch_buffer_token_size) {
                            offset++;
                            continue;
                        }
                        page_idx = offset++;
                    }

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
                            uint32_t route =
                                get_route<topology, mesh_rows, mesh_cols>(linearized_mesh_coord, expert_chip);
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
        }

        plan[0] = entry_count;
        cb_push_back(cb_plan_id, 1);

        DPRINT_DISPATCH << "Reader untilize batch=" << batch_idx << " entries=" << entry_count << ENDL();
    }

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
