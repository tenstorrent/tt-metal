// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/dprint.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "ttnn/operations/ccl/common/kernels/moe_utils.hpp"

#define ENABLE_DISPATCH_DEBUG 0

#if ENABLE_DISPATCH_DEBUG
#define DPRINT_DISPATCH DPRINT
#else
#define DPRINT_DISPATCH \
    if (0)              \
    DebugPrinter()
#endif

constexpr uint32_t ROUTE_INFO_SENTINEL = 0xFFFFFFFF;

void kernel_main() {
    using namespace ttnn::operations::ccl::common;

    // ===== Compile Time Args =====
    // CB IDs (indices 0-9)
    constexpr uint32_t cb_input_id = get_compile_time_arg_val(0);
    constexpr uint32_t cb_indices_id = get_compile_time_arg_val(1);
    constexpr uint32_t cb_weights_id = get_compile_time_arg_val(2);
    constexpr uint32_t cb_offsets_id = get_compile_time_arg_val(3);
    constexpr uint32_t cb_route_info_id = get_compile_time_arg_val(4);
    constexpr uint32_t cb_payload_for_writer_id = get_compile_time_arg_val(5);
    constexpr uint32_t cb_metadata_for_writer_id = get_compile_time_arg_val(6);
    constexpr uint32_t cb_metadata_temp_id = get_compile_time_arg_val(7);
    constexpr uint32_t cb_packet_header_id = get_compile_time_arg_val(8);
    constexpr uint32_t cb_dispatch_table_id = get_compile_time_arg_val(9);

    // Page counts (indices 10-16)
    constexpr uint32_t input_pages = get_compile_time_arg_val(10);
    constexpr uint32_t indices_pages = get_compile_time_arg_val(11);
    constexpr uint32_t weights_pages = get_compile_time_arg_val(12);
    constexpr uint32_t offsets_pages = get_compile_time_arg_val(13);
    constexpr uint32_t output_pages = get_compile_time_arg_val(14);
    constexpr uint32_t metadata_pages = get_compile_time_arg_val(15);
    constexpr uint32_t dispatch_table_pages = get_compile_time_arg_val(16);

    // Page sizes (indices 17-23)
    constexpr uint32_t input_page_size = get_compile_time_arg_val(17);
    constexpr uint32_t indices_page_size = get_compile_time_arg_val(18);
    constexpr uint32_t weights_page_size = get_compile_time_arg_val(19);
    constexpr uint32_t output_page_size = get_compile_time_arg_val(21);
    constexpr uint32_t metadata_page_size = get_compile_time_arg_val(22);

    // Operation parameters (indices 24-30)
    constexpr uint32_t num_devices = get_compile_time_arg_val(24);
    constexpr uint32_t hidden_size = get_compile_time_arg_val(25);
    constexpr uint32_t experts_per_chip = get_compile_time_arg_val(26);
    constexpr uint32_t n_routed_experts = get_compile_time_arg_val(27);
    constexpr uint32_t num_experts_per_tok = get_compile_time_arg_val(28);
    constexpr uint32_t metadata_len = get_compile_time_arg_val(29);
    constexpr uint32_t tokens_per_device = get_compile_time_arg_val(30);

    // Mesh information (indices 31-35)
    constexpr uint32_t src_mesh_id = get_compile_time_arg_val(31);
    constexpr uint32_t src_chip_id = get_compile_time_arg_val(32);
    constexpr uint32_t mesh_rows = get_compile_time_arg_val(33);
    constexpr uint32_t mesh_cols = get_compile_time_arg_val(34);
    constexpr uint32_t linearized_mesh_coord = get_compile_time_arg_val(35);

    // Aligned page sizes (indices 36-42)
    constexpr uint32_t aligned_input_page_size = get_compile_time_arg_val(36);
    constexpr uint32_t aligned_indices_page_size = get_compile_time_arg_val(37);
    constexpr uint32_t aligned_weights_page_size = get_compile_time_arg_val(38);
    constexpr uint32_t aligned_offsets_page_size = get_compile_time_arg_val(39);
    constexpr uint32_t aligned_output_page_size = get_compile_time_arg_val(40);
    constexpr uint32_t aligned_metadata_page_size = get_compile_time_arg_val(41);
    constexpr uint32_t aligned_dispatch_table_page_size = get_compile_time_arg_val(42);

    // Fabric configuration (indices 43-46)
    constexpr uint32_t fabric_max_packet_size = get_compile_time_arg_val(43);
    constexpr uint32_t l1_alignment = get_compile_time_arg_val(44);
    constexpr uint32_t num_links = get_compile_time_arg_val(45);
    constexpr tt::tt_fabric::Topology topology = (tt::tt_fabric::Topology)get_compile_time_arg_val(46);

    // Batch configuration (index 47)
    constexpr uint32_t read_batch_size = get_compile_time_arg_val(47);

    // Total dispatch buffer token capacity (shared across all local experts).
    constexpr uint32_t max_dispatch_buffer_token_size = get_compile_time_arg_val(48);

    // TensorAccessorArgs for all 7 tensors (starting at index 49, after the
    // trailing max_dispatch_buffer_token_size arg).
    constexpr auto input_args = TensorAccessorArgs<49>();
    constexpr auto indices_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
    constexpr auto weights_args = TensorAccessorArgs<indices_args.next_compile_time_args_offset()>();
    constexpr auto offsets_args = TensorAccessorArgs<weights_args.next_compile_time_args_offset()>();
    constexpr auto output_args = TensorAccessorArgs<offsets_args.next_compile_time_args_offset()>();
    constexpr auto metadata_args = TensorAccessorArgs<output_args.next_compile_time_args_offset()>();
    constexpr auto dispatch_table_args = TensorAccessorArgs<metadata_args.next_compile_time_args_offset()>();

#ifdef IS_TILE_LAYOUT
    // Reader-only compile-time args (appended after TensorAccessorArgs)
    constexpr uint32_t cb_untilize_id = get_compile_time_arg_val(dispatch_table_args.next_compile_time_args_offset());
    constexpr uint32_t aligned_row_major_input_page_size =
        get_compile_time_arg_val(dispatch_table_args.next_compile_time_args_offset() + 1);
    constexpr uint32_t num_idle_cores =
        get_compile_time_arg_val(dispatch_table_args.next_compile_time_args_offset() + 2);
    constexpr uint32_t cb_signal_id = get_compile_time_arg_val(dispatch_table_args.next_compile_time_args_offset() + 3);
    constexpr uint32_t total_workers =
        get_compile_time_arg_val(dispatch_table_args.next_compile_time_args_offset() + 4);
    constexpr uint32_t cb_route_table_scratch_id =
        get_compile_time_arg_val(dispatch_table_args.next_compile_time_args_offset() + 5);

    constexpr uint32_t tiles_per_row = hidden_size / 32;
    constexpr uint32_t writer_page_size = aligned_row_major_input_page_size;
#else
    constexpr uint32_t writer_page_size = aligned_input_page_size;
#endif

    // ===== Runtime Args =====
    uint32_t rt_args = 0;
    uint32_t input_tensor_address = get_arg_val<uint32_t>(rt_args++);
    uint32_t indices_tensor_address = get_arg_val<uint32_t>(rt_args++);
    uint32_t weights_tensor_address = get_arg_val<uint32_t>(rt_args++);
    uint32_t offsets_tensor_address = get_arg_val<uint32_t>(rt_args++);
    uint32_t output_tensor_address = get_arg_val<uint32_t>(rt_args++);
    uint32_t metadata_tensor_address = get_arg_val<uint32_t>(rt_args++);
    uint32_t dispatch_table_tensor_address = get_arg_val<uint32_t>(rt_args++);
    uint32_t cross_device_semaphore_address = get_arg_val<uint32_t>(rt_args++);
    uint32_t init_semaphore_address = get_arg_val<uint32_t>(rt_args++);
    uint32_t token_start_idx = get_arg_val<uint32_t>(rt_args++);
    uint32_t token_end_idx = get_arg_val<uint32_t>(rt_args++);
    uint32_t dispatch_core_idx = get_arg_val<uint32_t>(rt_args++);
    uint32_t num_dispatch_cores = get_arg_val<uint32_t>(rt_args++);
    uint32_t core_mask = num_dispatch_cores - 1;

#ifdef IS_TILE_LAYOUT
    // Inter-core sync args
    uint32_t data_ready_semaphore_id = get_arg_val<uint32_t>(rt_args++);
    uint32_t start_semaphore_id = get_arg_val<uint32_t>(rt_args++);
    uint32_t addr_ready_semaphore_id = get_arg_val<uint32_t>(rt_args++);
    uint32_t addr_value_semaphore_id = get_arg_val<uint32_t>(rt_args++);
    uint32_t mbox_ready_semaphore_id = get_arg_val<uint32_t>(rt_args++);
    uint32_t mbox_scratch_addr_semaphore_id = get_arg_val<uint32_t>(rt_args++);

    volatile tt_l1_ptr uint32_t* data_ready_sem_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(data_ready_semaphore_id));
    uint32_t start_sem_l1_offset = get_semaphore(start_semaphore_id);
    uint32_t addr_ready_sem_l1_offset = get_semaphore(addr_ready_semaphore_id);
    uint32_t addr_value_sem_l1_offset = get_semaphore(addr_value_semaphore_id);

    // Read per-idle-core NOC addresses for per-batch start signaling (unicast)
    // x/y kept separately to build idle_mailbox_noc_addrs after the mbox_ready rendezvous.
    uint64_t idle_start_noc_addrs[num_idle_cores];
    uint32_t idle_noc_xs[num_idle_cores];
    uint32_t idle_noc_ys[num_idle_cores];
    for (uint32_t c = 0; c < num_idle_cores; c++) {
        idle_noc_xs[c] = get_arg_val<uint32_t>(rt_args++);
        idle_noc_ys[c] = get_arg_val<uint32_t>(rt_args++);
        idle_start_noc_addrs[c] = get_noc_addr(idle_noc_xs[c], idle_noc_ys[c], start_sem_l1_offset);
    }

    // Bounding box covering all idle cores in this sender's group (for multicast)
    uint32_t mcast_x_start = get_arg_val<uint32_t>(rt_args++);
    uint32_t mcast_y_start = get_arg_val<uint32_t>(rt_args++);
    uint32_t mcast_x_end = get_arg_val<uint32_t>(rt_args++);
    uint32_t mcast_y_end = get_arg_val<uint32_t>(rt_args++);
    uint64_t mcast_addr_value_noc_addr =
        get_noc_multicast_addr(mcast_x_start, mcast_y_start, mcast_x_end, mcast_y_end, addr_value_sem_l1_offset);
    uint64_t mcast_addr_ready_noc_addr =
        get_noc_multicast_addr(mcast_x_start, mcast_y_start, mcast_x_end, mcast_y_end, addr_ready_sem_l1_offset);
#endif

#ifdef AXIS
    constexpr ReplicateGroup axis = ReplicateGroup(AXIS);
    constexpr uint32_t row = linearized_mesh_coord / mesh_cols;
    constexpr uint32_t col = linearized_mesh_coord % mesh_cols;
    constexpr uint32_t device_begin_idx = axis == ReplicateGroup::COLS ? col : row * mesh_cols;
    constexpr uint32_t device_end_idx =
        (axis == ReplicateGroup::COLS) ? (col + mesh_rows * mesh_cols) : (row * mesh_cols + mesh_cols);
    constexpr uint32_t device_stride = axis == ReplicateGroup::COLS ? mesh_cols : 1;
#else
    constexpr ReplicateGroup axis = ReplicateGroup::NONE;
    constexpr uint32_t device_begin_idx = 0;
    constexpr uint32_t device_end_idx = num_devices;
    constexpr uint32_t device_stride = 1;
#endif

    DPRINT_DISPATCH << "Reader kernel: tokens=[" << token_start_idx << "," << token_end_idx << ")"
                    << " dispatch_core=" << dispatch_core_idx << "/" << num_dispatch_cores
#ifdef IS_TILE_LAYOUT
                    << " num_idle_cores=" << num_idle_cores
#endif
                    << ENDL();

    // Read offsets into local scratch
    const auto offsets_addr_gen = TensorAccessor(offsets_args, offsets_tensor_address);
    cb_reserve_back(cb_offsets_id, offsets_pages);
    uint32_t offsets_base_addr = get_write_ptr(cb_offsets_id);
    for (uint32_t i = 0; i < offsets_pages; i++) {
        noc_async_read_page(i, offsets_addr_gen, offsets_base_addr + i * aligned_offsets_page_size);
    }
    noc_async_read_barrier();
    uint32_t* offsets = (uint32_t*)offsets_base_addr;

    // Read dispatch table into local scratch
    const auto dispatch_table_addr_gen = TensorAccessor(dispatch_table_args, dispatch_table_tensor_address);
    cb_reserve_back(cb_dispatch_table_id, dispatch_table_pages);
    uint32_t dispatch_table_base_addr = get_write_ptr(cb_dispatch_table_id);
    for (uint32_t i = 0; i < dispatch_table_pages; i++) {
        noc_async_read_page(
            i, dispatch_table_addr_gen, dispatch_table_base_addr + i * aligned_dispatch_table_page_size);
    }
    noc_async_read_barrier();
    int32_t* expert_dispatch_table = (int32_t*)dispatch_table_base_addr;

    // Reserve scratch space once — these CBs are not used as FIFOs. Each batch
    // overwrites the same region at offsets [0, batch_count) without push/pop.
    // DRAM reads are batched to saturate DRAM bandwidth, while the L1-to-writer-CB
    // copies below are done one page at a time — this avoids CB FIFO pointer wrapping
    cb_reserve_back(cb_indices_id, read_batch_size);
    uint32_t indices_base = get_write_ptr(cb_indices_id);
    cb_reserve_back(cb_weights_id, read_batch_size);
    uint32_t weights_base = get_write_ptr(cb_weights_id);

#ifdef IS_TILE_LAYOUT
    const auto indices_addr_gen = TensorAccessor(indices_args, indices_tensor_address, aligned_indices_page_size);
    const auto weights_addr_gen = TensorAccessor(weights_args, weights_tensor_address, aligned_weights_page_size);
    const auto output_addr_gen = TensorAccessor(output_args, output_tensor_address, aligned_output_page_size);
    const auto metadata_addr_gen = TensorAccessor(metadata_args, metadata_tensor_address, aligned_metadata_page_size);
#else
    cb_reserve_back(cb_input_id, read_batch_size);
    uint32_t input_base = get_write_ptr(cb_input_id);
    const auto input_addr_gen = TensorAccessor(input_args, input_tensor_address);
    const auto indices_addr_gen = TensorAccessor(indices_args, indices_tensor_address);
    const auto weights_addr_gen = TensorAccessor(weights_args, weights_tensor_address);
    const auto output_addr_gen = TensorAccessor(output_args, output_tensor_address);
    const auto metadata_addr_gen = TensorAccessor(metadata_args, metadata_tensor_address);
#endif

    cb_reserve_back(cb_metadata_temp_id, 1);
    uint32_t metadata_temp_addr = get_write_ptr(cb_metadata_temp_id);

#ifdef IS_TILE_LAYOUT
    uint32_t untilize_base = get_write_ptr(cb_untilize_id);
    uint32_t rt_scratch_base = get_write_ptr(cb_route_table_scratch_id);

    // Multicast two addresses to all idle cores in this sender's group before signaling addr_ready:
    //   1. receive buffer address (c_18 base) → idle cores write untilized data here
    //   2. route-table scratch base → idle cores NOC-write their mailbox L1 address into
    //      slot [core_id * 4] of this buffer so the sender can read it locally (no NOC read).
    volatile tt_l1_ptr uint32_t* addr_value_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr_value_sem_l1_offset);
    *addr_value_ptr = untilize_base;
    noc_async_write_multicast(addr_value_sem_l1_offset, mcast_addr_value_noc_addr, sizeof(uint32_t), num_idle_cores);

    uint32_t mbox_scratch_addr_sem_l1_offset = get_semaphore(mbox_scratch_addr_semaphore_id);
    volatile tt_l1_ptr uint32_t* mbox_scratch_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mbox_scratch_addr_sem_l1_offset);
    *mbox_scratch_addr_ptr = rt_scratch_base;
    uint64_t mcast_mbox_scratch_noc_addr =
        get_noc_multicast_addr(mcast_x_start, mcast_y_start, mcast_x_end, mcast_y_end, mbox_scratch_addr_sem_l1_offset);
    noc_async_write_multicast(
        mbox_scratch_addr_sem_l1_offset, mcast_mbox_scratch_noc_addr, sizeof(uint32_t), num_idle_cores);
    noc_async_write_barrier();
    noc_semaphore_inc_multicast(mcast_addr_ready_noc_addr, 1, num_idle_cores);
    noc_async_atomic_barrier();

    // Wait for all idle cores to NOC-write their mailbox L1 addresses into rt_scratch_base.
    // Because idle cores use noc_inline_dw_write (ordered before noc_semaphore_inc on the NOC),
    // each slot is guaranteed to be visible in local L1 once mbox_ready reaches num_idle_cores.
    volatile tt_l1_ptr uint32_t* mbox_ready_sem_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(mbox_ready_semaphore_id));
    noc_semaphore_wait(mbox_ready_sem_ptr, num_idle_cores);
    noc_semaphore_set(mbox_ready_sem_ptr, 0);

    // Read idle mailbox addresses directly from own L1 scratch — no NOC reads needed.
    volatile tt_l1_ptr uint32_t* rt_scratch_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(rt_scratch_base);
    uint64_t idle_mailbox_noc_addrs[num_idle_cores];
    for (uint32_t c = 0; c < num_idle_cores; c++) {
        idle_mailbox_noc_addrs[c] = get_noc_addr(idle_noc_xs[c], idle_noc_ys[c], rt_scratch_ptr[c]);
    }

    // Self-untilize setup: TensorAccessor for reading tiles from DRAM
    const auto self_input_addr_gen = TensorAccessor(input_args, input_tensor_address, aligned_input_page_size);
    constexpr uint32_t block_ct_dim = 8;
    constexpr uint32_t num_tile_blocks = tiles_per_row / block_ct_dim;
#else
    // Prefetch first batch of DRAM reads
    uint32_t first_batch_end =
        (token_start_idx + read_batch_size < token_end_idx) ? token_start_idx + read_batch_size : token_end_idx;
    uint32_t first_batch_count = first_batch_end - token_start_idx;
    if (first_batch_count > 0) {
        for (uint32_t t = 0; t < first_batch_count; t++) {
            noc_async_read_page(token_start_idx + t, input_addr_gen, input_base + t * aligned_input_page_size);
            noc_async_read_page(token_start_idx + t, indices_addr_gen, indices_base + t * aligned_indices_page_size);
            noc_async_read_page(token_start_idx + t, weights_addr_gen, weights_base + t * aligned_weights_page_size);
        }
        noc_async_read_barrier();
    }
#endif

    // ===== Unified batch loop =====
    uint32_t total_batches = (token_end_idx - token_start_idx + read_batch_size - 1) / read_batch_size;
    for (uint32_t B = 0; B < total_batches; B++) {
        uint32_t batch_start = token_start_idx + B * read_batch_size;
        uint32_t batch_end =
            (batch_start + read_batch_size < token_end_idx) ? batch_start + read_batch_size : token_end_idx;
        uint32_t batch_count = batch_end - batch_start;
        bool batch_did_local_write = false;

#ifdef IS_TILE_LAYOUT
        // Batch prep: delegate to idle core, or self-untilize locally.
        uint32_t C = B % total_workers;

        if (C < num_idle_cores) {
            // ---- Worker-batch: delegate to idle core C ----
            // Read indices/weights before building the route table so routing
            // decisions are known before we signal the idle core.
            for (uint32_t t = 0; t < batch_count; t++) {
                noc_async_read_page(batch_start + t, indices_addr_gen, indices_base + t * aligned_indices_page_size);
                noc_async_read_page(batch_start + t, weights_addr_gen, weights_base + t * aligned_weights_page_size);
            }
            noc_async_read_barrier();

            // Build local-expert route table into the route table scratch CB.
            // Only LOCAL entries go in the mailbox; cross-device entries are
            // handled by the sender's main routing loop as normal.
            // We track per-expert offset increments locally to mirror the main
            // loop without touching the shared offsets[] array.
            uint32_t local_offset_delta[n_routed_experts] = {};
            volatile tt_l1_ptr uint32_t* rt = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(rt_scratch_base);
            uint32_t entry_count = 0;
            bool has_non_local = false;

            for (uint32_t t = 0; t < batch_count; t++) {
                int32_t* indices_t = (int32_t*)(indices_base + t * aligned_indices_page_size);
                uint16_t* weights_t = (uint16_t*)(weights_base + t * aligned_weights_page_size);
                for (uint32_t k = 0; k < num_experts_per_tok; k++) {
                    auto routed_expert = indices_t[k];
                    if (((uint32_t)routed_expert & core_mask) != dispatch_core_idx) {
                        continue;
                    }
                    auto expert_chip_og = expert_dispatch_table[routed_expert];
                    if (expert_chip_og == -1) {
                        continue;
                    }
                    // Mirror offset++ from main loop (covers both overflow and valid paths)
                    uint32_t effective_offset = offsets[routed_expert] + local_offset_delta[routed_expert];
                    local_offset_delta[routed_expert]++;

                    if (effective_offset >= max_dispatch_buffer_token_size) {
                        continue;
                    }
                    auto expert_chip = device_begin_idx + expert_chip_og * device_stride;
                    if (expert_chip != linearized_mesh_coord) {
                        has_non_local = true;
                        continue;  // cross-device: sender handles in main routing loop
                    }
                    auto page_idx = effective_offset;

                    uint32_t base = 1 + entry_count * 6;
                    rt[base + 0] = t;
                    rt[base + 1] = (uint32_t)page_idx;
                    rt[base + 2] = batch_start + t;
                    rt[base + 3] = k;
                    rt[base + 4] = (uint32_t)routed_expert;
                    rt[base + 5] = (uint32_t)(int32_t)(int16_t)weights_t[k];
                    entry_count++;
                }
            }
            // High bit signals the idle core whether to bulk-send back to us.
            rt[0] = entry_count | (has_non_local ? 0x80000000u : 0u);

            // Write route table to idle core's mailbox, then signal start.
            uint32_t mailbox_write_bytes = sizeof(uint32_t) + entry_count * 6 * sizeof(uint32_t);
            noc_async_write(rt_scratch_base, idle_mailbox_noc_addrs[C], mailbox_write_bytes);
            noc_async_write_barrier();

            noc_semaphore_inc(idle_start_noc_addrs[C], 1);
            noc_async_atomic_barrier();

            // Only wait if the idle core will actually send data back.
            if (has_non_local) {
                DPRINT_DISPATCH << "Waiting for idle core " << C << " batch " << B << ENDL();
                noc_semaphore_wait(data_ready_sem_ptr, 1);
                noc_semaphore_set(data_ready_sem_ptr, 0);
                DPRINT_DISPATCH << "Got batch " << B << " from idle core " << C << ENDL();
            }
        } else {
            // ---- Self-batch: read tiles, untilize locally, no NOC transfer ----
            DPRINT_DISPATCH << "Self-untilize batch " << B << ENDL();
            uint32_t tile_base_page = B * tiles_per_row;

            // Signal compute to untilize (before streaming tiles so compute is ready)
            cb_reserve_back(cb_signal_id, 1);
            volatile tt_l1_ptr uint32_t* signal_ptr =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_signal_id));
            signal_ptr[0] = 0x00000000;
            cb_push_back(cb_signal_id, 1);

            // Stream tiles in blocks of 8 — compute consumes each block as it arrives
            for (uint32_t blk = 0; blk < num_tile_blocks; blk++) {
                cb_reserve_back(cb_input_id, block_ct_dim);
                uint32_t blk_write_ptr = get_write_ptr(cb_input_id);
                uint32_t blk_start = tile_base_page + blk * block_ct_dim;
                for (uint32_t col = 0; col < block_ct_dim; col++) {
                    noc_async_read_page(
                        blk_start + col, self_input_addr_gen, blk_write_ptr + col * aligned_input_page_size);
                }
                noc_async_read_barrier();
                cb_push_back(cb_input_id, block_ct_dim);
            }

            // Overlap reads with compute
            for (uint32_t t = 0; t < batch_count; t++) {
                noc_async_read_page(batch_start + t, indices_addr_gen, indices_base + t * aligned_indices_page_size);
                noc_async_read_page(batch_start + t, weights_addr_gen, weights_base + t * aligned_weights_page_size);
            }

            // Wait for compute to finish (it writes untilized data into cb_untilize_id / c_18)
            cb_wait_front(cb_untilize_id, read_batch_size);
            noc_async_read_barrier();
            DPRINT_DISPATCH << "Self-untilize batch " << B << " done" << ENDL();
        }
#endif

        // ---- Common inner token loop ----
#ifdef IS_TILE_LAYOUT
        bool is_delegated_batch = (C < num_idle_cores);
#endif
        for (uint32_t t = 0; t < batch_count; t++) {
            uint32_t token_idx = batch_start + t;
#ifdef IS_TILE_LAYOUT
            uint32_t token_input_addr = untilize_base + t * writer_page_size;
#else
            uint32_t token_input_addr = input_base + t * aligned_input_page_size;
#endif
            int32_t* indices = (int32_t*)(indices_base + t * aligned_indices_page_size);
            uint16_t* weights = (uint16_t*)(weights_base + t * aligned_weights_page_size);

            for (uint32_t k = 0; k < num_experts_per_tok; ++k) {
                auto routed_expert = indices[k];

                if (((uint32_t)routed_expert & core_mask) != dispatch_core_idx) {
                    continue;
                }

                auto expert_chip_og = expert_dispatch_table[routed_expert];
                if (expert_chip_og == -1) {
                    continue;
                }
                auto expert_chip = device_begin_idx + expert_chip_og * device_stride;
                auto& offset = offsets[routed_expert];
                if (offset >= max_dispatch_buffer_token_size) {
                    // Token would overflow the dispatch buffer - skip to prevent
                    // out-of-bounds DRAM writes that corrupt memory and cause hangs.
                    offset++;
                    continue;
                }
                auto page_idx = offset;

                if (expert_chip == linearized_mesh_coord) {
#ifdef IS_TILE_LAYOUT
                    // For delegated batches the idle core's writer kernel handles
                    // local DRAM writes directly; skip them here.
                    if (!is_delegated_batch)
#endif
                    {
                        volatile tt_l1_ptr int32_t* metadata =
                            reinterpret_cast<volatile tt_l1_ptr int32_t*>(metadata_temp_addr);
                        metadata[0] = linearized_mesh_coord;
                        metadata[1] = token_idx;
                        metadata[2] = k;
                        metadata[3] = routed_expert;
                        metadata[4] = static_cast<int16_t>(weights[k]);

                        noc_async_write_page(page_idx, output_addr_gen, token_input_addr);
                        noc_async_write_page(page_idx, metadata_addr_gen, metadata_temp_addr);
                        noc_async_writes_flushed();
                        batch_did_local_write = true;
                    }
                } else {
                    if constexpr (is_1d_topology<topology>()) {
                        uint32_t route = get_route<topology, mesh_rows, mesh_cols>(linearized_mesh_coord, expert_chip);
                        uint32_t distance =
                            manhattan_distance<topology, mesh_rows, mesh_cols>(linearized_mesh_coord, expert_chip);

                        cb_reserve_back(cb_route_info_id, 1);
                        volatile tt_l1_ptr uint32_t* route_info =
                            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_route_info_id));
                        route_info[0] = route;
                        route_info[1] = distance;
                        route_info[2] = page_idx;
                        route_info[3] = 0;
                        cb_push_back(cb_route_info_id, 1);

                        cb_reserve_back(cb_payload_for_writer_id, 1);
                        uint32_t payload_dst = get_write_ptr(cb_payload_for_writer_id);
                        noc_async_read(get_noc_addr(token_input_addr), payload_dst, writer_page_size);
                        noc_async_read_barrier();
                        cb_push_back(cb_payload_for_writer_id, 1);

                        cb_reserve_back(cb_metadata_for_writer_id, 1);
                        volatile tt_l1_ptr int32_t* meta_dst =
                            reinterpret_cast<volatile tt_l1_ptr int32_t*>(get_write_ptr(cb_metadata_for_writer_id));
                        meta_dst[0] = linearized_mesh_coord;
                        meta_dst[1] = token_idx;
                        meta_dst[2] = k;
                        meta_dst[3] = routed_expert;
                        meta_dst[4] = static_cast<int16_t>(weights[k]);
                        cb_push_back(cb_metadata_for_writer_id, 1);
                    }
                }

                offset++;
            }
        }

#ifdef IS_TILE_LAYOUT
        if (batch_did_local_write) {
            noc_async_write_barrier();
        }
        // Release CB after self-batch so compute can reuse it next time
        if (C >= num_idle_cores) {
            cb_pop_front(cb_untilize_id, read_batch_size);
        }
#else
        // Issue next batch DRAM reads BEFORE write barrier to overlap read/write NOC channels
        uint32_t next_batch_start = batch_start + read_batch_size;
        bool has_next_batch = (next_batch_start < token_end_idx);
        if (has_next_batch) {
            uint32_t next_batch_end = (next_batch_start + read_batch_size < token_end_idx)
                                          ? next_batch_start + read_batch_size
                                          : token_end_idx;
            uint32_t next_batch_count = next_batch_end - next_batch_start;
            for (uint32_t t = 0; t < next_batch_count; t++) {
                noc_async_read_page(next_batch_start + t, input_addr_gen, input_base + t * aligned_input_page_size);
                noc_async_read_page(
                    next_batch_start + t, indices_addr_gen, indices_base + t * aligned_indices_page_size);
                noc_async_read_page(
                    next_batch_start + t, weights_addr_gen, weights_base + t * aligned_weights_page_size);
            }
        }

        if (batch_did_local_write) {
            noc_async_write_barrier();
        }

        if (has_next_batch) {
            noc_async_read_barrier();
        }
#endif
    }

#ifdef IS_TILE_LAYOUT
    // Signal compute to exit
    cb_reserve_back(cb_signal_id, 1);
    volatile tt_l1_ptr uint32_t* signal_sentinel =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_signal_id));
    signal_sentinel[0] = ROUTE_INFO_SENTINEL;
    cb_push_back(cb_signal_id, 1);
#endif

    // Push sentinel to signal writer that all dispatches are done
    cb_reserve_back(cb_route_info_id, 1);
    volatile tt_l1_ptr uint32_t* route_info =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_route_info_id));
    route_info[0] = ROUTE_INFO_SENTINEL;
    route_info[1] = 0;
    route_info[2] = 0;
    route_info[3] = 0;
    cb_push_back(cb_route_info_id, 1);
}
