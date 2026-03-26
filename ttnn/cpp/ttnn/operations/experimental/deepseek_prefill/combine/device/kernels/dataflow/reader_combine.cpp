// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/dprint.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "ttnn/operations/ccl/common/kernels/moe_utils.hpp"

#define ENABLE_COMBINE_DEBUG 0
#if ENABLE_COMBINE_DEBUG
#define DPRINT_COMBINE DPRINT
#else
#define DPRINT_COMBINE \
    if (0)             \
    DebugPrinter()
#endif

// Signal last element to writer to break out of loop
constexpr uint32_t ROUTE_INFO_SENTINEL = 0xFFFFFFFF;

void kernel_main() {
    using namespace ttnn::operations::ccl::common;

    // ===== Compile Time Args =====
    // CB IDs (indices 0-5)
    constexpr uint32_t cb_dispatched_buffer_id = get_compile_time_arg_val(0);
    constexpr uint32_t cb_dispatched_metadata_id = get_compile_time_arg_val(1);
    constexpr uint32_t cb_experts_tok_counter_id = get_compile_time_arg_val(2);
    constexpr uint32_t cb_route_info_id = get_compile_time_arg_val(3);
    constexpr uint32_t cb_output_for_writer_id = get_compile_time_arg_val(4);
    constexpr uint32_t cb_packet_header_id = get_compile_time_arg_val(5);

    // Page counts (indices 6-9)
    constexpr uint32_t dispatched_buffer_pages = get_compile_time_arg_val(6);
    constexpr uint32_t dispatched_metadata_pages = get_compile_time_arg_val(7);
    constexpr uint32_t experts_tok_counter_pages = get_compile_time_arg_val(8);
    constexpr uint32_t output_pages = get_compile_time_arg_val(9);

    // Page sizes (indices 10-13)
    constexpr uint32_t dispatched_buffer_page_size = get_compile_time_arg_val(10);
    constexpr uint32_t dispatched_metadata_page_size = get_compile_time_arg_val(11);
    constexpr uint32_t experts_tok_counter_page_size = get_compile_time_arg_val(12);
    constexpr uint32_t output_page_size = get_compile_time_arg_val(13);

    // Operation parameters (indices 14-18)
    constexpr uint32_t num_chips = get_compile_time_arg_val(14);
    constexpr uint32_t experts_per_chip = get_compile_time_arg_val(15);
    constexpr uint32_t num_experts_per_tok = get_compile_time_arg_val(16);
    constexpr uint32_t seq_len_per_chip = get_compile_time_arg_val(17);
    constexpr uint32_t max_dispatched_tokens_per_expert = get_compile_time_arg_val(18);

    // Hidden dimension (index 19)
    constexpr uint32_t hidden_size = get_compile_time_arg_val(19);

    // Aligned page sizes (indices 20-23)
    constexpr uint32_t aligned_dispatched_buffer_page_size = get_compile_time_arg_val(20);
    constexpr uint32_t aligned_dispatched_metadata_page_size = get_compile_time_arg_val(21);
    constexpr uint32_t aligned_experts_tok_counter_page_size = get_compile_time_arg_val(22);
    constexpr uint32_t aligned_output_page_size = get_compile_time_arg_val(23);

    // Mesh information (indices 24-28)
    constexpr uint32_t src_mesh_id = get_compile_time_arg_val(24);
    constexpr uint32_t src_chip_id = get_compile_time_arg_val(25);
    constexpr uint32_t mesh_rows = get_compile_time_arg_val(26);
    constexpr uint32_t mesh_cols = get_compile_time_arg_val(27);
    constexpr uint32_t linearized_mesh_coord = get_compile_time_arg_val(28);

    // Fabric configuration (indices 29-32)
    constexpr uint32_t fabric_max_packet_size = get_compile_time_arg_val(29);
    constexpr uint32_t l1_alignment = get_compile_time_arg_val(30);
    constexpr uint32_t num_links = get_compile_time_arg_val(31);
    constexpr tt::tt_fabric::Topology topology = (tt::tt_fabric::Topology)get_compile_time_arg_val(32);

    // TensorAccessorArgs for all 4 tensors (starting at index 33)
    constexpr auto dispatched_buffer_args = TensorAccessorArgs<33>();
    constexpr auto dispatched_metadata_args =
        TensorAccessorArgs<dispatched_buffer_args.next_compile_time_args_offset()>();
    constexpr auto experts_tok_counter_args =
        TensorAccessorArgs<dispatched_metadata_args.next_compile_time_args_offset()>();
    constexpr auto output_args = TensorAccessorArgs<experts_tok_counter_args.next_compile_time_args_offset()>();

#if INIT_ZEROS
    // Zero-init args follow immediately after the TensorAccessorArgs block
    constexpr uint32_t zi_cb_id = get_compile_time_arg_val(output_args.next_compile_time_args_offset());
    constexpr uint32_t num_idle_cores = get_compile_time_arg_val(output_args.next_compile_time_args_offset() + 1);
#endif

    // ===== Runtime Args =====
    uint32_t rt_args = 0;
    uint32_t dispatched_buffer_addr = get_arg_val<uint32_t>(rt_args++);
    uint32_t dispatched_metadata_addr = get_arg_val<uint32_t>(rt_args++);
    uint32_t experts_tok_counter_addr = get_arg_val<uint32_t>(rt_args++);
    uint32_t output_addr = get_arg_val<uint32_t>(rt_args++);
    uint32_t zero_init_semaphore_id = get_arg_val<uint32_t>(rt_args++);
    uint32_t zero_init_barrier_semaphore_id = get_arg_val<uint32_t>(rt_args++);
    uint32_t num_cores = get_arg_val<uint32_t>(rt_args++);
    uint32_t expert_start_idx = get_arg_val<uint32_t>(rt_args++);
    uint32_t expert_end_idx = get_arg_val<uint32_t>(rt_args++);
    uint32_t zero_init_semaphore_address = get_semaphore(zero_init_semaphore_id);
    uint32_t zero_init_barrier_address = get_semaphore(zero_init_barrier_semaphore_id);

    DPRINT_COMBINE << "Combine Reader: experts=[" << expert_start_idx << "," << expert_end_idx << ")"
                   << " linearized_mesh_coord=" << linearized_mesh_coord << ENDL();

    const auto output_addr_gen = TensorAccessor(output_args, output_addr, aligned_output_page_size);

#if INIT_ZEROS
    // Hybrid row zero-init: this core zeroes its assigned page range, then waits for idle row cores
    {
        uint32_t page_start = get_arg_val<uint32_t>(rt_args++);
        uint32_t page_end = get_arg_val<uint32_t>(rt_args++);
        uint32_t zi_done_semaphore_id = get_arg_val<uint32_t>(rt_args++);
        uint32_t zi_done_sem_address = get_semaphore(zi_done_semaphore_id);

        cb_reserve_back(zi_cb_id, 1);
        uint32_t zero_buf = get_write_ptr(zi_cb_id);
        uint64_t zeros_noc = get_noc_addr(NOC_X(my_x[0]), NOC_Y(my_y[0]), MEM_ZEROS_BASE);
        for (uint32_t off = 0; off < NOC_MAX_BURST_SIZE; off += MEM_ZEROS_SIZE) {
            uint32_t chunk = ((uint32_t)MEM_ZEROS_SIZE < (NOC_MAX_BURST_SIZE - off)) ? (uint32_t)MEM_ZEROS_SIZE
                                                                                     : (NOC_MAX_BURST_SIZE - off);
            noc_async_read(zeros_noc, zero_buf + off, chunk);
        }
        noc_async_read_barrier();

        for (uint32_t page = page_start; page < page_end; page++) {
            uint64_t page_noc_addr = get_noc_addr(page, output_addr_gen);
            uint32_t remaining = aligned_output_page_size;
            while (remaining > 0) {
                uint32_t curr = (remaining > (uint32_t)NOC_MAX_BURST_SIZE) ? (uint32_t)NOC_MAX_BURST_SIZE : remaining;
                noc_async_write(zero_buf, page_noc_addr, curr);
                page_noc_addr += curr;
                remaining -= curr;
            }
        }

        volatile tt_l1_ptr uint32_t* zi_done_sem_ptr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(zi_done_sem_address);
        noc_semaphore_wait(zi_done_sem_ptr, num_idle_cores);
        noc_semaphore_set(zi_done_sem_ptr, 0);
    }
#endif

    // Signal writer that zero-init is complete
    volatile tt_l1_ptr uint32_t* zero_init_sem_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(zero_init_semaphore_address);
    noc_semaphore_set(zero_init_sem_ptr, 1);

    // Wait for ALL writers (all cores) to complete init exchange.
    // Each writer signals all readers' barrier sems via noc_semaphore_inc,
    // so this reader waits for num_cores signals before proceeding.
    volatile tt_l1_ptr uint32_t* barrier_sem_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(zero_init_barrier_address);
    noc_semaphore_wait(barrier_sem_ptr, num_cores);
    noc_semaphore_set(barrier_sem_ptr, 0);

    // Read expert token counts
    const auto experts_tok_counter_addr_gen =
        TensorAccessor(experts_tok_counter_args, experts_tok_counter_addr, aligned_experts_tok_counter_page_size);
    cb_reserve_back(cb_experts_tok_counter_id, experts_tok_counter_pages);
    uint32_t counter_base_addr = get_write_ptr(cb_experts_tok_counter_id);
    for (uint32_t i = 0; i < experts_tok_counter_pages; i++) {
        noc_async_read_page(
            i, experts_tok_counter_addr_gen, counter_base_addr + i * aligned_experts_tok_counter_page_size);
    }
    noc_async_read_barrier();
    volatile tt_l1_ptr uint32_t* experts_tok_counter_l1 =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(counter_base_addr);

    // Set up scratch buffers for batched reads
    constexpr uint32_t read_batch_size = 8;
    cb_reserve_back(cb_dispatched_buffer_id, read_batch_size);
    uint32_t buffer_base = get_write_ptr(cb_dispatched_buffer_id);
    cb_reserve_back(cb_dispatched_metadata_id, read_batch_size);
    uint32_t metadata_base = get_write_ptr(cb_dispatched_metadata_id);

    const auto dispatched_buffer_addr_gen =
        TensorAccessor(dispatched_buffer_args, dispatched_buffer_addr, aligned_dispatched_buffer_page_size);
    const auto dispatched_metadata_addr_gen =
        TensorAccessor(dispatched_metadata_args, dispatched_metadata_addr, aligned_dispatched_metadata_page_size);

    constexpr auto expert_stride = max_dispatched_tokens_per_expert;

    // Process each expert in assigned range
    for (uint32_t local_expert = expert_start_idx; local_expert < expert_end_idx; local_expert++) {
        uint32_t start_page = local_expert * expert_stride;
        uint32_t expert_tokens = experts_tok_counter_l1[local_expert];
        uint32_t end_page = start_page + expert_tokens;

        DPRINT_COMBINE << "Expert=" << local_expert << " tokens=" << expert_tokens << ENDL();

        // Prefetch first batch
        uint32_t first_batch_end = (start_page + read_batch_size < end_page) ? start_page + read_batch_size : end_page;
        uint32_t first_batch_count = first_batch_end - start_page;
        if (first_batch_count > 0) {
            for (uint32_t t = 0; t < first_batch_count; t++) {
                noc_async_read_page(
                    start_page + t, dispatched_buffer_addr_gen, buffer_base + t * aligned_dispatched_buffer_page_size);
                noc_async_read_page(
                    start_page + t,
                    dispatched_metadata_addr_gen,
                    metadata_base + t * aligned_dispatched_metadata_page_size);
            }
            noc_async_read_barrier();
        }

        for (uint32_t batch_start = start_page; batch_start < end_page; batch_start += read_batch_size) {
            uint32_t batch_end = (batch_start + read_batch_size < end_page) ? batch_start + read_batch_size : end_page;
            uint32_t batch_count = batch_end - batch_start;
            bool batch_did_local_write = false;

            for (uint32_t t = 0; t < batch_count; t++) {
                uint32_t buffer_scratch_addr = buffer_base + t * aligned_dispatched_buffer_page_size;
                uint32_t metadata_scratch_addr = metadata_base + t * aligned_dispatched_metadata_page_size;
                volatile tt_l1_ptr uint32_t* metadata =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(metadata_scratch_addr);

                auto dst_chip = metadata[0];
                auto dst_token_idx = metadata[1];
                auto dst_topk_indice = metadata[2];
                uint32_t output_page_idx = dst_token_idx * num_experts_per_tok + dst_topk_indice;

                if (dst_chip == linearized_mesh_coord) {
                    noc_async_write_page(output_page_idx, output_addr_gen, buffer_scratch_addr);
                    noc_async_writes_flushed();
                    batch_did_local_write = true;
                } else {
                    if constexpr (is_1d_topology<topology>()) {
                        uint32_t route = get_route<topology, mesh_rows, mesh_cols>(linearized_mesh_coord, dst_chip);
                        uint32_t distance =
                            manhattan_distance<topology, mesh_rows, mesh_cols>(linearized_mesh_coord, dst_chip);

                        // Push route info to writer
                        cb_reserve_back(cb_route_info_id, 1);
                        volatile tt_l1_ptr uint32_t* route_info =
                            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_route_info_id));
                        route_info[0] = route;
                        route_info[1] = distance;
                        route_info[2] = output_page_idx;
                        route_info[3] = 0;
                        cb_push_back(cb_route_info_id, 1);

                        // Push output payload to writer
                        cb_reserve_back(cb_output_for_writer_id, 1);
                        uint32_t output_dst = get_write_ptr(cb_output_for_writer_id);
                        noc_async_read(
                            get_noc_addr(buffer_scratch_addr), output_dst, aligned_dispatched_buffer_page_size);
                        noc_async_read_barrier();
                        cb_push_back(cb_output_for_writer_id, 1);
                    }
                }
            }

            // Issue next batch reads BEFORE write barrier
            uint32_t next_batch_start = batch_start + read_batch_size;
            bool has_next_batch = (next_batch_start < end_page);
            if (has_next_batch) {
                uint32_t next_batch_end =
                    (next_batch_start + read_batch_size < end_page) ? next_batch_start + read_batch_size : end_page;
                uint32_t next_batch_count = next_batch_end - next_batch_start;
                for (uint32_t t = 0; t < next_batch_count; t++) {
                    noc_async_read_page(
                        next_batch_start + t,
                        dispatched_buffer_addr_gen,
                        buffer_base + t * aligned_dispatched_buffer_page_size);
                    noc_async_read_page(
                        next_batch_start + t,
                        dispatched_metadata_addr_gen,
                        metadata_base + t * aligned_dispatched_metadata_page_size);
                }
            }

            if (batch_did_local_write) {
                noc_async_write_barrier();
            }

            if (has_next_batch) {
                noc_async_read_barrier();
            }
        }
    }

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
