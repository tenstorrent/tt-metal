// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Prefill combine reader kernel
// This kernel reads expert outputs and metadata, then routes them back to original token positions

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/dprint.h"
#include "tt_metal/tools/profiler/kernel_profiler.hpp"

// Debug print control - set to 0 to disable combine debug prints, 1 to enable
#define ENABLE_COMBINE_DEBUG 0
#if ENABLE_COMBINE_DEBUG
#define DPRINT_COMBINE DPRINT
#else
#define DPRINT_COMBINE \
    if (0)             \
    DebugPrinter()
#endif

void kernel_main() {
    // ===== Compile Time Args =====
    // CB IDs (indices 0-3)
    constexpr uint32_t cb_dispatched_buffer_id = get_compile_time_arg_val(0);
    constexpr uint32_t cb_dispatched_metadata_id = get_compile_time_arg_val(1);
    constexpr uint32_t cb_experts_tok_counter_id = get_compile_time_arg_val(2);
    constexpr uint32_t cb_output_id = get_compile_time_arg_val(3);

    // Page counts (indices 4-7)
    constexpr uint32_t dispatched_buffer_pages = get_compile_time_arg_val(4);
    constexpr uint32_t dispatched_metadata_pages = get_compile_time_arg_val(5);
    constexpr uint32_t experts_tok_counter_pages = get_compile_time_arg_val(6);
    constexpr uint32_t output_pages = get_compile_time_arg_val(7);

    // Page sizes (indices 8-11)
    constexpr uint32_t dispatched_buffer_page_size = get_compile_time_arg_val(8);
    constexpr uint32_t dispatched_metadata_page_size = get_compile_time_arg_val(9);
    constexpr uint32_t experts_tok_counter_page_size = get_compile_time_arg_val(10);
    constexpr uint32_t output_page_size = get_compile_time_arg_val(11);

    // Operation parameters (indices 12-16)
    constexpr uint32_t num_chips = get_compile_time_arg_val(12);
    constexpr uint32_t experts_per_chip = get_compile_time_arg_val(13);
    constexpr uint32_t num_experts_per_tok = get_compile_time_arg_val(14);
    constexpr uint32_t seq_len_per_chip = get_compile_time_arg_val(15);
    constexpr uint32_t max_dispatched_tokens_per_expert = get_compile_time_arg_val(16);

    // Hidden dimension (index 17)
    constexpr uint32_t hidden_size = get_compile_time_arg_val(17);

    // Aligned page sizes (indices 18-21)
    constexpr uint32_t aligned_dispatched_buffer_page_size = get_compile_time_arg_val(18);
    constexpr uint32_t aligned_dispatched_metadata_page_size = get_compile_time_arg_val(19);
    constexpr uint32_t aligned_experts_tok_counter_page_size = get_compile_time_arg_val(20);
    constexpr uint32_t aligned_output_page_size = get_compile_time_arg_val(21);

    // Mesh information (indices 22-26) - reader doesn't use these, but they're in the args
    // constexpr uint32_t src_mesh_id = get_compile_time_arg_val(22);
    // constexpr uint32_t src_chip_id = get_compile_time_arg_val(23);
    // constexpr uint32_t mesh_rows = get_compile_time_arg_val(24);
    // constexpr uint32_t mesh_cols = get_compile_time_arg_val(25);
    // constexpr uint32_t linearized_mesh_coord = get_compile_time_arg_val(26);

    // Fabric configuration (indices 27-30) - reader doesn't use these, but they're in the args
    // constexpr uint32_t fabric_max_packet_size = get_compile_time_arg_val(27);
    // constexpr uint32_t l1_alignment = get_compile_time_arg_val(28);
    // constexpr uint32_t num_links = get_compile_time_arg_val(29);
    // constexpr tt::tt_fabric::Topology topology = (tt::tt_fabric::Topology)get_compile_time_arg_val(30);

    // TensorAccessorArgs for all 4 tensors (starting at index 31)
    constexpr auto dispatched_buffer_args = TensorAccessorArgs<31>();
    constexpr auto dispatched_metadata_args =
        TensorAccessorArgs<dispatched_buffer_args.next_compile_time_args_offset()>();
    constexpr auto experts_tok_counter_args =
        TensorAccessorArgs<dispatched_metadata_args.next_compile_time_args_offset()>();
    constexpr auto output_args = TensorAccessorArgs<experts_tok_counter_args.next_compile_time_args_offset()>();

    // ===== Runtime Args =====
    uint32_t rt_args = 0;
    uint32_t dispatched_buffer_addr = get_arg_val<uint32_t>(rt_args++);
    uint32_t dispatched_metadata_addr = get_arg_val<uint32_t>(rt_args++);
    uint32_t experts_tok_counter_addr = get_arg_val<uint32_t>(rt_args++);
    uint32_t output_addr = get_arg_val<uint32_t>(rt_args++);
    uint32_t expert_start_idx = get_arg_val<uint32_t>(rt_args++);
    uint32_t expert_end_idx = get_arg_val<uint32_t>(rt_args++);

    DPRINT_COMBINE << "Combine Reader: experts=[" << expert_start_idx << "," << expert_end_idx << ")"
                   << " experts_per_chip=" << experts_per_chip
                   << " max_dispatched_tokens_per_expert=" << max_dispatched_tokens_per_expert
                   << " hidden_size=" << hidden_size << ENDL();

    {
        DeviceZoneScopedN("combine-read-counters");
        const auto experts_tok_counter_addr_gen =
            TensorAccessor(experts_tok_counter_args, experts_tok_counter_addr, aligned_experts_tok_counter_page_size);
        for (uint32_t i = 0; i < experts_tok_counter_pages; i++) {
            DPRINT_COMBINE << "Fetching experts token counter; page=" << i << ENDL();
            cb_reserve_back(cb_experts_tok_counter_id, 1);
            uint32_t l1_write_addr = get_write_ptr(cb_experts_tok_counter_id);
            noc_async_read_page(i, experts_tok_counter_addr_gen, l1_write_addr);
        }
        noc_async_read_barrier();
        cb_push_back(cb_experts_tok_counter_id, experts_tok_counter_pages);
    }

    const auto dispatched_buffer_addr_gen =
        TensorAccessor(dispatched_buffer_args, dispatched_buffer_addr, aligned_dispatched_buffer_page_size);
    const auto dispatched_metadata_addr_gen =
        TensorAccessor(dispatched_metadata_args, dispatched_metadata_addr, aligned_dispatched_metadata_page_size);

    constexpr auto expert_stride = max_dispatched_tokens_per_expert;

    uint32_t experts_tok_counter_l1_addr = get_read_ptr(cb_experts_tok_counter_id);
    volatile tt_l1_ptr uint32_t* experts_tok_counter_l1 =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(experts_tok_counter_l1_addr);

    {
        constexpr uint32_t read_batch_size = 8;

        const uint32_t buf_fifo_limit = get_local_cb_interface(cb_dispatched_buffer_id).fifo_limit;
        const uint32_t buf_fifo_size = get_local_cb_interface(cb_dispatched_buffer_id).fifo_size;
        const uint32_t meta_fifo_limit = get_local_cb_interface(cb_dispatched_metadata_id).fifo_limit;
        const uint32_t meta_fifo_size = get_local_cb_interface(cb_dispatched_metadata_id).fifo_size;

        DeviceZoneScopedN("combine-read-expert-data");
        for (uint32_t local_expert = expert_start_idx; local_expert < expert_end_idx; local_expert++) {
            DPRINT_COMBINE << "Local expert=" << local_expert << ENDL();
            uint32_t start_page = local_expert * expert_stride;
            uint32_t expert_tokens = experts_tok_counter_l1[local_expert];
            uint32_t end_page = start_page + expert_tokens;

            DPRINT_COMBINE << "  Tokens for expert: " << expert_tokens << " (pages [" << start_page << "," << end_page
                           << "))" << ENDL();

            for (uint32_t token = start_page; token < end_page; token += read_batch_size) {
                uint32_t batch_end = (token + read_batch_size < end_page) ? token + read_batch_size : end_page;
                uint32_t batch_count = batch_end - token;

                {
                    DeviceZoneScopedN("combine-read-wait-writer");
                    cb_reserve_back(cb_dispatched_buffer_id, batch_count);
                    cb_reserve_back(cb_dispatched_metadata_id, batch_count);
                }

                {
                    DeviceZoneScopedN("combine-read-dram");
                    uint32_t buffer_base = get_write_ptr(cb_dispatched_buffer_id);
                    uint32_t metadata_base = get_write_ptr(cb_dispatched_metadata_id);
                    for (uint32_t t = 0; t < batch_count; t++) {
                        uint32_t l1_buffer_write_addr = buffer_base + t * aligned_dispatched_buffer_page_size;
                        if (l1_buffer_write_addr >= buf_fifo_limit) {
                            l1_buffer_write_addr -= buf_fifo_size;
                        }
                        noc_async_read_page(token + t, dispatched_buffer_addr_gen, l1_buffer_write_addr);
                        uint32_t l1_metadata_write_addr = metadata_base + t * aligned_dispatched_metadata_page_size;
                        if (l1_metadata_write_addr >= meta_fifo_limit) {
                            l1_metadata_write_addr -= meta_fifo_size;
                        }
                        noc_async_read_page(token + t, dispatched_metadata_addr_gen, l1_metadata_write_addr);
                    }
                    noc_async_read_barrier();
                }

                uint32_t buf_pages_until_wrap =
                    (buf_fifo_limit - get_write_ptr(cb_dispatched_buffer_id)) / aligned_dispatched_buffer_page_size;
                if (batch_count <= buf_pages_until_wrap) {
                    cb_push_back(cb_dispatched_buffer_id, batch_count);
                } else {
                    cb_push_back(cb_dispatched_buffer_id, buf_pages_until_wrap);
                    cb_push_back(cb_dispatched_buffer_id, batch_count - buf_pages_until_wrap);
                }

                uint32_t meta_pages_until_wrap = (meta_fifo_limit - get_write_ptr(cb_dispatched_metadata_id)) /
                                                 aligned_dispatched_metadata_page_size;
                if (batch_count <= meta_pages_until_wrap) {
                    cb_push_back(cb_dispatched_metadata_id, batch_count);
                } else {
                    cb_push_back(cb_dispatched_metadata_id, meta_pages_until_wrap);
                    cb_push_back(cb_dispatched_metadata_id, batch_count - meta_pages_until_wrap);
                }
            }
        }
    }
}
