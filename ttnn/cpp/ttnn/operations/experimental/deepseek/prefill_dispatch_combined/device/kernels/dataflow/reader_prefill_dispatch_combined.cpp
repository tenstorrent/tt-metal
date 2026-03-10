// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Prefill dispatch reader kernel (combined metadata+payload variant)
// Writes payload at an offset within the combined CB page, leaving space
// for the writer to prepend metadata before a single fabric/DRAM transfer.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/dprint.h"
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

void kernel_main() {
    // ===== Compile Time Args =====
    // CB IDs (indices 0-4)
    constexpr uint32_t cb_combined_id = get_compile_time_arg_val(0);
    constexpr uint32_t cb_indices_id = get_compile_time_arg_val(1);
    constexpr uint32_t cb_weights_id = get_compile_time_arg_val(2);
    constexpr uint32_t cb_offsets_id = get_compile_time_arg_val(3);
    constexpr uint32_t cb_packet_header_id = get_compile_time_arg_val(4);

    // Page counts (indices 5-10)
    constexpr uint32_t input_pages = get_compile_time_arg_val(5);
    constexpr uint32_t indices_pages = get_compile_time_arg_val(6);
    constexpr uint32_t weights_pages = get_compile_time_arg_val(7);
    constexpr uint32_t offsets_pages = get_compile_time_arg_val(8);
    constexpr uint32_t combined_output_pages = get_compile_time_arg_val(9);
    constexpr uint32_t counter_pages = get_compile_time_arg_val(10);

    // Page sizes (indices 11-16)
    constexpr uint32_t input_page_size = get_compile_time_arg_val(11);
    constexpr uint32_t indices_page_size = get_compile_time_arg_val(12);
    constexpr uint32_t weights_page_size = get_compile_time_arg_val(13);
    constexpr uint32_t offsets_page_size = get_compile_time_arg_val(14);
    constexpr uint32_t combined_output_page_size = get_compile_time_arg_val(15);
    constexpr uint32_t counter_page_size = get_compile_time_arg_val(16);

    // Operation parameters (indices 17-25)
    constexpr uint32_t num_devices = get_compile_time_arg_val(17);
    constexpr uint32_t hidden_size = get_compile_time_arg_val(18);
    constexpr uint32_t experts_per_chip = get_compile_time_arg_val(19);
    constexpr uint32_t n_routed_experts = get_compile_time_arg_val(20);
    constexpr uint32_t num_experts_per_tok = get_compile_time_arg_val(21);
    constexpr uint32_t metadata_len = get_compile_time_arg_val(22);
    constexpr uint32_t max_dispatched_tokens_per_expert = get_compile_time_arg_val(23);
    constexpr uint32_t tokens_per_device = get_compile_time_arg_val(24);
    constexpr uint32_t padded_metadata_bytes = get_compile_time_arg_val(25);

    // Mesh information (indices 26-30)
    constexpr uint32_t src_mesh_id = get_compile_time_arg_val(26);
    constexpr uint32_t src_chip_id = get_compile_time_arg_val(27);
    constexpr uint32_t mesh_rows = get_compile_time_arg_val(28);
    constexpr uint32_t mesh_cols = get_compile_time_arg_val(29);
    constexpr uint32_t linearized_mesh_coord = get_compile_time_arg_val(30);

    // Aligned page sizes (indices 31-36)
    constexpr uint32_t aligned_input_page_size = get_compile_time_arg_val(31);
    constexpr uint32_t aligned_indices_page_size = get_compile_time_arg_val(32);
    constexpr uint32_t aligned_weights_page_size = get_compile_time_arg_val(33);
    constexpr uint32_t aligned_offsets_page_size = get_compile_time_arg_val(34);
    constexpr uint32_t aligned_combined_output_page_size = get_compile_time_arg_val(35);
    constexpr uint32_t aligned_counter_page_size = get_compile_time_arg_val(36);

    // Fabric configuration (indices 37-40)
    constexpr uint32_t fabric_max_packet_size = get_compile_time_arg_val(37);
    constexpr uint32_t l1_alignment = get_compile_time_arg_val(38);
    constexpr uint32_t num_links = get_compile_time_arg_val(39);
    constexpr tt::tt_fabric::Topology topology = (tt::tt_fabric::Topology)get_compile_time_arg_val(40);

    // Combined CB page = metadata padding + payload
    constexpr uint32_t combined_cb_page_size = padded_metadata_bytes + aligned_input_page_size;

    // TensorAccessorArgs for 6 tensors (starting at index 41)
    constexpr auto input_args = TensorAccessorArgs<41>();
    constexpr auto indices_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
    constexpr auto weights_args = TensorAccessorArgs<indices_args.next_compile_time_args_offset()>();
    constexpr auto offsets_args = TensorAccessorArgs<weights_args.next_compile_time_args_offset()>();
    constexpr auto combined_output_args = TensorAccessorArgs<offsets_args.next_compile_time_args_offset()>();
    constexpr auto counter_args = TensorAccessorArgs<combined_output_args.next_compile_time_args_offset()>();

    // ===== Runtime Args =====
    uint32_t rt_args = 0;
    uint32_t input_tensor_address = get_arg_val<uint32_t>(rt_args++);
    uint32_t indices_tensor_address = get_arg_val<uint32_t>(rt_args++);
    uint32_t weights_tensor_address = get_arg_val<uint32_t>(rt_args++);
    uint32_t offsets_tensor_address = get_arg_val<uint32_t>(rt_args++);
    uint32_t combined_output_address = get_arg_val<uint32_t>(rt_args++);
    uint32_t counter_tensor_address = get_arg_val<uint32_t>(rt_args++);
    uint32_t cross_device_semaphore_address = get_arg_val<uint32_t>(rt_args++);
    uint32_t init_semaphore_address = get_arg_val<uint32_t>(rt_args++);
    uint32_t token_start_idx = get_arg_val<uint32_t>(rt_args++);
    uint32_t token_end_idx = get_arg_val<uint32_t>(rt_args++);

    DPRINT_DISPATCH << "Reader combined kernel: tokens=[" << token_start_idx << "," << token_end_idx << ")"
                    << " hidden_size=" << hidden_size << " padded_metadata_bytes=" << padded_metadata_bytes
                    << " combined_cb_page_size=" << combined_cb_page_size << ENDL();

    // Read offsets tensor
    {
        DeviceZoneScopedN("dispatch-combined-read-offsets");
        const auto offsets_addr_gen = TensorAccessor(offsets_args, offsets_tensor_address, offsets_page_size);
        for (uint32_t i = 0; i < offsets_pages; i++) {
            cb_reserve_back(cb_offsets_id, 1);
            uint32_t l1_write_addr = get_write_ptr(cb_offsets_id);
            noc_async_read_page(i, offsets_addr_gen, l1_write_addr);
        }
        noc_async_read_barrier();
        cb_push_back(cb_offsets_id, offsets_pages);
    }

    // Read input/indices/weights tokens in batches
    {
        constexpr uint32_t read_batch_size = 8;

        const uint32_t indices_fifo_limit = get_local_cb_interface(cb_indices_id).fifo_limit;
        const uint32_t indices_fifo_size = get_local_cb_interface(cb_indices_id).fifo_size;
        const uint32_t weights_fifo_limit = get_local_cb_interface(cb_weights_id).fifo_limit;
        const uint32_t weights_fifo_size = get_local_cb_interface(cb_weights_id).fifo_size;
        const uint32_t combined_fifo_limit = get_local_cb_interface(cb_combined_id).fifo_limit;
        const uint32_t combined_fifo_size = get_local_cb_interface(cb_combined_id).fifo_size;

        DeviceZoneScopedN("dispatch-combined-read-tokens");
        const auto input_addr_gen = TensorAccessor(input_args, input_tensor_address, aligned_input_page_size);
        const auto indices_addr_gen = TensorAccessor(indices_args, indices_tensor_address, aligned_indices_page_size);
        const auto weights_addr_gen = TensorAccessor(weights_args, weights_tensor_address, aligned_weights_page_size);

        for (uint32_t token = token_start_idx; token < token_end_idx; token += read_batch_size) {
            uint32_t batch_end = (token + read_batch_size < token_end_idx) ? token + read_batch_size : token_end_idx;
            uint32_t batch_count = batch_end - token;

            {
                DeviceZoneScopedN("dispatch-combined-read-wait-writer");
                cb_reserve_back(cb_indices_id, batch_count);
                cb_reserve_back(cb_weights_id, batch_count);
                cb_reserve_back(cb_combined_id, batch_count);
            }

            {
                DeviceZoneScopedN("dispatch-combined-read-dram");
                uint32_t indices_base = get_write_ptr(cb_indices_id);
                uint32_t weights_base = get_write_ptr(cb_weights_id);
                uint32_t combined_base = get_write_ptr(cb_combined_id);

                for (uint32_t t = 0; t < batch_count; t++) {
                    uint32_t indices_addr = indices_base + t * aligned_indices_page_size;
                    if (indices_addr >= indices_fifo_limit) {
                        indices_addr -= indices_fifo_size;
                    }
                    noc_async_read_page(token + t, indices_addr_gen, indices_addr);

                    uint32_t weights_addr = weights_base + t * aligned_weights_page_size;
                    if (weights_addr >= weights_fifo_limit) {
                        weights_addr -= weights_fifo_size;
                    }
                    noc_async_read_page(token + t, weights_addr_gen, weights_addr);

                    // Write payload at offset padded_metadata_bytes within the combined CB page
                    uint32_t combined_page_addr = combined_base + t * combined_cb_page_size;
                    if (combined_page_addr >= combined_fifo_limit) {
                        combined_page_addr -= combined_fifo_size;
                    }
                    uint32_t payload_addr = combined_page_addr + padded_metadata_bytes;
                    noc_async_read_page(token + t, input_addr_gen, payload_addr);
                }
                noc_async_read_barrier();
            }

            auto split_push_back = [](uint32_t cb_id, uint32_t count, uint32_t fifo_limit, uint32_t page_size) {
                uint32_t pages_until_wrap = (fifo_limit - get_write_ptr(cb_id)) / page_size;
                if (count <= pages_until_wrap) {
                    cb_push_back(cb_id, count);
                } else {
                    cb_push_back(cb_id, pages_until_wrap);
                    cb_push_back(cb_id, count - pages_until_wrap);
                }
            };

            split_push_back(cb_indices_id, batch_count, indices_fifo_limit, aligned_indices_page_size);
            split_push_back(cb_weights_id, batch_count, weights_fifo_limit, aligned_weights_page_size);
            split_push_back(cb_combined_id, batch_count, combined_fifo_limit, combined_cb_page_size);
        }
    }
}
